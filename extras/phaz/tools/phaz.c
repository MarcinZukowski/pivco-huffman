/* phaz — single CLI for the pivco-Huffman-ANS entropy transplant onto zstd.
 *
 * Keeps zstd's LZ parse + copy engine; replaces the entropy layer (FSE seq
 * codes + literal-Huffman) with pivco-Huffman/PHA over the pivoted ll/ml/of/lit
 * streams.  Links the patched libzstd (capture hook + ZSTD_phazDecode) and
 * pivco-huffman's libpivco_huffman (PH/PHA stream codec).
 *
 * Commands:
 *   c  IN [OUT]            compress IN -> OUT (default IN.phaz)
 *   d  IN [OUT]            decompress a .phaz -> OUT (default IN sans .phaz)
 *   stats IN               compress in-memory: size vs stock zstd + fused decode timing
 *   dump  IN OUTDIR        debug: write the raw pivoted streams + meta.txt
 *   profile parse|litcost IN   profile stock zstd via its public API
 *   (-l N sets the zstd level, default 3 = zstd's own default; -h for help)
 *
 * The .phaz container is host-endian and deliberately hacky (research tool, not
 * a stable format): [magic"phaz"+ver][n nseq lits extrabits nblk u64][bns,btl
 * u32][xblen u64 + xb][per stream: u8 method(0=raw,1=PH/PHA) + u64 len + bytes].
 */
#define ZSTD_STATIC_LINKING_ONLY
#include <zstd.h>
#include "pivcohuf_file.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <time.h>

/* Buffer codec + shared helpers (capture hook globals, ZSTD_phazDecode,
 * phaz_compress/phaz_decompress, phaz_capture_run, phaz_pack_stream). */
#include "phaz_codec.h"

static double now(void){ struct timespec t; clock_gettime(CLOCK_MONOTONIC,&t);
    return t.tv_sec + t.tv_nsec*1e-9; }

static unsigned char *rd_file(const char *p, size_t *n){
    FILE *f=fopen(p,"rb"); if(!f){perror(p);exit(2);}
    fseek(f,0,SEEK_END); long s=ftell(f); fseek(f,0,SEEK_SET);
    unsigned char *b=malloc((size_t)(s?s:1)+64);
    if(fread(b,1,(size_t)s,f)!=(size_t)s){perror(p);exit(2);}
    fclose(f); *n=(size_t)s; return b; }
static void wr_file(const char *p, const void *d, size_t n){
    FILE *f=fopen(p,"wb"); if(!f){perror(p);exit(2);}
    if(fwrite(d,1,n,f)!=n){perror(p);exit(2);} fclose(f); }


/* ====================== commands ====================== */

static int cmd_c(const char *in, const char *out, int level){
    size_t n; unsigned char *src=rd_file(in,&n);
    size_t cap=phaz_compress_bound(n); unsigned char *buf=malloc(cap);
    if(!buf){fprintf(stderr,"phaz: oom\n");return 2;}
    phaz_stats st; memset(&st,0,sizeof st);
    double t0=now();
    size_t osz=phaz_compress(src,n,buf,cap,level,&st);
    double tot=now()-t0;
    if(!osz){fprintf(stderr,"phaz: compress failed\n");return 2;}
    wr_file(out,buf,osz);
    double t_pk=st.pack_ms[0]+st.pack_ms[1]+st.pack_ms[2]+st.pack_ms[3];
    printf("%s -> %s  %zu -> %zu  ratio %.3f\n",in,out,n,osz,(double)n/osz);
    printf("  compress %.2f ms (%.1f MB/s)  [zstd-parse+capture %.2f ms, PH-encode %.2f ms]\n",
           tot*1e3, n/(tot*1e6), st.capture_ms, t_pk);
    for(int i=0;i<4;i++)
        printf("    %-3s %8zu -> %8zu  %.3f ms (%.2f GB/s)\n",
               phaz_stream_names[i],st.stream_raw[i],st.stream_enc[i],
               st.pack_ms[i], st.stream_raw[i]/(st.pack_ms[i]*1e6));
    free(src); free(buf);
    return 0;
}

static int cmd_d(const char *in, const char *out){
    size_t fn; unsigned char *buf=rd_file(in,&fn);
    if(fn<5+8){fprintf(stderr,"%s: not a phaz container\n",in); return 2;}
    uint64_t n; memcpy(&n, buf+5, 8);       /* hdr[0] = original size */
    unsigned char *dst=malloc(n+64);
    if(!dst){fprintf(stderr,"phaz: oom\n");return 2;}
    phaz_stats st; memset(&st,0,sizeof st);
    double t0=now();
    size_t got=phaz_decompress(buf,fn,dst,n+64,&st);
    double tot=now()-t0;
    if(got!=n){fprintf(stderr,"%s: decode failed (got %zu, expected %llu)\n",
                       in,got,(unsigned long long)n); return 3;}
    wr_file(out,dst,n);
    double t_ent=st.entropy_ms[0]+st.entropy_ms[1]+st.entropy_ms[2]+st.entropy_ms[3];
    printf("%s -> %s  %zu bytes\n",in,out,(size_t)n);
    printf("  decode %.2f ms (%.1f MB/s)  [PH-entropy %.2f ms, reconstruct+copy %.2f ms]\n",
           tot*1e3, n/(tot*1e6), t_ent, st.reconstruct_ms);
    for(int i=0;i<4;i++)
        printf("    %-3s %8zu  %.3f ms (%.2f GB/s)\n",
               phaz_stream_names[i], st.stream_raw[i],
               st.entropy_ms[i], st.stream_raw[i]/(st.entropy_ms[i]*1e6));
    free(buf); free(dst);
    return 0;
}

static int cmd_stats(const char *in, int level){
    size_t n; unsigned char *src=rd_file(in,&n);
    size_t zsize=phaz_capture_run(src,n,level,1);
    if(zsize==(size_t)-1){fprintf(stderr,"phaz: capture failed\n");return 2;}
    size_t nseq=g_phaz_nseq, lits=g_phaz_lits;
    size_t s_ll,s_ml,s_of,s_lit;
    phaz_pack_stream(NULL,NULL,g_phaz_llc,nseq,&s_ll,0);
    phaz_pack_stream(NULL,NULL,g_phaz_mlc,nseq,&s_ml,0);
    phaz_pack_stream(NULL,NULL,g_phaz_ofc,nseq,&s_of,0);
    phaz_pack_stream(NULL,NULL,g_phaz_lit,lits,&s_lit,0);
    size_t s_xb=(g_phaz_xbpos+7)/8;
    size_t our=s_ll+s_ml+s_of+s_lit+s_xb;
    printf("%-22s n=%zu nseq=%zu lits=%zu nblk=%zu\n",in,n,nseq,lits,g_phaz_nblk);
    printf("  zstd-%-2d      %9zu  ratio %.3f\n",level,zsize,(double)n/zsize);
    printf("  phaz         %9zu  ratio %.3f   %+.1f%% vs zstd\n",
           our,(double)n/our,100.0*((double)our-zsize)/zsize);
    printf("    ll %zu  ml %zu  of %zu  lit %zu  xbits %zu\n",
           s_ll,s_ml,s_of,s_lit,s_xb);

    /* fused decode timing: PHA-decode streams + reconstruct/exec, vs stock zstd.
     * Bound must cover the LARGEST stream (ll/ml/of are nseq bytes, lit is lits). */
    size_t big=nseq>lits?nseq:lits; if(!big)big=1;
    size_t bound=pivcohuf_compress_bound(big);
    unsigned char *cll=malloc(bound),*cml=malloc(bound),*cof=malloc(bound),*clit=malloc(bound);
    size_t ll0=bound,ml0=bound,of0=bound,lit0=bound;
    int o0 = pivcohuf_compress_ex(g_phaz_llc,nseq,cll,&ll0,1)==PIVCOHUF_OK;
    int o1 = pivcohuf_compress_ex(g_phaz_mlc,nseq,cml,&ml0,1)==PIVCOHUF_OK;
    int o2 = pivcohuf_compress_ex(g_phaz_ofc,nseq,cof,&of0,1)==PIVCOHUF_OK;
    int o3 = pivcohuf_compress_ex(g_phaz_lit,lits,clit,&lit0,1)==PIVCOHUF_OK;
    if(!(o0&&o1&&o2&&o3)){ fprintf(stderr,"phaz: stats stream re-compress failed\n"); return 2; }
    unsigned char *rl=malloc(nseq+64),*rm=malloc(nseq+64),*ro=malloc(nseq+64),*rt=malloc(lits+64);
    unsigned char *dst=malloc(n+64);
    int reps=200; double phaz=1e30,zstd=1e30;
    for(int r=0;r<reps;r++){ double t=now();
        size_t g; g=nseq; pivcohuf_decompress(cll,ll0,rl,&g);
        g=nseq; pivcohuf_decompress(cml,ml0,rm,&g); g=nseq; pivcohuf_decompress(cof,of0,ro,&g);
        g=lits; pivcohuf_decompress(clit,lit0,rt,&g);
        size_t got=ZSTD_phazDecode(dst,n+64,rl,rm,ro,g_phaz_xb,rt,lits,g_phaz_blk_ns,g_phaz_blk_tl,g_phaz_blk_cf,g_phaz_nblk);
        if(got!=n){fprintf(stderr,"phaz: stats decode got %zu != %zu\n",got,n);return 2;}
        double dt=now()-t; if(dt<phaz)phaz=dt; }
    size_t cb=ZSTD_compressBound(n); unsigned char *cz=malloc(cb);
    size_t cz_l=ZSTD_compress(cz,cb,src,n,level);
    for(int r=0;r<reps;r++){ double t=now(); ZSTD_decompress(dst,n+64,cz,cz_l);
        double dt=now()-t; if(dt<zstd)zstd=dt; }
    printf("  decode: phaz %.3f ms (%.2f GB/s)  zstd-%d %.3f ms (%.2f GB/s)  %.2fx\n",
           phaz*1e3, n/(phaz*1e9), level, zstd*1e3, n/(zstd*1e9), zstd/phaz);

    /* DIAG: old-style "isolated" timing (each entropy stream best-of-N separately,
     * summed, + reconstruct best-of-N separately) -- replicates ~/src/phaz. */
    double bll=1e30,bml=1e30,bof=1e30,bli=1e30,brec=1e30; size_t g;
    for(int r=0;r<reps;r++){double t=now();g=nseq;pivcohuf_decompress(cll,ll0,rl,&g);double d=now()-t;if(d<bll)bll=d;}
    for(int r=0;r<reps;r++){double t=now();g=nseq;pivcohuf_decompress(cml,ml0,rm,&g);double d=now()-t;if(d<bml)bml=d;}
    for(int r=0;r<reps;r++){double t=now();g=nseq;pivcohuf_decompress(cof,of0,ro,&g);double d=now()-t;if(d<bof)bof=d;}
    for(int r=0;r<reps;r++){double t=now();g=lits;pivcohuf_decompress(clit,lit0,rt,&g);double d=now()-t;if(d<bli)bli=d;}
    for(int r=0;r<reps;r++){double t=now();ZSTD_phazDecode(dst,n+64,rl,rm,ro,g_phaz_xb,rt,lits,g_phaz_blk_ns,g_phaz_blk_tl,g_phaz_blk_cf,g_phaz_nblk);double d=now()-t;if(d<brec)brec=d;}
    double iso=bll+bml+bof+bli+brec;
    printf("  DIAG isolated(old-style): phaz %.3f ms (%.2f GB/s) %.2fx  [ent %.3f + rec %.3f]\n",
           iso*1e3, n/(iso*1e9), zstd/iso, (bll+bml+bof+bli)*1e3, brec*1e3);
    return 0;
}

static int cmd_dump(const char *in, const char *dir, int level){
    size_t n; unsigned char *src=rd_file(in,&n);
    size_t zsize=phaz_capture_run(src,n,level,1);
    if(zsize==(size_t)-1){fprintf(stderr,"phaz: capture failed\n");return 2;}
    char p[1024];
#define WR(nm,d,len) do{ snprintf(p,sizeof p,"%s/%s",dir,nm); \
    FILE*f=fopen(p,"wb"); if(!f){perror(p);exit(2);} fwrite((d),1,(len),f); fclose(f);}while(0)
    WR("ll",g_phaz_llc,g_phaz_nseq); WR("ml",g_phaz_mlc,g_phaz_nseq);
    WR("of",g_phaz_ofc,g_phaz_nseq); WR("lit",g_phaz_lit,g_phaz_lits);
    WR("xb",g_phaz_xb,(size_t)((g_phaz_xbpos+7)/8));
#undef WR
    snprintf(p,sizeof p,"%s/blocks",dir); FILE*bf=fopen(p,"wb"); if(!bf){perror(p);exit(2);}
    fwrite(&g_phaz_nblk,sizeof(size_t),1,bf);
    fwrite(g_phaz_blk_ns,sizeof(unsigned),g_phaz_nblk,bf);
    fwrite(g_phaz_blk_tl,sizeof(unsigned),g_phaz_nblk,bf); fclose(bf);
    snprintf(p,sizeof p,"%s/meta.txt",dir); FILE*m=fopen(p,"w"); if(!m){perror(p);exit(2);}
    fprintf(m,"%zu %zu %zu %llu %zu\n",n,g_phaz_nseq,g_phaz_lits,g_phaz_extrabits,zsize);
    fclose(m);
    printf("dumped %s -> %s/ (nseq=%zu lits=%zu nblk=%zu)\n",in,dir,g_phaz_nseq,g_phaz_lits,g_phaz_nblk);
    return 0;
}

/* ---- profile (stock zstd via public API) ---- */
static int mode_parse(const char *path,int level){
    size_t n; unsigned char *src=rd_file(path,&n);
    ZSTD_CCtx *zc=ZSTD_createCCtx(); ZSTD_CCtx_setParameter(zc,ZSTD_c_compressionLevel,level);
    size_t cap=ZSTD_sequenceBound(n); ZSTD_Sequence *seq=malloc(cap*sizeof(*seq));
    size_t ns=ZSTD_generateSequences(zc,seq,cap,src,n);
    if(ZSTD_isError(ns)){fprintf(stderr,"generateSequences: %s\n",ZSTD_getErrorName(ns));return 2;}
    unsigned long long real=0,omax=0,reps=0,o16=0,o18=0,o24=0,llcap=0,mlcap=0;
    for(size_t i=0;i<ns;i++){ unsigned o=seq[i].offset,ll=seq[i].litLength,ml=seq[i].matchLength;
        if(o==0&&ml==0)continue; real++; if(seq[i].rep)reps++; if(o>omax)omax=o;
        if(o>=(1u<<16))o16++; if(o>=(1u<<18))o18++; if(o>=(1u<<24))o24++;
        if(ll>16)llcap++; if(ml>64)mlcap++; }
    printf("%-26s n=%zu lvl=%d\n",path,n,level);
    printf("  zstd parse: nseq=%llu  bytes/seq=%.1f  reps=%.1f%%\n",real,(double)n/real,100.0*reps/real);
    printf("  offsets:   max=%llu (%.0f KB)  >16bit=%.1f%%  >18bit=%.1f%%  >24bit=%.1f%%\n",
           omax,omax/1024.0,100.0*o16/real,100.0*o18/real,100.0*o24/real);
    printf("  cap exceed: ll>16=%.1f%%  ml>64=%.1f%%\n",100.0*llcap/real,100.0*mlcap/real);
    return 0;
}
static double bestdec(void *d,size_t raw,const void *c,size_t cl,int reps){
    double best=1e30; for(int r=0;r<reps;r++){ double t=now();
        size_t g=ZSTD_decompress(d,raw,c,cl); if(ZSTD_isError(g)||g!=raw)return -1;
        double dt=now()-t; if(dt<best)best=dt; } return best; }
static int mode_litcost(const char *path,int level){
    size_t n; unsigned char *src=rd_file(path,&n); void *dst=malloc(n+64);
    size_t bound=ZSTD_compressBound(n); unsigned char *c=malloc(bound);
    ZSTD_CCtx *cc=ZSTD_createCCtx(); int reps=40;
    int modes[2]={ZSTD_lcm_huffman,ZSTD_lcm_uncompressed}; const char *nm[2]={"huffman","uncompressed"};
    double mbps[2]={0,0};
    printf("%s  (%zu bytes, level %d, best of %d)\n",path,n,level,reps);
    printf("%-16s %10s %7s %12s\n","literals","size","ratio","dec MB/s");
    for(int m=0;m<2;m++){ ZSTD_CCtx_reset(cc,ZSTD_reset_session_and_parameters);
        ZSTD_CCtx_setParameter(cc,ZSTD_c_compressionLevel,level);
        size_t rc=ZSTD_CCtx_setParameter(cc,ZSTD_c_literalCompressionMode,modes[m]);
        if(ZSTD_isError(rc))printf("  (literalCompressionMode unsupported: %s)\n",ZSTD_getErrorName(rc));
        size_t cl=ZSTD_compress2(cc,c,bound,src,n);
        if(ZSTD_isError(cl)){printf("compress err: %s\n",ZSTD_getErrorName(cl));continue;}
        double dt=bestdec(dst,n,c,cl,reps);
        if(dt<0||memcmp(dst,src,n)){printf("VERIFY FAIL\n");continue;}
        mbps[m]=n/dt/1e6; printf("%-16s %10zu %7.3f %12.0f\n",nm[m],cl,(double)n/cl,mbps[m]); }
    if(mbps[0]>0&&mbps[1]>0)
        printf("  -> literal-Huffman decode cost: %+.0f%% speed when removed\n",100*(mbps[1]-mbps[0])/mbps[0]);
    return 0;
}

/* ====================== CLI ====================== */
static void usage(FILE *f, const char *p){
    fprintf(f,
      "phaz — pivco-Huffman entropy transplant onto zstd (compress/decompress + analysis).\n"
      "usage: %s <command> [args] [-l LEVEL]\n"
      "  c  IN [OUT]                 compress  (OUT default IN.phaz)\n"
      "  d  IN [OUT]                 decompress a .phaz  (OUT default IN sans .phaz)\n"
      "  stats IN                    size vs stock zstd + fused decode timing\n"
      "  dump  IN OUTDIR             debug: write raw pivoted streams + meta.txt\n"
      "  profile parse|litcost IN    profile stock zstd via its public API\n"
      "  -l N, --level N             zstd level (default 3, same as the zstd CLI);  -h for this help\n", p);
}

int main(int argc, char **argv){
    int level=ZSTD_CLEVEL_DEFAULT;   /* 3 — same default as the zstd CLI */
    const char *pos[4]={0,0,0,0}; int np=0;
    for(int i=1;i<argc;i++){
        if(!strcmp(argv[i],"-h")||!strcmp(argv[i],"--help")){ usage(stdout,argv[0]); return 0; }
        if(!strcmp(argv[i],"-l")||!strcmp(argv[i],"--level")){
            if(i+1>=argc){fprintf(stderr,"%s: %s needs an argument\n",argv[0],argv[i]);return 2;}
            level=atoi(argv[++i]); continue; }
        if(argv[i][0]=='-'){ fprintf(stderr,"%s: unknown option '%s'\n",argv[0],argv[i]); usage(stderr,argv[0]); return 2; }
        if(np<4) pos[np++]=argv[i]; else { fprintf(stderr,"%s: too many arguments\n",argv[0]); return 2; }
    }
    if(np<1){ usage(stderr,argv[0]); return 1; }
    const char *cmd=pos[0];

    char obuf[1024];
    if(!strcmp(cmd,"c")){
        if(np<2){fprintf(stderr,"usage: %s c IN [OUT]\n",argv[0]);return 1;}
        const char *out=pos[2]; if(!out){ snprintf(obuf,sizeof obuf,"%s.phaz",pos[1]); out=obuf; }
        return cmd_c(pos[1],out,level);
    }
    if(!strcmp(cmd,"d")){
        if(np<2){fprintf(stderr,"usage: %s d IN [OUT]\n",argv[0]);return 1;}
        const char *out=pos[2];
        if(!out){
            size_t L=strlen(pos[1]);
            if(L>5 && !strcmp(pos[1]+L-5,".phaz")) snprintf(obuf,sizeof obuf,"%.*s",(int)(L-5),pos[1]);
            else                                   snprintf(obuf,sizeof obuf,"%s.out",pos[1]);
            out=obuf;
        }
        return cmd_d(pos[1],out);
    }
    if(!strcmp(cmd,"stats")){
        if(np<2){fprintf(stderr,"usage: %s stats IN\n",argv[0]);return 1;}
        return cmd_stats(pos[1],level);
    }
    if(!strcmp(cmd,"dump")){
        if(np<3){fprintf(stderr,"usage: %s dump IN OUTDIR\n",argv[0]);return 1;}
        return cmd_dump(pos[1],pos[2],level);
    }
    if(!strcmp(cmd,"profile")){
        if(np<3){fprintf(stderr,"usage: %s profile <parse|litcost> IN\n",argv[0]);return 1;}
        if(!strcmp(pos[1],"parse"))   return mode_parse(pos[2],level);
        if(!strcmp(pos[1],"litcost")) return mode_litcost(pos[2],level);
        fprintf(stderr,"%s: unknown profile mode '%s' (want parse|litcost)\n",argv[0],pos[1]); return 1;
    }
    fprintf(stderr,"%s: unknown command '%s'\n",argv[0],cmd); usage(stderr,argv[0]); return 1;
}
