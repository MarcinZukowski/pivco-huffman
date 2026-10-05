/* Duda's context binning on cross-field contexts of zstd's ll/ml/of codes.
 * Contexts are the other streams at the same sequence (decode order of -> ml -> ll)
 * or earlier sequences; the map merges them greedily into B bins by the smallest
 * entropy increase, and each bin is coded with a static table built on the data it
 * codes (table cost B x (alphabet+2)/2 bytes).  Rows per model and B:
 *   full     unmerged contexts, map and tables on the whole file (overfit lower bound)
 *   per-file map fit and costed on the whole file (what phaz transmits; map not charged)
 *   split    map fit on the first half of the sequences, cost on the second half
 *   pooled   map fit on the other files given, cost on this file
 *   map      fixed-width map bytes (raw stages and the nested stage)
 * usage: kbin DUMPDIR LEVEL B[,B...] file...   reads DUMPDIR/dumpLEVEL_file/{ll,ml,of}
 * (phaz dump output); ledger extras/order1/RESULTS.html section 27. */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>
#define A 64
static unsigned char *rdf(const char*path,size_t*n){ FILE*f=fopen(path,"rb"); if(!f){perror(path);exit(1);} fseek(f,0,SEEK_END); *n=ftell(f); fseek(f,0,SEEK_SET); unsigned char*b=malloc(*n+1); if(fread(b,1,*n,f)!=*n)exit(1); fclose(f); return b; }
static double cost(const uint32_t *c){ double n=0; for(int y=0;y<A;y++) n+=c[y]; if(!n) return 0; double b=n*log2(n); for(int y=0;y<A;y++) if(c[y]) b-=c[y]*log2((double)c[y]); return b; }
/* greedy merge of V contexts (counts cnt[V][A]) into B bins; returns bin id per context */
typedef struct { double d; int a,b; } cand_t;
static cand_t *heap; static size_t hn,hcap;
static void hpush(cand_t c){ if(hn==hcap){hcap=hcap?hcap*2:1<<16; heap=realloc(heap,hcap*sizeof*heap);} size_t i=hn++; while(i&&heap[(i-1)/2].d>c.d){ heap[i]=heap[(i-1)/2]; i=(i-1)/2; } heap[i]=c; }
static cand_t hpop(void){ cand_t r=heap[0]; cand_t last=heap[--hn]; size_t i=0; for(;;){ size_t l=2*i+1,rr=l+1,m=i; if(l<hn&&heap[l].d<(m==i?last.d:heap[m].d)) m=l; if(rr<hn&&heap[rr].d<(m==i?last.d:heap[m].d)) m=rr; if(m==i)break; heap[i]=heap[m]; i=m; } heap[i]=last; return r; }
static int bin_greedy(uint32_t *cnt, int V, int B, int *binof){
  /* nodes 0..V-1 are the contexts; merged nodes appended */
  int maxn=2*V; uint32_t *nc=calloc((size_t)maxn*A,sizeof*nc); double *cs=calloc(maxn,sizeof*cs); int *alive=calloc(maxn,sizeof*alive); int *parent=malloc(maxn*sizeof*parent);
  int nn=V, live=0; for(int v=0;v<V;v++){ memcpy(nc+(size_t)v*A,cnt+(size_t)v*A,A*sizeof*nc); cs[v]=cost(nc+(size_t)v*A); alive[v]=1; parent[v]=v; double s=0; for(int y=0;y<A;y++) s+=nc[(size_t)v*A+y]; if(s>0) live++; else { alive[v]=0; } }
  hn=0; for(int a=0;a<V;a++){ if(!alive[a])continue; for(int b=a+1;b<V;b++){ if(!alive[b])continue; uint32_t u[A]; for(int y=0;y<A;y++) u[y]=nc[(size_t)a*A+y]+nc[(size_t)b*A+y]; hpush((cand_t){cost(u)-cs[a]-cs[b],a,b}); } }
  while(live>B && hn){ cand_t c=hpop(); if(!alive[c.a]||!alive[c.b]) continue; int m=nn++; for(int y=0;y<A;y++) nc[(size_t)m*A+y]=nc[(size_t)c.a*A+y]+nc[(size_t)c.b*A+y]; cs[m]=cost(nc+(size_t)m*A); alive[c.a]=alive[c.b]=0; parent[c.a]=parent[c.b]=m; parent[m]=m; alive[m]=1; live--;
    for(int o=0;o<nn-1;o++){ if(!alive[o])continue; uint32_t u[A]; for(int y=0;y<A;y++) u[y]=nc[(size_t)m*A+y]+nc[(size_t)o*A+y]; hpush((cand_t){cost(u)-cs[m]-cs[o],m,o}); } }
  /* label bins; contexts unseen in training go to the most populous bin */
  int *lab=malloc(nn*sizeof*lab); int nb=0; for(int i=0;i<nn;i++) lab[i]=-1;
  int big=-1; double bigs=-1; for(int i=0;i<nn;i++){ if(!alive[i])continue; double s=0; for(int y=0;y<A;y++) s+=nc[(size_t)i*A+y]; if(s>bigs){bigs=s;big=i;} }
  for(int v=0;v<V;v++){ double s=0; for(int y=0;y<A;y++) s+=cnt[(size_t)v*A+y]; int r=v; if(s==0) r=big; else while(parent[r]!=r) r=parent[r]; if(lab[r]<0) lab[r]=nb++; binof[v]=lab[r]; }
  free(nc);free(cs);free(alive);free(parent);free(lab); return nb; }

typedef struct { const char *name; int tgt; int nf; int f[4]; int d[4]; int nested; } def_t;
#define NF 16
static unsigned char *st[NF][3]; static size_t nn_[NF];
static int ctx1(const def_t*d,int fi,size_t i,int w1){ int a=d->d[0]<=(int)i?st[fi][d->f[0]][i-d->d[0]]:0, b=d->nf>=2&&d->d[1]<=(int)i?st[fi][d->f[1]][i-d->d[1]]:0; return a*w1+b; }
static int ctx2(const def_t*d,int fi,size_t i,int w3){ int a=d->d[2]<=(int)i?st[fi][d->f[2]][i-d->d[2]]:0, b=d->d[3]<=(int)i?st[fi][d->f[3]][i-d->d[3]]:0; return a*w3+b; }
/* fit maps on (files in 'train' mask, sequence range lo..hi fraction), evaluate on (file ev, range) */
static double evalcost(const def_t*d,int B,int w1,int V1,int w3,int V2,int trainmask,double tlo,double thi,int ev,double elo,double ehi,int Aq,int *nbins,double *mapbytes){
  int tgt=d->tgt;
  uint32_t *cnt=calloc((size_t)V1*A,sizeof*cnt), *cnt2=d->nested?calloc((size_t)V2*A,sizeof*cnt2):NULL;
  for(int fi=0;fi<NF;fi++){ if(!(trainmask>>fi&1)||!nn_[fi])continue; size_t lo=nn_[fi]*tlo, hi=nn_[fi]*thi; for(size_t i=lo;i<hi;i++){ cnt[(size_t)ctx1(d,fi,i,w1)*A+st[fi][tgt][i]]++; if(d->nested) cnt2[(size_t)ctx2(d,fi,i,w3)*A+st[fi][tgt][i]]++; } }
  int *map1=malloc(V1*sizeof*map1); int nb1=bin_greedy(cnt,V1,d->nested?32:B,map1); free(cnt);
  int *map2=NULL,nb2=0,*mapn=NULL,nbn=nb1;
  if(d->nested){ map2=malloc(V2*sizeof*map2); nb2=bin_greedy(cnt2,V2,32,map2); free(cnt2);
    int VN=nb1*nb2; uint32_t *cntn=calloc((size_t)VN*A,sizeof*cntn);
    for(int fi=0;fi<NF;fi++){ if(!(trainmask>>fi&1)||!nn_[fi])continue; size_t lo=nn_[fi]*tlo, hi=nn_[fi]*thi; for(size_t i=lo;i<hi;i++) cntn[(size_t)(map1[ctx1(d,fi,i,w1)]*nb2+map2[ctx2(d,fi,i,w3)])*A+st[fi][tgt][i]]++; }
    mapn=malloc(VN*sizeof*mapn); nbn=bin_greedy(cntn,VN,B,mapn); free(cntn); }
  uint32_t *tc=calloc((size_t)nbn*A,sizeof*tc); size_t lo=nn_[ev]*elo, hi=nn_[ev]*ehi;
  for(size_t i=lo;i<hi;i++){ int c=map1[ctx1(d,ev,i,w1)]; if(d->nested) c=mapn[c*nb2+map2[ctx2(d,ev,i,w3)]]; tc[(size_t)c*A+st[ev][tgt][i]]++; }
  double bc=0; for(int v=0;v<nbn;v++) bc+=cost(tc+(size_t)v*A); bc+=8.0*nbn*(Aq+2)/2; *nbins=nbn;
  if(mapbytes){ double mb=V1*ceil(log2(nb1>1?nb1:2)); if(d->nested) mb+=V2*ceil(log2(nb2>1?nb2:2))+(double)nb1*nb2*ceil(log2(nbn>1?nbn:2)); *mapbytes=mb/8; }
  free(tc); free(map1); free(map2); free(mapn); return bc; }
/* unmerged contexts on the whole file: the static cost with one table per context (non-nested models) */
static double fullcost(const def_t*d,int fi,int w1,int V1,int Aq){
  uint32_t *cnt=calloc((size_t)V1*A,sizeof*cnt); for(size_t i=0;i<nn_[fi];i++) cnt[(size_t)ctx1(d,fi,i,w1)*A+st[fi][d->tgt][i]]++;
  double bc=0; int used=0; for(int v=0;v<V1;v++){ double c=cost(cnt+(size_t)v*A); if(c>0||1){ uint32_t s=0; for(int y=0;y<A;y++) s+=cnt[(size_t)v*A+y]; if(s){ bc+=c; used++; } } }
  free(cnt); return bc+8.0*used*(Aq+2)/2; }
int main(int argc,char**argv){
  /* usage: kbin5 DUMPDIR LEVEL B file... ; dumps at DUMPDIR/dumpLEVEL_file/{ll,ml,of} */
  if(argc<5){ fprintf(stderr,"usage: kbin DUMPDIR LEVEL B[,B...] file...\n"); return 2; }
  const char*nm[3]={"ll","ml","of"}; char path[1024]; int nf=argc-4; const char**names=(const char**)argv+4;
  int Bs[16], nB=0; for(char *t=strtok(argv[3],","); t&&nB<16; t=strtok(NULL,",")) Bs[nB++]=atoi(t);
  for(int fi=0;fi<nf;fi++) for(int k=0;k<3;k++){ snprintf(path,sizeof path,"%s/dump%s_%s/%s",argv[1],argv[2],names[fi],nm[k]); size_t m; st[fi][k]=rdf(path,&m); nn_[fi]=m; }
  int ms[3]={0,0,0}; for(int fi=0;fi<nf;fi++) for(int k=0;k<3;k++) for(size_t i=0;i<nn_[fi];i++) if(st[fi][k][i]>ms[k]) ms[k]=st[fi][k][i];
  /* f: 0 ll 1 ml 2 of; d: lag.  cf = usable by the chain-free decoder (of order-0, ml from of history, ll from ml/of). */
  def_t defs[]={
    {"ll | ml            cf",0,1,{1},{0},0}, {"ll | of            cf",0,1,{2},{0},0}, {"ll | ll-1",0,1,{0},{1},0},
    {"ll | ml,of         cf",0,2,{1,2},{0,0},0}, {"ll | ml,ml-1       cf",0,2,{1,1},{0,1},0}, {"ll | ml,ll-1",0,2,{1,0},{0,1},0}, {"ll | ml,of-1       cf",0,2,{1,2},{0,1},0},
    {"ll | ml,of + ml-1,of-1  cf",0,4,{1,2,1,2},{0,0,1,1},1}, {"ll | ml,ml-1 + of,of-1  cf",0,4,{1,1,2,2},{0,1,0,1},1}, {"ll | ml,of + ml-1,ml-2  cf",0,4,{1,2,1,1},{0,0,1,2},1},
    {"ml | of            cf",1,1,{2},{0},0}, {"ml | of-1          cf",1,1,{2},{1},0}, {"ml | ll-1",1,1,{0},{1},0}, {"ml | ml-1",1,1,{1},{1},0},
    {"ml | of,of-1       cf",1,2,{2,2},{0,1},0}, {"ml | of,ll-1",1,2,{2,0},{0,1},0}, {"ml | of,ml-1",1,2,{2,1},{0,1},0}, {"ml | ll-1,ml-1",1,2,{0,1},{1,1},0},
    {"ml | of,of-1 + of-2,of-3  cf",1,4,{2,2,2,2},{0,1,2,3},1}, {"ml | of,ll-1 + of-1,ml-1",1,4,{2,0,2,1},{0,1,1,1},1},
    {"of | of-1",2,1,{2},{1},0}, {"of | ll-1",2,1,{0},{1},0}, {"of | ml-1",2,1,{1},{1},0}, {"of | ll-1,ml-1",2,2,{0,1},{1,1},0}, {"of | of-1,ml-1",2,2,{2,1},{1,1},0} };
  printf("L%s: %% of the file's order-0 cost (static, incl. table cost); 100 = order-0.  full = unmerged contexts (overfit bound);  per-file = map fit and costed on the whole file (what phaz transmits);  split = map fit on the 1st half, cost on the 2nd half;  pooled = map fit on the other files;  map = fixed-width map bytes\n",argv[2]);
  for(size_t di=0;di<sizeof defs/sizeof defs[0];di++){ def_t*d=&defs[di]; int tgt=d->tgt, Aq=ms[tgt]+1; int w1=d->nf>=2?ms[d->f[1]]+1:1, V1=(ms[d->f[0]]+1)*w1; int w3=d->nested?ms[d->f[3]]+1:1, V2=d->nested?(ms[d->f[2]]+1)*w3:1;
    double H0w[NF], H0h[NF]; for(int fi=0;fi<nf;fi++){ uint32_t c[A]={0}; for(size_t i=0;i<nn_[fi];i++) c[st[fi][tgt][i]]++; H0w[fi]=cost(c)+8.0*(Aq+2)/2; uint32_t h[A]={0}; for(size_t i=nn_[fi]/2;i<nn_[fi];i++) h[st[fi][tgt][i]]++; H0h[fi]=cost(h)+8.0*(Aq+2)/2; }
    printf("%-28s", d->name); for(int fi=0;fi<nf;fi++) printf(" %8s",names[fi]); printf("\n");
    if(!d->nested){ printf("%-28s","  full"); for(int fi=0;fi<nf;fi++) printf(" %7.1f%%",100*fullcost(d,fi,w1,V1,Aq)/H0w[fi]); printf("\n"); }
    for(int bi=0;bi<nB;bi++){ int B=Bs[bi]; char lab[32]; int nb; double mb;
      snprintf(lab,sizeof lab,"  per-file B=%d",B); printf("%-28s",lab); for(int fi=0;fi<nf;fi++){ double c=evalcost(d,B,w1,V1,w3,V2,1<<fi,0,1.0,fi,0,1.0,Aq,&nb,&mb); printf(" %7.1f%%",100*c/H0w[fi]); } printf("\n");
      snprintf(lab,sizeof lab,"  split B=%d",B); printf("%-28s",lab); for(int fi=0;fi<nf;fi++){ double c=evalcost(d,B,w1,V1,w3,V2,1<<fi,0,0.5,fi,0.5,1.0,Aq,&nb,NULL); printf(" %7.1f%%",100*c/H0h[fi]); } printf("\n");
      if(nf>1){ snprintf(lab,sizeof lab,"  pooled B=%d",B); printf("%-28s",lab); for(int fi=0;fi<nf;fi++){ double c=evalcost(d,B,w1,V1,w3,V2,((1<<nf)-1)&~(1<<fi),0,1.0,fi,0,1.0,Aq,&nb,NULL); printf(" %7.1f%%",100*c/H0w[fi]); } printf("\n"); }
      snprintf(lab,sizeof lab,"  map B=%d",B); printf("%-28s",lab); for(int fi=0;fi<nf;fi++){ evalcost(d,B,w1,V1,w3,V2,1<<fi,0,1.0,fi,0,1.0,Aq,&nb,&mb); printf(" %6.0f B",mb); } printf("\n"); } }
  return 0; }
