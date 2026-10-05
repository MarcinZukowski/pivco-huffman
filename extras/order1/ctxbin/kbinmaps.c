/* Duda's context binning on cross-field sequence-code contexts, evaluated honestly:
 * the bin map is fit on the first half of the sequences (greedy merge on the merge
 * penalty), then the second half is coded with per-bin static tables built on itself
 * (tables ship with the data; the map is a fixed design), plus a table cost of
 * B x (alphabet+2)/2 bytes.  Unbinned static = overfit reference on the same half. */
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


/* Built-in pair maps for phaz: fit on all 12 Silesia L19 parses pooled, B=64.
 * ll | (ml, of); ml | (of, ll-1); of | (ll-1, ml-1).  Alphabets 36/53/32. */
#define NLL 36
#define NML 53
#define NOF 32
int main(int argc,char**argv){
  const char*names[12]={"dickens","mozilla","mr","nci","ooffice","osdb","reymont","samba","sao","webster","x-ray","xml"}; const char*nm[3]={"ll","ml","of"}; char path[1024];
  unsigned char *st[12][3]; size_t nn[12];
  for(int fi=0;fi<12;fi++) for(int k=0;k<3;k++){ snprintf(path,sizeof path,"%s/dump19_%s/%s",argv[1],names[fi],nm[k]); size_t m; st[fi][k]=rdf(path,&m); nn[fi]=m; }
  struct { const char *cname; int tgt; int fa,da,fb,db; int wa,wb; } defs[4]={ {"phaz_map_ll",0,1,0,2,0,NML,NOF}, {"phaz_map_ml",1,2,0,0,1,NOF,NLL}, {"phaz_map_of",2,0,1,1,1,NLL,NML}, {"phaz_map_ml_cf",1,2,0,2,1,NOF,NOF} };
  printf("/* generated: context bin maps, B=64, pooled over Silesia L19 parses */\n#define PHAZ_CTX_B 64\n");
  for(int di=0;di<4;di++){ int V=defs[di].wa*defs[di].wb; uint32_t *cnt=calloc((size_t)V*A,sizeof*cnt);
    for(int fi=0;fi<12;fi++) for(size_t i=0;i<nn[fi];i++){ int a=defs[di].da<=(int)i?st[fi][defs[di].fa][i-defs[di].da]:0, b=defs[di].db<=(int)i?st[fi][defs[di].fb][i-defs[di].db]:0; cnt[(size_t)(a*defs[di].wb+b)*A+st[fi][defs[di].tgt][i]]++; }
    int *map=malloc(V*sizeof*map); int nb=bin_greedy(cnt,V,64,map);
    printf("static const unsigned char %s[%d*%d] = { /* %d bins */\n",defs[di].cname,defs[di].wa,defs[di].wb,nb);
    for(int v=0;v<V;v++){ printf("%d,",map[v]); if(v%32==31) printf("\n"); } printf("};\n"); free(cnt); free(map); }
  return 0; }
