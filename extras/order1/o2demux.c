/* o2demux.c -- order-1 demux kernel shootout on M4 (NEON).
 *
 * Problem: literals were split into K bucket streams by residence class
 * (class of the PREVIOUS byte); reconstruct the original order.  Classes are
 * raw top bits: K=4 -> byte>>6, K=2 -> byte>>7.  Segments restart context
 * (c=0) and give the lockstep harness independent chains.
 *
 * Kernels (all emit-per-dependent-lookup machines; the table walk consumes
 * only elements whose routing bits are visible in the index, plus ONE
 * terminal element whose position is determined but class unknown -- its
 * class is extracted from the already-loaded window at runtime):
 *
 *  heads4  K=4, visibility = head class of each bucket (user's scheme,
 *          simplified: per-lookup shuffle+8B store instead of batched concat).
 *          idx = 4 head classes (8b) + cur bucket (2b) -> 1024 entries.
 *  win4d3  K=4, visibility = 3 elements deep in CURRENT bucket + heads of
 *          the other three.  idx = deep(6b)+others(6b)+cur(2b) -> 16384.
 *  k2nt    K=2 (6,6) window kernel, replica of the o1demux winner (no
 *          terminal): idx = 6+6 class bits + side -> 8192.
 *  k2t     same + terminal emit (+1 byte/iter, class read from window).
 *
 * Harness: u1 (single chain, latency wall), u4 / u8 (lockstep hard-unrolled).
 * Verify: full memcmp vs input every kernel/file.  Self-test on random data.
 */
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#if defined(__aarch64__)
#include <arm_neon.h>
#define SHUF8(dst, lo, hi, pat64) vst1_u8(dst, vqtbl1_u8( \
    vcombine_u8(vcreate_u8(lo), vcreate_u8(hi)), vcreate_u8(pat64)))
#define SHUF16(dst, lo, hi, patp) vst1q_u8(dst, vqtbl1q_u8( \
    vcombine_u8(vcreate_u8(lo), vcreate_u8(hi)), vld1q_u8(patp)))
#else
#include <immintrin.h>
#define SHUF8(dst, lo, hi, pat64) do { \
    __m128i W_ = _mm_set_epi64x((long long)(hi), (long long)(lo)); \
    __m128i P_ = _mm_cvtsi64_si128((long long)(pat64)); \
    _mm_storel_epi64((__m128i *)(void *)(dst), _mm_shuffle_epi8(W_, P_)); } while (0)
#define SHUF16(dst, lo, hi, patp) do { \
    __m128i W_ = _mm_set_epi64x((long long)(hi), (long long)(lo)); \
    __m128i P_ = _mm_loadu_si128((const __m128i *)(const void *)(patp)); \
    _mm_storeu_si128((__m128i *)(void *)(dst), _mm_shuffle_epi8(W_, P_)); } while (0)
#endif
#if defined(__aarch64__)
#define MUX64(cond1, a, b) ((cond1) ? (b) : (a))          /* clang/gcc: csel */
#define MUXP(cond1, a, b) ((cond1) ? (b) : (a))
#else                       /* x86 gcc turns ternaries into BRANCHES: force ALU */
#define MUX64(cond1, a, b) ((a) ^ (((a) ^ (b)) & (uint64_t)-(uint64_t)(cond1)))
#define MUXP(cond1, a, b) ((const uint8_t *)(((uintptr_t)(a)) ^ \
    ((((uintptr_t)(a)) ^ ((uintptr_t)(b))) & (uintptr_t)-(uintptr_t)(cond1))))
#endif

#define MARGIN 32   /* k2g pair-step writes reach o + 12 + 16 = o+28 */
#ifdef PROF
static uint64_t g_it;
static uint64_t g_hist[16384];
#define ITC g_it++;
#define HST(ix) g_hist[ix]++;
#else
#define ITC
#define HST(ix)
#endif

typedef struct {
    int K, nseg;
    size_t N;
    size_t *segoff;          /* nseg+1 */
    uint8_t *(*bk)[16];      /* per segment (jittered start) */
    uint8_t *(*raw)[16];     /* allocation base for free() */
    size_t (*bl)[16];
} ENC;

static ENC *encode(const uint8_t *v, size_t n, int K, size_t segb, int shift) {
    ENC *E = calloc(1, sizeof *E);
    int nseg = (int)((n + segb - 1) / segb); if (nseg < 1) nseg = 1;
    E->K = K; E->nseg = nseg; E->N = n;
    E->segoff = malloc((nseg + 1) * sizeof *E->segoff);
    E->bk = calloc(nseg, sizeof *E->bk);
    E->raw = calloc(nseg, sizeof *E->raw);
    E->bl = calloc(nseg, sizeof *E->bl);
    size_t segn = segb;
    for (int s = 0; s < nseg; s++) {
        E->segoff[s] = s * segn;
        size_t lo = s * segn, hi = (s == nseg - 1) ? n : (s + 1) * segn;
        for (int b = 0; b < K; b++) {          /* jitter page offsets: 1B/step
            walks keep same-offset pointers locked -> 4K-alias stalls */
            E->raw[s][b] = malloc(hi - lo + 96);
            E->bk[s][b] = E->raw[s][b] + ((s * 5 + b * 3) & 7) * 8;
        }
        uint32_t c = 0;
        for (size_t j = lo; j < hi; j++) {
            E->bk[s][c][E->bl[s][c]++] = v[j];
            c = v[j] >> shift;
        }
        for (int b = 0; b < K; b++)          /* pad for wide loads */
            memset(E->bk[s][b] + E->bl[s][b], 0, 32);
    }
    E->segoff[nseg] = n;
    return E;
}
static void enc_free(ENC *E) {
    for (int s = 0; s < E->nseg; s++)
        for (int b = 0; b < E->K; b++) free(E->raw[s][b]);
    free(E->segoff); free(E->bk); free(E->raw); free(E->bl); free(E);
}

/* ---------------- reference scalar (indexed cursor, branch-free mux) ------ */
static void dec_ref(const ENC *E, uint8_t *out, int shift) {
    for (int s = 0; s < E->nseg; s++) {
        const uint8_t *P[4] = {E->bk[s][0], E->bk[s][1], E->bk[s][2], E->bk[s][3]};
        uint32_t c = 0;
        for (size_t j = E->segoff[s]; j < E->segoff[s + 1]; j++) {
            uint8_t b = *P[c]++;
            out[j] = b; c = b >> shift;
        }
    }
}

/* ---------------- heads4 tables ------------------------------------------ */
/* meta: cnt(3b@0) n0..n3(2b@3,5,7,9) tshift(6b@11) sel(1b@17) */
static uint32_t h4_meta[1024];
static uint64_t h4_pat[1024];

static void build_h4(void) {
    for (uint32_t ix = 0; ix < 1024; ix++) {
        uint32_t h[4] = {ix & 3, (ix >> 2) & 3, (ix >> 4) & 3, (ix >> 6) & 3};
        uint32_t cur = ix >> 8, used = 0, n[4] = {0, 0, 0, 0}, cnt = 0;
        uint64_t pat = 0; uint32_t tshift = 0, sel = 0;
        for (;;) {
            uint32_t b = cur;
            if (used >> b & 1) {              /* terminal: 2nd elem of b */
                pat |= (uint64_t)(4 * b + n[b]) << (8 * cnt);
                tshift = (b & 1) * 32 + n[b] * 8 + 6; sel = b >> 1;
                n[b]++; cnt++; break;
            }
            used |= 1u << b;
            pat |= (uint64_t)(4 * b + n[b]) << (8 * cnt);
            cur = h[b]; n[b]++; cnt++;
        }
        h4_meta[ix] = cnt | n[0] << 3 | n[1] << 5 | n[2] << 7 | n[3] << 9
                    | tshift << 11 | sel << 17;
        h4_pat[ix] = pat;
    }
}

#define H4STEP(A0, A1, A2, A3, cc, oo) do { \
    ITC \
    uint32_t w0_, w1_, w2_, w3_; \
    memcpy(&w0_, A0, 4); memcpy(&w1_, A1, 4); memcpy(&w2_, A2, 4); memcpy(&w3_, A3, 4); \
    uint32_t ix_ = ((w0_ >> 6) & 3) | (((w1_ >> 6) & 3) << 2) | (((w2_ >> 6) & 3) << 4) \
                 | (((w3_ >> 6) & 3) << 6) | ((cc) << 8); \
    HST(ix_) \
    uint32_t m_ = h4_meta[ix_]; \
    uint64_t u01_ = w0_ | ((uint64_t)w1_ << 32), u23_ = w2_ | ((uint64_t)w3_ << 32); \
    SHUF8(oo, u01_, u23_, h4_pat[ix_]); \
    A0 += (m_ >> 3) & 3; A1 += (m_ >> 5) & 3; A2 += (m_ >> 7) & 3; A3 += (m_ >> 9) & 3; \
    uint64_t sel_ = MUX64((m_ >> 17) & 1, u01_, u23_); \
    (cc) = (uint32_t)(sel_ >> ((m_ >> 11) & 63)) & 3; \
    (oo) += m_ & 7; \
} while (0)

/* ---------------- win4d3 tables ------------------------------------------ */
/* idx: deep d0..d2(2b each@0) | others o0..o2(2b each@6) | cur(2b@12)
 * meta: cnt(3b@0) n0..n3(3b@3,6,9,12) tshift(6b@15) sel(1b@21) */
static uint32_t w3_meta[16384];
static uint64_t w3_pat[16384];

static void build_w3(void) {
    for (uint32_t ix = 0; ix < 16384; ix++) {
        uint32_t d[3] = {ix & 3, (ix >> 2) & 3, (ix >> 4) & 3};
        uint32_t oth[3] = {(ix >> 6) & 3, (ix >> 8) & 3, (ix >> 10) & 3};
        uint32_t c = ix >> 12;
        uint32_t cur = c, n[4] = {0, 0, 0, 0}, cnt = 0;
        uint64_t pat = 0; uint32_t tshift = 0, sel = 0;
        for (;;) {
            uint32_t b = cur, allow = (b == c) ? 3 : 1;
            if (n[b] >= allow) {              /* terminal */
                pat |= (uint64_t)(4 * b + n[b]) << (8 * cnt);
                tshift = (b & 1) * 32 + n[b] * 8 + 6; sel = b >> 1;
                n[b]++; cnt++; break;
            }
            pat |= (uint64_t)(4 * b + n[b]) << (8 * cnt);
            cur = (b == c) ? d[n[b]] : oth[b < c ? b : b - 1];
            n[b]++; cnt++;
        }
        w3_meta[ix] = cnt | n[0] << 3 | n[1] << 6 | n[2] << 9 | n[3] << 12
                    | tshift << 15 | sel << 21;
        w3_pat[ix] = pat;
    }
}

#define W3STEP(A0, A1, A2, A3, cc, oo) do { \
    ITC \
    uint32_t w0_, w1_, w2_, w3_; \
    memcpy(&w0_, A0, 4); memcpy(&w1_, A1, 4); memcpy(&w2_, A2, 4); memcpy(&w3_, A3, 4); \
    uint64_t u01_ = w0_ | ((uint64_t)w1_ << 32), u23_ = w2_ | ((uint64_t)w3_ << 32); \
    uint64_t du_ = MUX64(((cc) >> 1) & 1, u01_, u23_); \
    uint32_t wc_ = (uint32_t)(du_ >> (((cc) & 1) << 5)); \
    uint32_t t_ = (wc_ >> 6) & 0x00030303u; \
    uint32_t deep_ = ((t_ * 0x10410u) >> 16) & 63; \
    uint32_t H8_ = ((w0_ >> 6) & 3) | (((w1_ >> 6) & 3) << 2) | (((w2_ >> 6) & 3) << 4) \
                 | (((w3_ >> 6) & 3) << 6); \
    uint32_t lo_ = H8_ & ((1u << (2 * (cc))) - 1); \
    uint32_t hi_ = (H8_ >> (2 * (cc) + 2)) << (2 * (cc)); \
    uint32_t ix_ = deep_ | (((lo_ | hi_) & 63) << 6) | ((cc) << 12); \
    HST(ix_) \
    uint32_t m_ = w3_meta[ix_]; \
    SHUF8(oo, u01_, u23_, w3_pat[ix_]); \
    A0 += (m_ >> 3) & 7; A1 += (m_ >> 6) & 7; A2 += (m_ >> 9) & 7; A3 += (m_ >> 12) & 7; \
    uint64_t sel_ = MUX64((m_ >> 21) & 1, u01_, u23_); \
    (cc) = (uint32_t)(sel_ >> ((m_ >> 15) & 63)) & 3; \
    (oo) += m_ & 7; \
} while (0)

/* ---------------- k2 (6,6) tables ---------------------------------------- */
/* idx: 6 L class bits | 6 R class bits <<6 | cur <<12 -> 8192.
 * nt meta: cnt(4b@0) nl(3b@4) nr(3b@7) cnext(1b@10)
 * t  meta: cnt(4b@0) nl(3b@4) nr(3b@7) tshift(6b@10) sel(1b@16) */
static uint32_t k2nt_meta[8192], k2t_meta[8192];
static uint8_t  k2nt_pat[8192][16], k2t_pat[8192][16];

static void build_k2(void) {
    for (uint32_t ix = 0; ix < 8192; ix++) {
        uint32_t c0 = ix >> 12;
        uint32_t nl = 0, nr = 0, cnt = 0, cur = c0;
        uint8_t pat[16] = {0};
        for (;;) {
            if (cur == 0) { if (nl >= 6) break; pat[cnt++] = nl; cur = (ix >> nl) & 1; nl++; }
            else          { if (nr >= 6) break; pat[cnt++] = 8 + nr; cur = (ix >> (6 + nr)) & 1; nr++; }
        }
        k2nt_meta[ix] = cnt | nl << 4 | nr << 7 | cur << 10;
        memcpy(k2nt_pat[ix], pat, 16);
        /* terminal variant: emit the stopper too, class read at runtime */
        uint32_t tshift, sel;
        if (cur == 0) { pat[cnt++] = nl; tshift = nl * 8 + 7; sel = 0; nl++; }
        else          { pat[cnt++] = 8 + nr; tshift = nr * 8 + 7; sel = 1; nr++; }
        k2t_meta[ix] = cnt | nl << 4 | nr << 7 | tshift << 10 | sel << 16;
        memcpy(k2t_pat[ix], pat, 16);
    }
}

#define BB  0x0101010101010101ULL
#define MUL 0x0102040810204080ULL

#define K2NT_STEP(LL, RR, cc, oo) do { \
    ITC \
    uint64_t l8_, r8_; memcpy(&l8_, LL, 8); memcpy(&r8_, RR, 8); \
    uint32_t gl_ = (uint32_t)((((l8_ >> 7) & BB) * MUL) >> 56) & 63; \
    uint32_t gr_ = (uint32_t)((((r8_ >> 7) & BB) * MUL) >> 56) & 63; \
    uint32_t ix_ = gl_ | (gr_ << 6) | ((cc) << 12); \
    HST(ix_) \
    uint32_t m_ = k2nt_meta[ix_]; \
    SHUF16(oo, l8_, r8_, k2nt_pat[ix_]); \
    LL += (m_ >> 4) & 7; RR += (m_ >> 7) & 7; \
    (cc) = (m_ >> 10) & 1; \
    (oo) += m_ & 15; \
} while (0)

#define K2T_STEP(LL, RR, cc, oo) do { \
    ITC \
    uint64_t l8_, r8_; memcpy(&l8_, LL, 8); memcpy(&r8_, RR, 8); \
    uint32_t gl_ = (uint32_t)((((l8_ >> 7) & BB) * MUL) >> 56) & 63; \
    uint32_t gr_ = (uint32_t)((((r8_ >> 7) & BB) * MUL) >> 56) & 63; \
    uint32_t ix_ = gl_ | (gr_ << 6) | ((cc) << 12); \
    uint32_t m_ = k2t_meta[ix_]; \
    SHUF16(oo, l8_, r8_, k2t_pat[ix_]); \
    LL += (m_ >> 4) & 7; RR += (m_ >> 7) & 7; \
    uint64_t sel_ = MUX64((m_ >> 16) & 1, l8_, r8_); \
    (cc) = (uint32_t)(sel_ >> ((m_ >> 10) & 63)) & 1; \
    (oo) += m_ & 15; \
} while (0)

/* ------- k2d: MZ's (4,4)x2 double-step, 9-bit index, 12KB tables --------- */
/* idx: 4 L class bits | 4 R class bits <<4 | cur <<8 -> 512 entries.
 * Two chained lookups share one 8B+8B source load + one header computation;
 * second shuffle pattern is offset by step-1 consumption (per-side add). */
static uint32_t k2d_meta[512];      /* cnt(4b@0) cl(3b@4) cr(3b@7) cnext(1b@10) */
static uint64_t k2d_pat[512];

static void build_k2d(void) {
    for (uint32_t ix = 0; ix < 512; ix++) {
        uint32_t c0 = ix >> 8, nl = 0, nr = 0, cnt = 0, cur = c0;
        uint64_t pat = 0;
        for (;;) {
            if (cur == 0) { if (nl >= 4) break; pat |= (uint64_t)nl << (8 * cnt); cur = (ix >> nl) & 1; nl++; }
            else          { if (nr >= 4) break; pat |= (uint64_t)(8 + nr) << (8 * cnt); cur = (ix >> (4 + nr)) & 1; nr++; }
            cnt++;
        }
        k2d_meta[ix] = cnt | nl << 4 | nr << 7 | cur << 10;
        k2d_pat[ix] = pat;
    }
}

#if defined(__aarch64__)
#define K2D_STEP(LL, RR, cc, oo) do {     uint64_t l8_, r8_; memcpy(&l8_, LL, 8); memcpy(&r8_, RR, 8);     uint32_t gl_ = (uint32_t)((((l8_ >> 7) & BB) * MUL) >> 56);     uint32_t gr_ = (uint32_t)((((r8_ >> 7) & BB) * MUL) >> 56);     uint8x16_t W_ = vcombine_u8(vcreate_u8(l8_), vcreate_u8(r8_));     uint32_t i1_ = (gl_ & 15) | ((gr_ & 15) << 4) | ((cc) << 8);     uint32_t m1_ = k2d_meta[i1_];     vst1_u8(oo, vqtbl1_u8(W_, vcreate_u8(k2d_pat[i1_])));     uint32_t c1_ = m1_ & 15, cl1_ = (m1_ >> 4) & 7, cr1_ = (m1_ >> 7) & 7;     uint32_t i2_ = ((gl_ >> cl1_) & 15) | (((gr_ >> cr1_) & 15) << 4) | (((m1_ >> 10) & 1) << 8);     uint32_t m2_ = k2d_meta[i2_];     uint8x8_t p2_ = vcreate_u8(k2d_pat[i2_]);     uint8x8_t ad_ = vbsl_u8(vcge_u8(p2_, vdup_n_u8(8)), vdup_n_u8((uint8_t)cr1_), vdup_n_u8((uint8_t)cl1_));     vst1_u8((oo) + c1_, vqtbl1_u8(W_, vadd_u8(p2_, ad_)));     LL += cl1_ + ((m2_ >> 4) & 7); RR += cr1_ + ((m2_ >> 7) & 7);     (cc) = (m2_ >> 10) & 1;     (oo) += c1_ + (m2_ & 15); } while (0)
#else
#define K2D_STEP(LL, RR, cc, oo) do {     uint64_t l8_, r8_; memcpy(&l8_, LL, 8); memcpy(&r8_, RR, 8);     __m128i W_ = _mm_set_epi64x((long long)r8_, (long long)l8_);     uint32_t hb_ = (uint32_t)_mm_movemask_epi8(W_);     uint32_t gl_ = hb_ & 255, gr_ = (hb_ >> 8) & 255;     uint32_t i1_ = (gl_ & 15) | ((gr_ & 15) << 4) | ((cc) << 8);     uint32_t m1_ = k2d_meta[i1_];     _mm_storel_epi64((__m128i *)(void *)(oo),         _mm_shuffle_epi8(W_, _mm_cvtsi64_si128((long long)k2d_pat[i1_])));     uint32_t c1_ = m1_ & 15, cl1_ = (m1_ >> 4) & 7, cr1_ = (m1_ >> 7) & 7;     uint32_t i2_ = ((gl_ >> cl1_) & 15) | (((gr_ >> cr1_) & 15) << 4) | (((m1_ >> 10) & 1) << 8);     uint32_t m2_ = k2d_meta[i2_];     __m128i p2_ = _mm_cvtsi64_si128((long long)k2d_pat[i2_]);     __m128i ms_ = _mm_cmpgt_epi8(p2_, _mm_set1_epi8(7));     __m128i ad_ = _mm_blendv_epi8(_mm_set1_epi8((char)cl1_), _mm_set1_epi8((char)cr1_), ms_);     _mm_storel_epi64((__m128i *)(void *)((oo) + c1_),         _mm_shuffle_epi8(W_, _mm_add_epi8(p2_, ad_)));     LL += cl1_ + ((m2_ >> 4) & 7); RR += cr1_ + ((m2_ >> 7) & 7);     (cc) = (m2_ >> 10) & 1;     (oo) += c1_ + (m2_ & 15); } while (0)
/* k2m: the (6,6) kernel with one movemask replacing two multiply-gathers */
#define K2M_STEP(LL, RR, cc, oo) do {     uint64_t l8_, r8_; memcpy(&l8_, LL, 8); memcpy(&r8_, RR, 8);     __m128i W_ = _mm_set_epi64x((long long)r8_, (long long)l8_);     uint32_t hb_ = (uint32_t)_mm_movemask_epi8(W_);     uint32_t ix_ = (hb_ & 63) | (((hb_ >> 8) & 63) << 6) | ((cc) << 12);     uint32_t m_ = k2nt_meta[ix_];     _mm_storeu_si128((__m128i *)(void *)(oo),         _mm_shuffle_epi8(W_, _mm_loadu_si128((const __m128i *)(const void *)k2nt_pat[ix_])));     LL += (m_ >> 4) & 7; RR += (m_ >> 7) & 7;     (cc) = (m_ >> 10) & 1;     (oo) += m_ & 15; } while (0)
#endif

/* ------- k2g: MZ's 2x16B double -- one load pair, two (6,6) steps -------- */
/* Load 16 bytes per side ONCE, gather all 16 top bits per side ONCE, then
 * run TWO full (6,6) table steps out of the registers.  Step 2's 6+6 index
 * bits are just (gl>>nl1, gr>>nr1) of the same header words -- the second
 * stream load + movemask/gather drops off the dependency chain entirely
 * (that chained-adjust shift is what killed k2d/k2e, but those also paid
 * for it with crippled (4,4) windows; here both steps keep full yield and
 * reuse k2nt_meta unchanged).  Emit shuffles from the 32B {L16,R16} pair
 * via k2g_pat (R lanes stored at +16); step 2 adds +nl1 to L lanes and
 * +nr1 to R lanes of its pattern.  Consumption per pair <= 12/side, so
 * the 16B loads always cover both steps. */
static uint8_t k2g_pat[8192][16];
static void build_k2g(void) {
    for (uint32_t ix = 0; ix < 8192; ix++)
        for (int j = 0; j < 16; j++) {
            uint8_t v = k2nt_pat[ix][j];
            k2g_pat[ix][j] = v < 8 ? v : (uint8_t)(16 + (v - 8));
        }
}

#if defined(__aarch64__)
#define K2G_STEP(LL, RR, cc, oo) do { \
    ITC \
    uint64_t la_, lb_, ra_, rb_; \
    memcpy(&la_, LL, 8); memcpy(&lb_, (LL) + 8, 8); \
    memcpy(&ra_, RR, 8); memcpy(&rb_, (RR) + 8, 8); \
    uint32_t gl_ = (uint32_t)((((la_ >> 7) & BB) * MUL) >> 56) \
                 | ((uint32_t)((((lb_ >> 7) & BB) * MUL) >> 56) << 8); \
    uint32_t gr_ = (uint32_t)((((ra_ >> 7) & BB) * MUL) >> 56) \
                 | ((uint32_t)((((rb_ >> 7) & BB) * MUL) >> 56) << 8); \
    uint8x16x2_t W_; \
    W_.val[0] = vcombine_u8(vcreate_u8(la_), vcreate_u8(lb_)); \
    W_.val[1] = vcombine_u8(vcreate_u8(ra_), vcreate_u8(rb_)); \
    uint32_t i1_ = (gl_ & 63) | ((gr_ & 63) << 6) | ((cc) << 12); \
    uint32_t m1_ = k2nt_meta[i1_]; \
    vst1q_u8((oo), vqtbl2q_u8(W_, vld1q_u8(k2g_pat[i1_]))); \
    uint32_t c1_ = m1_ & 15, nl_ = (m1_ >> 4) & 7, nr_ = (m1_ >> 7) & 7; \
    ITC \
    uint32_t i2_ = ((gl_ >> nl_) & 63) | (((gr_ >> nr_) & 63) << 6) \
                 | (((m1_ >> 10) & 1) << 12); \
    uint32_t m2_ = k2nt_meta[i2_]; \
    uint8x16_t p2_ = vld1q_u8(k2g_pat[i2_]); \
    uint8x16_t ad_ = vbslq_u8(vcgeq_u8(p2_, vdupq_n_u8(16)), \
                              vdupq_n_u8((uint8_t)nr_), vdupq_n_u8((uint8_t)nl_)); \
    vst1q_u8((oo) + c1_, vqtbl2q_u8(W_, vaddq_u8(p2_, ad_))); \
    LL += nl_ + ((m2_ >> 4) & 7); RR += nr_ + ((m2_ >> 7) & 7); \
    (cc) = (m2_ >> 10) & 1; \
    (oo) += c1_ + (m2_ & 15); \
} while (0)
#define HAVE_K2G 1
#elif defined(__AVX512VBMI__)
#define K2G_STEP(LL, RR, cc, oo) do { \
    ITC \
    __m128i XL_ = _mm_loadu_si128((const __m128i *)(const void *)(LL)); \
    __m128i XR_ = _mm_loadu_si128((const __m128i *)(const void *)(RR)); \
    uint32_t gl_ = (uint32_t)_mm_movemask_epi8(XL_); \
    uint32_t gr_ = (uint32_t)_mm_movemask_epi8(XR_); \
    __m256i S_ = _mm256_inserti128_si256(_mm256_castsi128_si256(XL_), XR_, 1); \
    uint32_t i1_ = (gl_ & 63) | ((gr_ & 63) << 6) | ((cc) << 12); \
    uint32_t m1_ = k2nt_meta[i1_]; \
    _mm_storeu_si128((__m128i *)(void *)(oo), \
        _mm256_castsi256_si128(_mm256_permutexvar_epi8( \
            _mm256_castsi128_si256( \
                _mm_loadu_si128((const __m128i *)(const void *)k2g_pat[i1_])), S_))); \
    uint32_t c1_ = m1_ & 15, nl_ = (m1_ >> 4) & 7, nr_ = (m1_ >> 7) & 7; \
    ITC \
    uint32_t i2_ = ((gl_ >> nl_) & 63) | (((gr_ >> nr_) & 63) << 6) \
                 | (((m1_ >> 10) & 1) << 12); \
    uint32_t m2_ = k2nt_meta[i2_]; \
    __m128i p2_ = _mm_loadu_si128((const __m128i *)(const void *)k2g_pat[i2_]); \
    __m128i ms_ = _mm_cmpgt_epi8(p2_, _mm_set1_epi8(15)); \
    __m128i ad_ = _mm_blendv_epi8(_mm_set1_epi8((char)nl_), \
                                  _mm_set1_epi8((char)nr_), ms_); \
    _mm_storeu_si128((__m128i *)(void *)((oo) + c1_), \
        _mm256_castsi256_si128(_mm256_permutexvar_epi8( \
            _mm256_castsi128_si256(_mm_add_epi8(p2_, ad_)), S_))); \
    LL += nl_ + ((m2_ >> 4) & 7); RR += nr_ + ((m2_ >> 7) & 7); \
    (cc) = (m2_ >> 10) & 1; \
    (oo) += c1_ + (m2_ & 15); \
} while (0)
#define HAVE_K2G 1
#endif

/* ------- k2p/k2q: pext indices + byte-packed meta (x86 BMI2) ------------- */
/* Counter data: post-compaction ~11 of ~34 instr/step are field/index
 * bit-extraction (x86 pays shr+and per field; aarch64 has ubfx so this
 * lever is x86-only).  Two fixes: (1) meta repacked at byte offsets --
 * cnt@0 nl@8 nr@16 cnext@24 -- so every field is one movzx (cnext one
 * plain shr); (2) index construction via ONE pext: step 1's 6+6 bits =
 * pext(h, 0x003F003F) with h = gl|gr<<16, and step 2's window bits
 * (gl>>nl1)&63 | ((gr>>nr1)&63)<<6 = pext(h, mask) where mask =
 * 0x3F<<nl1 | 0x3F<<(16+nr1) is PRE-COMPUTED per step-1 table entry --
 * the whole chained-adjust shift/mask/or collapses into one table load
 * (parallel with meta) + one 3-cycle pext. */
static uint32_t k2p_meta[8192];   /* cnt | nl<<8 | nr<<16 | cnext<<24 */
static uint32_t k2p_mask[8192];   /* step-2 pext mask for this entry */
static void build_k2p(void) {
    for (uint32_t ix = 0; ix < 8192; ix++) {
        uint32_t m = k2nt_meta[ix];
        uint32_t nl = (m >> 4) & 7, nr = (m >> 7) & 7;
        k2p_meta[ix] = (m & 15) | nl << 8 | nr << 16 | ((m >> 10) & 1) << 24;
        k2p_mask[ix] = (0x3Fu << nl) | (0x3Fu << (16 + nr));
    }
}

#if !defined(__aarch64__) && defined(__BMI2__)
/* k2q: the k2m single step with pext index + byte meta */
#define K2Q_STEP(LL, RR, cc, oo) do { \
    ITC \
    uint64_t l8_, r8_; memcpy(&l8_, LL, 8); memcpy(&r8_, RR, 8); \
    __m128i W_ = _mm_set_epi64x((long long)r8_, (long long)l8_); \
    uint32_t hb_ = (uint32_t)_mm_movemask_epi8(W_); \
    uint32_t ix_ = _pext_u32(hb_, 0x3F3Fu) | ((cc) << 12); \
    uint32_t m_ = k2p_meta[ix_]; \
    _mm_storeu_si128((__m128i *)(void *)(oo), \
        _mm_shuffle_epi8(W_, _mm_loadu_si128((const __m128i *)(const void *)k2nt_pat[ix_]))); \
    LL += (m_ >> 8) & 255; RR += (m_ >> 16) & 255; \
    (cc) = m_ >> 24; \
    (oo) += m_ & 255; \
} while (0)
#define HAVE_K2Q 1
#endif

#if !defined(__aarch64__) && defined(__BMI2__) && defined(__AVX512VBMI__)
/* k2p: the k2g 2x16B double with pext indices + byte meta */
#define K2P_STEP(LL, RR, cc, oo) do { \
    ITC \
    __m128i XL_ = _mm_loadu_si128((const __m128i *)(const void *)(LL)); \
    __m128i XR_ = _mm_loadu_si128((const __m128i *)(const void *)(RR)); \
    uint32_t h_ = (uint32_t)_mm_movemask_epi8(XL_) \
                | ((uint32_t)_mm_movemask_epi8(XR_) << 16); \
    __m256i S_ = _mm256_inserti128_si256(_mm256_castsi128_si256(XL_), XR_, 1); \
    uint32_t i1_ = _pext_u32(h_, 0x003F003Fu) | ((cc) << 12); \
    uint32_t m1_ = k2p_meta[i1_]; \
    _mm_storeu_si128((__m128i *)(void *)(oo), \
        _mm256_castsi256_si128(_mm256_permutexvar_epi8( \
            _mm256_castsi128_si256( \
                _mm_loadu_si128((const __m128i *)(const void *)k2g_pat[i1_])), S_))); \
    uint32_t c1_ = m1_ & 255, nl_ = (m1_ >> 8) & 255, nr_ = (m1_ >> 16) & 255; \
    ITC \
    uint32_t i2_ = _pext_u32(h_, k2p_mask[i1_]) | ((m1_ >> 24) << 12); \
    uint32_t m2_ = k2p_meta[i2_]; \
    __m128i p2_ = _mm_loadu_si128((const __m128i *)(const void *)k2g_pat[i2_]); \
    __m128i ms_ = _mm_cmpgt_epi8(p2_, _mm_set1_epi8(15)); \
    __m128i ad_ = _mm_blendv_epi8(_mm_set1_epi8((char)nl_), \
                                  _mm_set1_epi8((char)nr_), ms_); \
    _mm_storeu_si128((__m128i *)(void *)((oo) + c1_), \
        _mm256_castsi256_si128(_mm256_permutexvar_epi8( \
            _mm256_castsi128_si256(_mm_add_epi8(p2_, ad_)), S_))); \
    LL += nl_ + ((m2_ >> 8) & 255); RR += nr_ + ((m2_ >> 16) & 255); \
    (cc) = m2_ >> 24; \
    (oo) += c1_ + (m2_ & 255); \
} while (0)
#define HAVE_K2P 1
#endif

/* ------- k2e: MZ's fused double -- combo id in 2nd index, one shuffle ----- */
/* t1: 9-bit idx -> combo(4b@0) c1(1b@4) cl1(3b@5) cr1(3b@8) + zero-padded 16B
 * pattern.  t2: combo | nextL4<<4 | nextR4<<8 | c1<<12 -> pattern PRE-offset
 * (+cl1/+cr1) and PRE-positioned at byte cnt1, zeros below.  P = P1|P2. */
static uint32_t k2e1_meta[512];  static uint8_t k2e1_pat[512][16];
static uint32_t k2e2_meta[8192]; static uint8_t k2e2_pat[8192][16];

static void build_k2e(void) {
    for (uint32_t ix = 0; ix < 512; ix++) {
        uint32_t c0 = ix >> 8, nl = 0, nr = 0, cnt = 0, cur = c0;
        uint8_t pat[16] = {0};
        for (;;) {
            if (cur == 0) { if (nl >= 4) break; pat[cnt] = nl; cur = (ix >> nl) & 1; nl++; }
            else          { if (nr >= 4) break; pat[cnt] = 8 + nr; cur = (ix >> (4 + nr)) & 1; nr++; }
            cnt++;
        }
        uint32_t combo = (nl == 4) ? nr : 5 + nl;      /* 9 reachable ids */
        k2e1_meta[ix] = combo | cur << 4 | nl << 5 | nr << 8;
        memcpy(k2e1_pat[ix], pat, 16);
    }
    for (uint32_t ix = 0; ix < 8192; ix++) {
        uint32_t combo = ix & 15;
        if (combo > 8) continue;
        uint32_t cl1 = combo <= 4 ? 4 : combo - 5, cr1 = combo <= 4 ? combo : 4;
        uint32_t cnt1 = cl1 + cr1;
        uint32_t L4 = (ix >> 4) & 15, R4 = (ix >> 8) & 15, cur = (ix >> 12) & 1;
        uint32_t nl = 0, nr = 0, cnt = 0;
        uint8_t pat[16] = {0};
        for (;;) {
            if (cur == 0) { if (nl >= 4) break; pat[cnt1 + cnt] = cl1 + nl; cur = (L4 >> nl) & 1; nl++; }
            else          { if (nr >= 4) break; pat[cnt1 + cnt] = 8 + cr1 + nr; cur = (R4 >> nr) & 1; nr++; }
            cnt++;
        }
        k2e2_meta[ix] = cnt | nl << 4 | nr << 7 | cur << 10;
        memcpy(k2e2_pat[ix], pat, 16);
    }
}

#if defined(__aarch64__)
#define K2E_STEP(LL, RR, cc, oo) do {     uint64_t l8_, r8_; memcpy(&l8_, LL, 8); memcpy(&r8_, RR, 8);     uint32_t gl_ = (uint32_t)((((l8_ >> 7) & BB) * MUL) >> 56);     uint32_t gr_ = (uint32_t)((((r8_ >> 7) & BB) * MUL) >> 56);     uint32_t i1_ = (gl_ & 15) | ((gr_ & 15) << 4) | ((cc) << 8);     uint32_t m1_ = k2e1_meta[i1_];     uint32_t cl1_ = (m1_ >> 5) & 7, cr1_ = (m1_ >> 8) & 7;     uint32_t i2_ = (m1_ & 15) | (((gl_ >> cl1_) & 15) << 4)                  | (((gr_ >> cr1_) & 15) << 8) | (((m1_ >> 4) & 1) << 12);     uint32_t m2_ = k2e2_meta[i2_];     uint8x16_t P_ = vorrq_u8(vld1q_u8(k2e1_pat[i1_]), vld1q_u8(k2e2_pat[i2_]));     vst1q_u8(oo, vqtbl1q_u8(vcombine_u8(vcreate_u8(l8_), vcreate_u8(r8_)), P_));     LL += cl1_ + ((m2_ >> 4) & 7); RR += cr1_ + ((m2_ >> 7) & 7);     (cc) = (m2_ >> 10) & 1;     (oo) += cl1_ + cr1_ + (m2_ & 15); } while (0)
#else
#define K2E_STEP(LL, RR, cc, oo) do {     uint64_t l8_, r8_; memcpy(&l8_, LL, 8); memcpy(&r8_, RR, 8);     __m128i W_ = _mm_set_epi64x((long long)r8_, (long long)l8_);     uint32_t hb_ = (uint32_t)_mm_movemask_epi8(W_);     uint32_t gl_ = hb_ & 255, gr_ = (hb_ >> 8) & 255;     uint32_t i1_ = (gl_ & 15) | ((gr_ & 15) << 4) | ((cc) << 8);     uint32_t m1_ = k2e1_meta[i1_];     uint32_t cl1_ = (m1_ >> 5) & 7, cr1_ = (m1_ >> 8) & 7;     uint32_t i2_ = (m1_ & 15) | (((gl_ >> cl1_) & 15) << 4)                  | (((gr_ >> cr1_) & 15) << 8) | (((m1_ >> 4) & 1) << 12);     uint32_t m2_ = k2e2_meta[i2_];     __m128i P_ = _mm_or_si128(         _mm_loadu_si128((const __m128i *)(const void *)k2e1_pat[i1_]),         _mm_loadu_si128((const __m128i *)(const void *)k2e2_pat[i2_]));     _mm_storeu_si128((__m128i *)(void *)(oo), _mm_shuffle_epi8(W_, P_));     LL += cl1_ + ((m2_ >> 4) & 7); RR += cr1_ + ((m2_ >> 7) & 7);     (cc) = (m2_ >> 10) & 1;     (oo) += cl1_ + cr1_ + (m2_ & 15); } while (0)
#endif

/* ------- k2f: MZ's big-table completion -- ONE chained load ---------------
 * idx = c<<16 | hb (full 16 header bits): entry carries i2, total advances,
 * next c.  i1 is ALU-computable; patterns factored via k2e tables + OR.  */
static uint32_t k2f_tab[1 << 17];   /* i2(13b@0) clT(4b@13) crT(4b@17) c(1b@21) */

static void build_k2f(void) {       /* composed from the k2e sub-tables */
    for (uint32_t c = 0; c < 2; c++)
        for (uint32_t hb = 0; hb < 65536; hb++) {
            uint32_t gl = hb & 255, gr = (hb >> 8) & 255;
            uint32_t i1 = (gl & 15) | ((gr & 15) << 4) | (c << 8);
            uint32_t m1 = k2e1_meta[i1];
            uint32_t cl1 = (m1 >> 5) & 7, cr1 = (m1 >> 8) & 7;
            uint32_t i2 = (m1 & 15) | (((gl >> cl1) & 15) << 4)
                        | (((gr >> cr1) & 15) << 8) | (((m1 >> 4) & 1) << 12);
            uint32_t m2 = k2e2_meta[i2];
            uint32_t clT = cl1 + ((m2 >> 4) & 7), crT = cr1 + ((m2 >> 7) & 7);
            k2f_tab[(c << 16) | hb] = i2 | clT << 13 | crT << 17 | ((m2 >> 10) & 1) << 21;
        }
}

#if defined(__aarch64__)
#define K2F_STEP(LL, RR, cc, oo) do {     uint64_t l8_, r8_; memcpy(&l8_, LL, 8); memcpy(&r8_, RR, 8);     uint32_t gl_ = (uint32_t)((((l8_ >> 7) & BB) * MUL) >> 56);     uint32_t gr_ = (uint32_t)((((r8_ >> 7) & BB) * MUL) >> 56);     uint32_t hb_ = gl_ | (gr_ << 8);     uint32_t e_ = k2f_tab[((cc) << 16) | hb_];     uint32_t i1_ = (gl_ & 15) | ((gr_ & 15) << 4) | ((cc) << 8);     uint8x16_t P_ = vorrq_u8(vld1q_u8(k2e1_pat[i1_]), vld1q_u8(k2e2_pat[e_ & 8191]));     vst1q_u8(oo, vqtbl1q_u8(vcombine_u8(vcreate_u8(l8_), vcreate_u8(r8_)), P_));     uint32_t clT_ = (e_ >> 13) & 15, crT_ = (e_ >> 17) & 15;     LL += clT_; RR += crT_;     (cc) = (e_ >> 21) & 1;     (oo) += clT_ + crT_; } while (0)
#else
#define K2F_STEP(LL, RR, cc, oo) do {     uint64_t l8_, r8_; memcpy(&l8_, LL, 8); memcpy(&r8_, RR, 8);     __m128i W_ = _mm_set_epi64x((long long)r8_, (long long)l8_);     uint32_t hb_ = (uint32_t)_mm_movemask_epi8(W_);     uint32_t e_ = k2f_tab[((cc) << 16) | hb_];     uint32_t i1_ = (hb_ & 15) | (((hb_ >> 8) & 15) << 4) | ((cc) << 8);     __m128i P_ = _mm_or_si128(         _mm_loadu_si128((const __m128i *)(const void *)k2e1_pat[i1_]),         _mm_loadu_si128((const __m128i *)(const void *)k2e2_pat[e_ & 8191]));     _mm_storeu_si128((__m128i *)(void *)(oo), _mm_shuffle_epi8(W_, P_));     uint32_t clT_ = (e_ >> 13) & 15, crT_ = (e_ >> 17) & 15;     LL += clT_; RR += crT_;     (cc) = (e_ >> 21) & 1;     (oo) += clT_ + crT_; } while (0)
#endif

/* ------- k4r: MZ's register-arithmetic K=4 walk (no tables) ---------------
 * H = 4x16-bit lanes of 8-deep 2-bit class headers; R = per-bucket next-ref
 * bytes (ref = 8c+n).  Identity: class shift = 2*ref.  8 elements/iteration,
 * unconditional (8-deep visibility per bucket => walk never stalls). */
#if defined(__aarch64__)
#define HAVE_K4R 1
#define K4R_HDR(l_, lane_) do {     uint64_t t_ = ((l_) >> 6) & 0x0303030303030303ULL;     uint64_t lo_ = ((t_ & 0xFFFFFFFFULL) * 0x0104104000000000ULL) >> 56;     uint64_t hi_ = ((t_ >> 32) * 0x0104104000000000ULL) >> 56;     H_ |= (lo_ | (hi_ << 8)) << (16 * (lane_)); } while (0)
#elif defined(__AVX512VBMI__) && defined(__AVX512VL__)
#define HAVE_K4R 1
#endif

#ifdef HAVE_K4R
#define K4R_W(k) { uint32_t s3_ = c_ << 3;     uint32_t ref_ = (R_ >> s3_) & 255;     P_ |= (uint64_t)ref_ << (8 * (k));     c_ = (uint32_t)(H_ >> (ref_ << 1)) & 3;     R_ += 1u << s3_; }

#if defined(__aarch64__)
#define K4R_STEP(A0, A1, A2, A3, cc, oo) do {     uint64_t l0_, l1_, l2_, l3_;     memcpy(&l0_, A0, 8); memcpy(&l1_, A1, 8); memcpy(&l2_, A2, 8); memcpy(&l3_, A3, 8);     uint64_t H_ = 0;     K4R_HDR(l0_, 0); K4R_HDR(l1_, 1); K4R_HDR(l2_, 2); K4R_HDR(l3_, 3);     uint32_t c_ = (cc), R_ = 0x18100800u; uint64_t P_ = 0;     K4R_W(0) K4R_W(1) K4R_W(2) K4R_W(3) K4R_W(4) K4R_W(5) K4R_W(6) K4R_W(7)     uint8x16x2_t T_ = {{ vcombine_u8(vcreate_u8(l0_), vcreate_u8(l1_)),                          vcombine_u8(vcreate_u8(l2_), vcreate_u8(l3_)) }};     vst1_u8(oo, vqtbl2_u8(T_, vcreate_u8(P_)));     uint32_t d_ = R_ - 0x18100800u;     A0 += d_ & 255; A1 += (d_ >> 8) & 255; A2 += (d_ >> 16) & 255; A3 += d_ >> 24;     (cc) = c_; (oo) += 8; } while (0)
#else
#define K4R_STEP(A0, A1, A2, A3, cc, oo) do {     uint64_t l0_, l1_, l2_, l3_;     memcpy(&l0_, A0, 8); memcpy(&l1_, A1, 8); memcpy(&l2_, A2, 8); memcpy(&l3_, A3, 8);     __m128i W01_ = _mm_set_epi64x((long long)l1_, (long long)l0_);     __m128i W23_ = _mm_set_epi64x((long long)l3_, (long long)l2_);     uint32_t b7a_ = (uint32_t)_mm_movemask_epi8(W01_);     uint32_t b6a_ = (uint32_t)_mm_movemask_epi8(_mm_add_epi8(W01_, W01_));     uint32_t b7b_ = (uint32_t)_mm_movemask_epi8(W23_);     uint32_t b6b_ = (uint32_t)_mm_movemask_epi8(_mm_add_epi8(W23_, W23_));     uint64_t H_ = (uint64_t)(_pdep_u32(b7a_, 0xAAAAAAAAu) | _pdep_u32(b6a_, 0x55555555u))                 | ((uint64_t)(_pdep_u32(b7b_, 0xAAAAAAAAu) | _pdep_u32(b6b_, 0x55555555u)) << 32);     uint32_t c_ = (cc), R_ = 0x18100800u; uint64_t P_ = 0;     K4R_W(0) K4R_W(1) K4R_W(2) K4R_W(3) K4R_W(4) K4R_W(5) K4R_W(6) K4R_W(7)     __m256i S_ = _mm256_set_epi64x((long long)l3_, (long long)l2_, (long long)l1_, (long long)l0_);     __m256i I_ = _mm256_castsi128_si256(_mm_cvtsi64_si128((long long)P_));     _mm_storel_epi64((__m128i *)(void *)(oo),         _mm256_castsi256_si128(_mm256_permutexvar_epi8(I_, S_)));     uint32_t d_ = R_ - 0x18100800u;     A0 += d_ & 255; A1 += (d_ >> 8) & 255; A2 += (d_ >> 16) & 255; A3 += d_ >> 24;     (cc) = c_; (oo) += 8; } while (0)
#endif
#endif /* HAVE_K4R */

/* ---------------- lockstep generators ------------------------------------ */
#define FOR1(X) X(0)
#define FOR1_2(X, A) X(0, A)
#define FOR4_2(X, A) X(0, A) X(1, A) X(2, A) X(3, A)
#define FOR8_2(X, A) X(0, A) X(1, A) X(2, A) X(3, A) X(4, A) X(5, A) X(6, A) X(7, A)
#define FOR2(X) X(0) X(1)
#define FOR2_2(X, A) X(0, A) X(1, A)
#define FOR6(X) X(0) X(1) X(2) X(3) X(4) X(5)
#define FOR6_2(X, A) X(0, A) X(1, A) X(2, A) X(3, A) X(4, A) X(5, A)
#define FOR12(X) X(0) X(1) X(2) X(3) X(4) X(5) X(6) X(7) X(8) X(9) X(10) X(11)
#define FOR12_2(X, A) X(0, A) X(1, A) X(2, A) X(3, A) X(4, A) X(5, A) X(6, A) X(7, A) \
                      X(8, A) X(9, A) X(10, A) X(11, A)
#define FOR24(X) FOR12(X) X(12) X(13) X(14) X(15) X(16) X(17) X(18) X(19) X(20) X(21) X(22) X(23)
#define FOR24_2(X, A) FOR12_2(X, A) X(12, A) X(13, A) X(14, A) X(15, A) X(16, A) X(17, A) \
                      X(18, A) X(19, A) X(20, A) X(21, A) X(22, A) X(23, A)
#define FOR16(X) X(0) X(1) X(2) X(3) X(4) X(5) X(6) X(7) \
                 X(8) X(9) X(10) X(11) X(12) X(13) X(14) X(15)
#define FOR16_2(X, A) X(0, A) X(1, A) X(2, A) X(3, A) X(4, A) X(5, A) X(6, A) X(7, A) \
                      X(8, A) X(9, A) X(10, A) X(11, A) X(12, A) X(13, A) X(14, A) X(15, A)

#define FOR4(X) X(0) X(1) X(2) X(3)
#define FOR8(X) X(0) X(1) X(2) X(3) X(4) X(5) X(6) X(7)

#define D4DECL(k) \
    const uint8_t *a0##k = 0, *a1##k = 0, *a2##k = 0, *a3##k = 0; \
    uint32_t c##k = 0; int si##k = k, done##k = 0; \
    uint8_t *o##k = 0, *lim##k = 0; \
    D4LOAD(k)
#define D4LOAD(k) if (si##k < E->nseg) { \
        a0##k = E->bk[si##k][0]; a1##k = E->bk[si##k][1]; \
        a2##k = E->bk[si##k][2]; a3##k = E->bk[si##k][3]; c##k = 0; \
        o##k = out + E->segoff[si##k]; \
        lim##k = out + E->segoff[si##k + 1] - MARGIN; \
    } else done##k = 1;
#define D4DRAINC(k) { const uint8_t *P_[4] = {a0##k, a1##k, a2##k, a3##k}; \
    uint8_t *e_ = out + E->segoff[si##k + 1]; uint32_t c_ = c##k; uint8_t *o_ = o##k; \
    while (o_ < e_) { uint8_t b_ = *P_[c_]++; *o_++ = b_; c_ = b_ >> 6; } }
#define D4TICK(k, NCH) if (!done##k && o##k > lim##k) { \
        D4DRAINC(k) si##k += NCH; D4LOAD(k) }
#define D4ANY(k)  any |= (uintptr_t)!done##k;

#define GEN_K4(NAME, FORN, NCH, STEPM) \
static void NAME(const ENC *E, uint8_t *out) { \
    FORN(D4DECL) \
    for (;;) { \
        FORN##_2(D4TICK, NCH) \
        uintptr_t any = 0; FORN(D4ANY) if (!any) break; \
        FORN(STEPM) \
    } \
}
#define H4S(k) if (!done##k) H4STEP(a0##k, a1##k, a2##k, a3##k, c##k, o##k);
#define W3S(k) if (!done##k) W3STEP(a0##k, a1##k, a2##k, a3##k, c##k, o##k);

GEN_K4(dec_h4_u1, FOR1, 1, H4S)
GEN_K4(dec_h4_u4, FOR4, 4, H4S)
GEN_K4(dec_h4_u8, FOR8, 8, H4S)
#ifdef HAVE_K4R
#define K4RS(k) if (!done##k) K4R_STEP(a0##k, a1##k, a2##k, a3##k, c##k, o##k);
GEN_K4(dec_k4r_u4, FOR4, 4, K4RS)
GEN_K4(dec_k4r_u8, FOR8, 8, K4RS)
#endif
GEN_K4(dec_w3_u1, FOR1, 1, W3S)
GEN_K4(dec_w3_u4, FOR4, 4, W3S)
GEN_K4(dec_w3_u8, FOR8, 8, W3S)

#define D2DECL(k) \
    const uint8_t *L##k = 0, *R##k = 0; \
    uint32_t c##k = 0; int si##k = k, done##k = 0; \
    uint8_t *o##k = 0, *lim##k = 0; \
    D2LOAD(k)
#define D2LOAD(k) if (si##k < E->nseg) { \
        L##k = E->bk[si##k][0]; R##k = E->bk[si##k][1]; c##k = 0; \
        o##k = out + E->segoff[si##k]; \
        lim##k = out + E->segoff[si##k + 1] - MARGIN; \
    } else done##k = 1;
#define D2DRAINC(k) { const uint8_t *P_[2] = {L##k, R##k}; \
    uint8_t *e_ = out + E->segoff[si##k + 1]; uint32_t c_ = c##k; uint8_t *o_ = o##k; \
    while (o_ < e_) { uint8_t b_ = *P_[c_]++; *o_++ = b_; c_ = b_ >> 7; } }
#define D2TICK(k, NCH) if (!done##k && o##k > lim##k) { \
        D2DRAINC(k) si##k += NCH; D2LOAD(k) }

#define GEN_K2(NAME, FORN, NCH, STEPM) \
static void NAME(const ENC *E, uint8_t *out) { \
    FORN(D2DECL) \
    for (;;) { \
        FORN##_2(D2TICK, NCH) \
        uintptr_t any = 0; FORN(D4ANY) if (!any) break; \
        FORN(STEPM) \
    } \
}
#define K2NTS(k) if (!done##k) K2NT_STEP(L##k, R##k, c##k, o##k);
#define K2TS(k)  if (!done##k) K2T_STEP(L##k, R##k, c##k, o##k);

GEN_K2(dec_k2nt_u8, FOR8, 8, K2NTS)
GEN_K2(dec_k2t_u8,  FOR8, 8, K2TS)
#define K2DS(k) if (!done##k) K2D_STEP(L##k, R##k, c##k, o##k);
GEN_K2(dec_k2d_u8, FOR8, 8, K2DS)
#define K2ES(k) if (!done##k) K2E_STEP(L##k, R##k, c##k, o##k);
GEN_K2(dec_k2e_u8, FOR8, 8, K2ES)
#define K2FS(k) if (!done##k) K2F_STEP(L##k, R##k, c##k, o##k);
GEN_K2(dec_k2f_u8, FOR8, 8, K2FS)
#if !defined(__aarch64__)
#define K2MS(k) if (!done##k) K2M_STEP(L##k, R##k, c##k, o##k);
GEN_K2(dec_k2m_u8, FOR8, 8, K2MS)
#endif

/* Chain-count sweep for the K=2 (6,6)-window kernels.  The u8 interleave
 * was inherited from the o1demux winner and never questioned; the K=4
 * scalar-walk sweep showed the default-8 was optimal on only one of four
 * platforms (Zen5) — optimum was 6 on M4/Graviton4 and 4 on GNR.  More
 * chains = more independent on-chain meta-table lookups in flight, until
 * register spills (each chain holds L,R,o,lim live) turn the tick block
 * into stack traffic. */
GEN_K2(dec_k2t_u2,   FOR2,  2,  K2TS)
GEN_K2(dec_k2t_u4,   FOR4,  4,  K2TS)
GEN_K2(dec_k2t_u6,   FOR6,  6,  K2TS)
GEN_K2(dec_k2t_u12,  FOR12, 12, K2TS)
GEN_K2(dec_k2t_u16,  FOR16, 16, K2TS)
GEN_K2(dec_k2nt_u2,  FOR2,  2,  K2NTS)
GEN_K2(dec_k2nt_u4,  FOR4,  4,  K2NTS)
GEN_K2(dec_k2nt_u6,  FOR6,  6,  K2NTS)
GEN_K2(dec_k2nt_u12, FOR12, 12, K2NTS)
GEN_K2(dec_k2nt_u16, FOR16, 16, K2NTS)
#if !defined(__aarch64__)
GEN_K2(dec_k2m_u2,   FOR2,  2,  K2MS)
GEN_K2(dec_k2m_u4,   FOR4,  4,  K2MS)
GEN_K2(dec_k2m_u6,   FOR6,  6,  K2MS)
GEN_K2(dec_k2m_u12,  FOR12, 12, K2MS)
GEN_K2(dec_k2m_u16,  FOR16, 16, K2MS)
#endif
#ifdef HAVE_K2G
#define K2GS(k) if (!done##k) K2G_STEP(L##k, R##k, c##k, o##k);
GEN_K2(dec_k2g_u2, FOR2, 2, K2GS)
GEN_K2(dec_k2g_u4, FOR4, 4, K2GS)
GEN_K2(dec_k2g_u6, FOR6, 6, K2GS)
GEN_K2(dec_k2g_u8, FOR8, 8, K2GS)
#endif

/* ------- cursor compaction (MZ): guard-free steady loop ------------------ */
/* Disassembly of the u4 loops showed ~7 of ~36 instructions/step are
 * harness: the !done test+branch before every tick and step, plus the
 * `any` reduction.  Fix: keep live slots DENSE.  Slots take segments
 * first-come-first-served from a shared counter (also balances better
 * than striding); when no segment remains, the finished slot is refilled
 * by CMOVE from the last live slot and control drops into the (n-1)-slot
 * loop.  Each steady loop has no done flags, no guards, no reduction --
 * only the o>lim tick compare -- and frees the done/si registers. */
#define CDECL(k) const uint8_t *L##k = 0, *R##k = 0; uint32_t c##k = 0; \
    uint8_t *o##k = 0, *lim##k = 0;
#define CLOAD(k, s) { L##k = E->bk[s][0]; R##k = E->bk[s][1]; c##k = 0; \
        o##k = out + E->segoff[s]; \
        lim##k = out + E->segoff[(s) + 1] - MARGIN; }
#define CDRAIN(k) { const uint8_t *P_[2] = {L##k, R##k}; \
    uint8_t *e_ = lim##k + MARGIN; uint32_t c_ = c##k; uint8_t *o_ = o##k; \
    while (o_ < e_) { uint8_t b_ = *P_[c_]++; *o_++ = b_; c_ = b_ >> 7; } }
#define CTICK(n, k) if (o##k > lim##k) { CDRAIN(k) \
        if (snext < nseg) { CLOAD(k, snext) snext++; } \
        else goto fin##n##_##k; }
#define CMOVE(d, s) { L##d = L##s; R##d = R##s; c##d = c##s; \
                      o##d = o##s; lim##d = lim##s; }

#define GEN_K2C(NAME, STEPM) \
static void NAME(const ENC *E, uint8_t *out) { \
    CDECL(0) CDECL(1) CDECL(2) CDECL(3) \
    int nseg = E->nseg, snext = 0; \
    if (nseg < 1) return; \
    CLOAD(0, 0) snext = 1; if (nseg < 2) goto loop1; \
    CLOAD(1, 1) snext = 2; if (nseg < 3) goto loop2; \
    CLOAD(2, 2) snext = 3; if (nseg < 4) goto loop3; \
    CLOAD(3, 3) snext = 4; \
    for (;;) { CTICK(4, 0) CTICK(4, 1) CTICK(4, 2) CTICK(4, 3) \
               STEPM(0) STEPM(1) STEPM(2) STEPM(3) } \
    fin4_0: CMOVE(0, 3) goto loop3; \
    fin4_1: CMOVE(1, 3) goto loop3; \
    fin4_2: CMOVE(2, 3) \
    fin4_3: \
    loop3: \
    for (;;) { CTICK(3, 0) CTICK(3, 1) CTICK(3, 2) \
               STEPM(0) STEPM(1) STEPM(2) } \
    fin3_0: CMOVE(0, 2) goto loop2; \
    fin3_1: CMOVE(1, 2) \
    fin3_2: \
    loop2: \
    for (;;) { CTICK(2, 0) CTICK(2, 1) \
               STEPM(0) STEPM(1) } \
    fin2_0: CMOVE(0, 1) \
    fin2_1: \
    loop1: \
    for (;;) { CTICK(1, 0) STEPM(0) } \
    fin1_0: return; \
}

#define CS_NT(k) K2NT_STEP(L##k, R##k, c##k, o##k);
GEN_K2C(dec_k2ntc_u4, CS_NT)
#define CS_T(k) K2T_STEP(L##k, R##k, c##k, o##k);
GEN_K2C(dec_k2tc_u4, CS_T)
#if !defined(__aarch64__)
#define CS_M(k) K2M_STEP(L##k, R##k, c##k, o##k);
GEN_K2C(dec_k2mc_u4, CS_M)
#endif
#ifdef HAVE_K2G
#define CS_G(k) K2G_STEP(L##k, R##k, c##k, o##k);
GEN_K2C(dec_k2gc_u4, CS_G)
#endif
#ifdef HAVE_K2Q
#define K2QS(k) if (!done##k) K2Q_STEP(L##k, R##k, c##k, o##k);
GEN_K2(dec_k2q_u4, FOR4, 4, K2QS)
GEN_K2(dec_k2q_u6, FOR6, 6, K2QS)
#define CS_Q(k) K2Q_STEP(L##k, R##k, c##k, o##k);
GEN_K2C(dec_k2qc_u4, CS_Q)
#endif
#ifdef HAVE_K2P
#define K2PS(k) if (!done##k) K2P_STEP(L##k, R##k, c##k, o##k);
GEN_K2(dec_k2p_u4, FOR4, 4, K2PS)
#define CS_P(k) K2P_STEP(L##k, R##k, c##k, o##k);
GEN_K2C(dec_k2pc_u4, CS_P)
#endif

/* ------- traditional scalar multi-walk (non-SIMD, range-partitioned) ----- */
static uint8_t g_cls4[256], g_cls2[256];   /* same values as >>6 / >>7: forces
                                              a real load for the "class via
                                              table" (unremapped) variants */
#define D4ADECL(k) \
    const uint8_t *P##k[4] = {0, 0, 0, 0}; \
    uint32_t c##k = 0; int si##k = k, done##k = 0; \
    uint8_t *o##k = 0, *lim##k = 0; \
    D4ALOAD(k)
#define D4ALOAD(k) if (si##k < E->nseg) { \
        P##k[0] = E->bk[si##k][0]; P##k[1] = E->bk[si##k][1]; \
        P##k[2] = E->bk[si##k][2]; P##k[3] = E->bk[si##k][3]; c##k = 0; \
        o##k = out + E->segoff[si##k]; \
        lim##k = out + E->segoff[si##k + 1] - MARGIN; \
    } else done##k = 1;
#define D4ADRAINC(k) { uint8_t *e_ = out + E->segoff[si##k + 1]; \
    while (o##k < e_) { uint8_t b_ = *P##k[c##k]++; *o##k++ = b_; c##k = b_ >> 6; } }
#define D4ATICK(k, NCH) if (!done##k && o##k > lim##k) { \
        D4ADRAINC(k) si##k += NCH; D4ALOAD(k) }
#define GEN_K4A(NAME, FORN, NCH, STEPM) \
static void NAME(const ENC *E, uint8_t *out) { \
    FORN(D4ADECL) \
    for (;;) { FORN##_2(D4ATICK, NCH) \
        uintptr_t any = 0; FORN(D4ANY) if (!any) break; \
        FORN(STEPM) } \
}
#define SC4AS(k)  if (!done##k) { uint8_t b_ = *P##k[c##k]++; *o##k++ = b_; c##k = b_ >> 6; }
#define SC4ALS(k) if (!done##k) { uint8_t b_ = *P##k[c##k]++; *o##k++ = b_; c##k = g_cls4[b_]; }
#define SC4RS(k)  if (!done##k) { \
    const uint8_t *pl_ = MUXP(c##k & 1, a0##k, a1##k); \
    const uint8_t *ph_ = MUXP(c##k & 1, a2##k, a3##k); \
    const uint8_t *p_  = MUXP((c##k >> 1) & 1, pl_, ph_); \
    uint8_t b_ = *p_; *o##k++ = b_; \
    a0##k += (c##k == 0); a1##k += (c##k == 1); \
    a2##k += (c##k == 2); a3##k += (c##k == 3); \
    c##k = b_ >> 6; }
#define SC2RS(k)  if (!done##k) { \
    const uint8_t *p_ = MUXP(c##k, L##k, R##k); uint8_t b_ = *p_; \
    *o##k++ = b_; R##k += c##k; L##k += 1u - c##k; c##k = b_ >> 7; }
#define SC2RLS(k) if (!done##k) { \
    const uint8_t *p_ = MUXP(c##k, L##k, R##k); uint8_t b_ = *p_; \
    *o##k++ = b_; R##k += c##k; L##k += 1u - c##k; c##k = g_cls2[b_]; }

/* generic-K scalar walk (array cursors, class = top bits) */
#define DGADECL(k) \
    const uint8_t *P##k[16]; \
    uint32_t c##k = 0; int si##k = k, done##k = 0; \
    uint8_t *o##k = 0, *lim##k = 0; \
    DGALOAD(k)
#define DGALOAD(k) if (si##k < E->nseg) { \
        for (int q_ = 0; q_ < E->K; q_++) P##k[q_] = E->bk[si##k][q_]; \
        c##k = 0; o##k = out + E->segoff[si##k]; \
        lim##k = out + E->segoff[si##k + 1] - MARGIN; \
    } else done##k = 1;
#define DGADRAINC(k) { uint8_t *e_ = out + E->segoff[si##k + 1]; \
    while (o##k < e_) { uint8_t b_ = *P##k[c##k]++; *o##k++ = b_; c##k = b_ >> SHIFT_; } }
#define DGATICK(k, NCH) if (!done##k && o##k > lim##k) { \
        DGADRAINC(k) si##k += NCH; DGALOAD(k) }
#define DGAS(k) if (!done##k) { uint8_t b_ = *P##k[c##k]++; *o##k++ = b_; c##k = b_ >> SHIFT_; }
#define GEN_KGA(NAME, FORN, NCH, SHIFT) \
static void NAME(const ENC *E, uint8_t *out) { \
    enum { SHIFT_ = SHIFT }; \
    FORN(DGADECL) \
    for (;;) { FORN##_2(DGATICK, NCH) \
        uintptr_t any = 0; FORN(D4ANY) if (!any) break; \
        FORN(DGAS) } \
}
GEN_KGA(dec_sc8a_u8,  FOR8, 8, 5)
GEN_KGA(dec_sc16a_u8, FOR8, 8, 4)

/* routed walk: class sequence PRE-DECODED (models ml|ll etc. where routing
 * comes from an already-decoded stream) -> no data-dependent chain at all */
static const uint8_t *g_route;
#define DGRDECL(k) \
    const uint8_t *P##k[16]; const uint8_t *rt##k = 0; \
    int si##k = k, done##k = 0; \
    uint8_t *o##k = 0, *lim##k = 0; \
    DGRLOAD(k)
#define DGRLOAD(k) if (si##k < E->nseg) { \
        for (int q_ = 0; q_ < E->K; q_++) P##k[q_] = E->bk[si##k][q_]; \
        o##k = out + E->segoff[si##k]; rt##k = g_route + E->segoff[si##k]; \
        lim##k = out + E->segoff[si##k + 1] - MARGIN; \
    } else done##k = 1;
#define DGRDRAINC(k) { uint8_t *e_ = out + E->segoff[si##k + 1]; \
    while (o##k < e_) { *o##k++ = *P##k[*rt##k++]++; } }
#define DGRTICK(k, NCH) if (!done##k && o##k > lim##k) { \
        DGRDRAINC(k) si##k += NCH; DGRLOAD(k) }
#define DGRS(k) if (!done##k) { *o##k++ = *P##k[*rt##k++]++; }
#define GEN_KGR(NAME, FORN, NCH) \
static void NAME(const ENC *E, uint8_t *out) { \
    FORN(DGRDECL) \
    for (;;) { FORN##_2(DGRTICK, NCH) \
        uintptr_t any = 0; FORN(D4ANY) if (!any) break; \
        FORN(DGRS) } \
}
GEN_KGR(dec_scr8_u8, FOR8, 8)
GEN_KGR(dec_scr8_u1, FOR1, 1)

GEN_K4A(dec_sc4a_u2,   FOR2,  2,  SC4AS)
GEN_K4A(dec_sc4a_u4c,  FOR4,  4,  SC4AS)
GEN_K4A(dec_sc4a_u6,   FOR6,  6,  SC4AS)
GEN_K4A(dec_sc4a_u12,  FOR12, 12, SC4AS)
GEN_K4A(dec_sc4a_u24,  FOR24, 24, SC4AS)
GEN_K4A(dec_sc4a_u8,   FOR8,  8,  SC4AS)
GEN_K4A(dec_sc4al_u8,  FOR8,  8,  SC4ALS)
GEN_K4A(dec_sc4a_u16,  FOR16, 16, SC4AS)
GEN_K4A(dec_sc4al_u16, FOR16, 16, SC4ALS)
GEN_K4(dec_sc4r_u8,    FOR8,  8,  SC4RS)
GEN_K4(dec_sc4r_u16,   FOR16, 16, SC4RS)
GEN_K2(dec_sc2r_u8,    FOR8,  8,  SC2RS)
GEN_K2(dec_sc2rl_u8,   FOR8,  8,  SC2RLS)
GEN_K2(dec_sc2r_u16,   FOR16, 16, SC2RS)
GEN_K2(dec_sc2rl_u16,  FOR16, 16, SC2RLS)

/* ------- k4v: MZ's SIMD multi-cursor walk -- 8 chains in zmm lanes --------
 * The k4r arithmetic walk vectorized ACROSS chains: per-lane H/R/C/P in
 * 512-bit registers, per-lane variable shifts run all 8 walks at once.
 * x86 AVX-512 only (8 lanes; NEON's 2 lanes can't amortize).            */
#if defined(__AVX512VBMI__) && defined(__AVX512VL__) && defined(__AVX512BW__)
#define HAVE_K4V 1
static void k4v_loadseg(const ENC *E, uint8_t *out, const uint8_t *A[4],
                        uint8_t **o, uint8_t **lim, uint64_t *c, int *si, int *done) {
    if (*si < E->nseg) {
        for (int b = 0; b < 4; b++) A[b] = E->bk[*si][b];
        *c = 0; *o = out + E->segoff[*si];
        *lim = out + E->segoff[*si + 1] - MARGIN;
    } else *done = 1;
}
static void dec_k4v_u8(const ENC *E, uint8_t *out) {
    const uint8_t *A[8][4]; uint8_t *o[8], *lim[8]; int si[8], done[8];
    uint64_t cs[8] __attribute__((aligned(64)));
    uint64_t hs[8] __attribute__((aligned(64)));
    uint64_t ps[8] __attribute__((aligned(64)));
    uint64_t rs[8] __attribute__((aligned(64)));
    uint64_t L[8][4];
    memset(L, 0, sizeof L);
    for (int k = 0; k < 8; k++) { si[k] = k; done[k] = 0; cs[k] = 0;
        k4v_loadseg(E, out, A[k], &o[k], &lim[k], &cs[k], &si[k], &done[k]); }
    const __m512i M255 = _mm512_set1_epi64(255), M3 = _mm512_set1_epi64(3);
    const __m512i ONE = _mm512_set1_epi64(1), R0 = _mm512_set1_epi64(0x18100800);
    for (;;) {
        int any = 0;
        for (int k = 0; k < 8; k++) {
            if (!done[k] && o[k] > lim[k]) {
                uint8_t *e_ = out + E->segoff[si[k] + 1];
                while (o[k] < e_) { uint8_t b_ = *A[k][cs[k]]++; *o[k]++ = b_; cs[k] = b_ >> 6; }
                si[k] += 8; k4v_loadseg(E, out, A[k], &o[k], &lim[k], &cs[k], &si[k], &done[k]);
            }
            any |= !done[k];
        }
        if (!any) break;
        for (int k = 0; k < 8; k++) {
            if (done[k]) { hs[k] = 0; continue; }
            memcpy(&L[k][0], A[k][0], 8); memcpy(&L[k][1], A[k][1], 8);
            memcpy(&L[k][2], A[k][2], 8); memcpy(&L[k][3], A[k][3], 8);
            __m128i w01 = _mm_set_epi64x((long long)L[k][1], (long long)L[k][0]);
            __m128i w23 = _mm_set_epi64x((long long)L[k][3], (long long)L[k][2]);
            uint32_t b7a = (uint32_t)_mm_movemask_epi8(w01);
            uint32_t b6a = (uint32_t)_mm_movemask_epi8(_mm_add_epi8(w01, w01));
            uint32_t b7b = (uint32_t)_mm_movemask_epi8(w23);
            uint32_t b6b = (uint32_t)_mm_movemask_epi8(_mm_add_epi8(w23, w23));
            hs[k] = (uint64_t)(_pdep_u32(b7a, 0xAAAAAAAAu) | _pdep_u32(b6a, 0x55555555u))
                  | ((uint64_t)(_pdep_u32(b7b, 0xAAAAAAAAu) | _pdep_u32(b6b, 0x55555555u)) << 32);
        }
        __m512i H = _mm512_load_si512((const void *)hs);
        __m512i C = _mm512_load_si512((const void *)cs);
        __m512i R = R0, P = _mm512_setzero_si512();
        #define K4V_W(k) {             __m512i S3 = _mm512_slli_epi64(C, 3);             __m512i REF = _mm512_and_si512(_mm512_srlv_epi64(R, S3), M255);             P = _mm512_or_si512(P, _mm512_slli_epi64(REF, 8 * (k)));             C = _mm512_and_si512(_mm512_srlv_epi64(H, _mm512_add_epi64(REF, REF)), M3);             R = _mm512_add_epi64(R, _mm512_sllv_epi64(ONE, S3)); }
        K4V_W(0) K4V_W(1) K4V_W(2) K4V_W(3) K4V_W(4) K4V_W(5) K4V_W(6) K4V_W(7)
        #undef K4V_W
        _mm512_store_si512((void *)ps, P);
        _mm512_store_si512((void *)rs, R);
        _mm512_store_si512((void *)cs, C);
        for (int k = 0; k < 8; k++) {
            if (done[k]) continue;
            __m256i S = _mm256_set_epi64x((long long)L[k][3], (long long)L[k][2],
                                          (long long)L[k][1], (long long)L[k][0]);
            __m256i I = _mm256_castsi128_si256(_mm_cvtsi64_si128((long long)ps[k]));
            _mm_storel_epi64((__m128i *)(void *)o[k],
                _mm256_castsi256_si128(_mm256_permutexvar_epi8(I, S)));
            uint32_t d = (uint32_t)rs[k] - 0x18100800u;
            A[k][0] += d & 255; A[k][1] += (d >> 8) & 255;
            A[k][2] += (d >> 16) & 255; A[k][3] += d >> 24;
            o[k] += 8; cs[k] &= 3;
        }
    }
}
static void dec_k4v_u16(const ENC *E, uint8_t *out) {
    const uint8_t *A[16][4]; uint8_t *o[16], *lim[16]; int si[16], done[16];
    uint64_t cs[16] __attribute__((aligned(64)));
    uint64_t hs[16] __attribute__((aligned(64)));
    uint64_t ps[16] __attribute__((aligned(64)));
    uint64_t rs[16] __attribute__((aligned(64)));
    uint64_t L[16][4];
    memset(L, 0, sizeof L);
    for (int k = 0; k < 16; k++) { si[k] = k; done[k] = 0; cs[k] = 0;
        k4v_loadseg(E, out, A[k], &o[k], &lim[k], &cs[k], &si[k], &done[k]); }
    const __m512i M255 = _mm512_set1_epi64(255), M3 = _mm512_set1_epi64(3);
    const __m512i ONE = _mm512_set1_epi64(1), R0 = _mm512_set1_epi64(0x18100800);
    for (;;) {
        int any = 0;
        for (int k = 0; k < 16; k++) {
            if (!done[k] && o[k] > lim[k]) {
                uint8_t *e_ = out + E->segoff[si[k] + 1];
                while (o[k] < e_) { uint8_t b_ = *A[k][cs[k]]++; *o[k]++ = b_; cs[k] = b_ >> 6; }
                si[k] += 16; k4v_loadseg(E, out, A[k], &o[k], &lim[k], &cs[k], &si[k], &done[k]);
            }
            any |= !done[k];
        }
        if (!any) break;
        for (int k = 0; k < 16; k++) {
            if (done[k]) { hs[k] = 0; continue; }
            memcpy(&L[k][0], A[k][0], 8); memcpy(&L[k][1], A[k][1], 8);
            memcpy(&L[k][2], A[k][2], 8); memcpy(&L[k][3], A[k][3], 8);
            __m128i w01 = _mm_set_epi64x((long long)L[k][1], (long long)L[k][0]);
            __m128i w23 = _mm_set_epi64x((long long)L[k][3], (long long)L[k][2]);
            uint32_t b7a = (uint32_t)_mm_movemask_epi8(w01);
            uint32_t b6a = (uint32_t)_mm_movemask_epi8(_mm_add_epi8(w01, w01));
            uint32_t b7b = (uint32_t)_mm_movemask_epi8(w23);
            uint32_t b6b = (uint32_t)_mm_movemask_epi8(_mm_add_epi8(w23, w23));
            hs[k] = (uint64_t)(_pdep_u32(b7a, 0xAAAAAAAAu) | _pdep_u32(b6a, 0x55555555u))
                  | ((uint64_t)(_pdep_u32(b7b, 0xAAAAAAAAu) | _pdep_u32(b6b, 0x55555555u)) << 32);
        }
        for (int g = 0; g < 2; g++) {
            __m512i H = _mm512_load_si512((const void *)(hs + 8 * g));
            __m512i C = _mm512_load_si512((const void *)(cs + 8 * g));
            __m512i R = R0, P = _mm512_setzero_si512();
            #define K4V_W2(k) { \
                __m512i S3 = _mm512_slli_epi64(C, 3); \
                __m512i REF = _mm512_and_si512(_mm512_srlv_epi64(R, S3), M255); \
                P = _mm512_or_si512(P, _mm512_slli_epi64(REF, 8 * (k))); \
                C = _mm512_and_si512(_mm512_srlv_epi64(H, _mm512_add_epi64(REF, REF)), M3); \
                R = _mm512_add_epi64(R, _mm512_sllv_epi64(ONE, S3)); }
            K4V_W2(0) K4V_W2(1) K4V_W2(2) K4V_W2(3) K4V_W2(4) K4V_W2(5) K4V_W2(6) K4V_W2(7)
            #undef K4V_W2
            _mm512_store_si512((void *)(ps + 8 * g), P);
            _mm512_store_si512((void *)(rs + 8 * g), R);
            _mm512_store_si512((void *)(cs + 8 * g), C);
        }
        for (int k = 0; k < 16; k++) {
            if (done[k]) continue;
            __m256i S = _mm256_set_epi64x((long long)L[k][3], (long long)L[k][2],
                                          (long long)L[k][1], (long long)L[k][0]);
            __m256i I = _mm256_castsi128_si256(_mm_cvtsi64_si128((long long)ps[k]));
            _mm_storel_epi64((__m128i *)(void *)o[k],
                _mm256_castsi256_si128(_mm256_permutexvar_epi8(I, S)));
            uint32_t d = (uint32_t)rs[k] - 0x18100800u;
            A[k][0] += d & 255; A[k][1] += (d >> 8) & 255;
            A[k][2] += (d >> 16) & 255; A[k][3] += d >> 24;
            o[k] += 8;
        }
    }
}
static void dec_k4v_u24(const ENC *E, uint8_t *out) {
    const uint8_t *A[24][4]; uint8_t *o[24], *lim[24]; int si[24], done[24];
    uint64_t cs[24] __attribute__((aligned(64)));
    uint64_t hs[24] __attribute__((aligned(64)));
    uint64_t ps[24] __attribute__((aligned(64)));
    uint64_t rs[24] __attribute__((aligned(64)));
    uint64_t L[24][4];
    memset(L, 0, sizeof L);
    for (int k = 0; k < 24; k++) { si[k] = k; done[k] = 0; cs[k] = 0;
        k4v_loadseg(E, out, A[k], &o[k], &lim[k], &cs[k], &si[k], &done[k]); }
    const __m512i M255 = _mm512_set1_epi64(255), M3 = _mm512_set1_epi64(3);
    const __m512i ONE = _mm512_set1_epi64(1), R0 = _mm512_set1_epi64(0x18100800);
    for (;;) {
        int any = 0;
        for (int k = 0; k < 24; k++) {
            if (!done[k] && o[k] > lim[k]) {
                uint8_t *e_ = out + E->segoff[si[k] + 1];
                while (o[k] < e_) { uint8_t b_ = *A[k][cs[k]]++; *o[k]++ = b_; cs[k] = b_ >> 6; }
                si[k] += 24; k4v_loadseg(E, out, A[k], &o[k], &lim[k], &cs[k], &si[k], &done[k]);
            }
            any |= !done[k];
        }
        if (!any) break;
        for (int k = 0; k < 24; k++) {
            if (done[k]) { hs[k] = 0; continue; }
            memcpy(&L[k][0], A[k][0], 8); memcpy(&L[k][1], A[k][1], 8);
            memcpy(&L[k][2], A[k][2], 8); memcpy(&L[k][3], A[k][3], 8);
            __m128i w01 = _mm_set_epi64x((long long)L[k][1], (long long)L[k][0]);
            __m128i w23 = _mm_set_epi64x((long long)L[k][3], (long long)L[k][2]);
            uint32_t b7a = (uint32_t)_mm_movemask_epi8(w01);
            uint32_t b6a = (uint32_t)_mm_movemask_epi8(_mm_add_epi8(w01, w01));
            uint32_t b7b = (uint32_t)_mm_movemask_epi8(w23);
            uint32_t b6b = (uint32_t)_mm_movemask_epi8(_mm_add_epi8(w23, w23));
            hs[k] = (uint64_t)(_pdep_u32(b7a, 0xAAAAAAAAu) | _pdep_u32(b6a, 0x55555555u))
                  | ((uint64_t)(_pdep_u32(b7b, 0xAAAAAAAAu) | _pdep_u32(b6b, 0x55555555u)) << 32);
        }
        for (int g = 0; g < 3; g++) {
            __m512i H = _mm512_load_si512((const void *)(hs + 8 * g));
            __m512i C = _mm512_load_si512((const void *)(cs + 8 * g));
            __m512i R = R0, P = _mm512_setzero_si512();
            #define K4V_W2(k) { \
                __m512i S3 = _mm512_slli_epi64(C, 3); \
                __m512i REF = _mm512_and_si512(_mm512_srlv_epi64(R, S3), M255); \
                P = _mm512_or_si512(P, _mm512_slli_epi64(REF, 8 * (k))); \
                C = _mm512_and_si512(_mm512_srlv_epi64(H, _mm512_add_epi64(REF, REF)), M3); \
                R = _mm512_add_epi64(R, _mm512_sllv_epi64(ONE, S3)); }
            K4V_W2(0) K4V_W2(1) K4V_W2(2) K4V_W2(3) K4V_W2(4) K4V_W2(5) K4V_W2(6) K4V_W2(7)
            #undef K4V_W2
            _mm512_store_si512((void *)(ps + 8 * g), P);
            _mm512_store_si512((void *)(rs + 8 * g), R);
            _mm512_store_si512((void *)(cs + 8 * g), C);
        }
        for (int k = 0; k < 24; k++) {
            if (done[k]) continue;
            __m256i S = _mm256_set_epi64x((long long)L[k][3], (long long)L[k][2],
                                          (long long)L[k][1], (long long)L[k][0]);
            __m256i I = _mm256_castsi128_si256(_mm_cvtsi64_si128((long long)ps[k]));
            _mm_storel_epi64((__m128i *)(void *)o[k],
                _mm256_castsi256_si128(_mm256_permutexvar_epi8(I, S)));
            uint32_t d = (uint32_t)rs[k] - 0x18100800u;
            A[k][0] += d & 255; A[k][1] += (d >> 8) & 255;
            A[k][2] += (d >> 16) & 255; A[k][3] += d >> 24;
            o[k] += 8;
        }
    }
}
/* k4w: k4v with VECTORIZED header build -- gathers + in-lane 2-bit extract */
static void dec_k4w_u16(const ENC *E, uint8_t *out) {
    uint64_t ap[16][4] __attribute__((aligned(64)));   /* cursor addresses */
    uint8_t *o[16], *lim[16]; int si[16], done[16];
    uint64_t cs[16] __attribute__((aligned(64)));
    uint64_t ps[16] __attribute__((aligned(64)));
    uint64_t rs[16] __attribute__((aligned(64)));
    uint64_t Lg[2][4][8] __attribute__((aligned(64)));  /* [group][bucket][chain] */
    for (int k = 0; k < 16; k++) {
        si[k] = k; done[k] = 0; cs[k] = 0;
        const uint8_t *A4[4] = {0,0,0,0};
        k4v_loadseg(E, out, A4, &o[k], &lim[k], &cs[k], &si[k], &done[k]);
        for (int b = 0; b < 4; b++) ap[k][b] = (uint64_t)(uintptr_t)A4[b];
    }
    const __m512i M255 = _mm512_set1_epi64(255), M3 = _mm512_set1_epi64(3);
    const __m512i ONE = _mm512_set1_epi64(1), R0 = _mm512_set1_epi64(0x18100800);
    const __m512i T03 = _mm512_set1_epi64(0x0303030303030303ULL);
    const __m512i M32 = _mm512_set1_epi64(0xFFFFFFFFULL);
    const __m512i MAG = _mm512_set1_epi64(0x0104104000000000ULL);
    for (;;) {
        /* tick: a chain within MARGIN of its segment end finishes the tail
         * with the exact scalar walk (a 16B bulk store may not cross the
         * boundary), then hops to its next segment.  Rare and predictable. */
        int any = 0;
        for (int k = 0; k < 16; k++) {
            if (!done[k] && o[k] > lim[k]) {
                uint8_t *e_ = out + E->segoff[si[k] + 1];
                const uint8_t *A4[4] = {(const uint8_t *)(uintptr_t)ap[k][0],
                    (const uint8_t *)(uintptr_t)ap[k][1],
                    (const uint8_t *)(uintptr_t)ap[k][2],
                    (const uint8_t *)(uintptr_t)ap[k][3]};
                while (o[k] < e_) { uint8_t b_ = *A4[cs[k]]++; *o[k]++ = b_; cs[k] = b_ >> 6; }
                si[k] += 16;                          /* stride to next segment */
                k4v_loadseg(E, out, A4, &o[k], &lim[k], &cs[k], &si[k], &done[k]);
                for (int b = 0; b < 4; b++) ap[k][b] = (uint64_t)(uintptr_t)A4[b];
            }
            any |= !done[k];
        }
        if (!any) break;
        for (int g = 0; g < 2; g++) {
            /* gather 8 chains x 4 buckets; extract 2-bit classes in-lane */
            __m512i H = _mm512_setzero_si512();
            for (int b = 0; b < 4; b++) {
                __m512i idx = _mm512_setr_epi64(
                    (long long)ap[8*g+0][b], (long long)ap[8*g+1][b],
                    (long long)ap[8*g+2][b], (long long)ap[8*g+3][b],
                    (long long)ap[8*g+4][b], (long long)ap[8*g+5][b],
                    (long long)ap[8*g+6][b], (long long)ap[8*g+7][b]);
                __m512i W = _mm512_i64gather_epi64(idx, (const void *)0, 1);
                _mm512_store_si512((void *)Lg[g][b], W);
                __m512i T = _mm512_and_si512(_mm512_srli_epi64(W, 6), T03);
                __m512i LO = _mm512_srli_epi64(
                    _mm512_mullo_epi64(_mm512_and_si512(T, M32), MAG), 56);
                __m512i HI = _mm512_srli_epi64(
                    _mm512_mullo_epi64(_mm512_srli_epi64(T, 32), MAG), 56);
                __m512i B = _mm512_or_si512(LO, _mm512_slli_epi64(HI, 8));
                H = _mm512_or_si512(H, _mm512_slli_epi64(B, 16 * b));
            }
            __m512i C = _mm512_load_si512((const void *)(cs + 8 * g));
            __m512i R = R0, P = _mm512_setzero_si512();
            #define K4W_W(k) { \
                __m512i S3 = _mm512_slli_epi64(C, 3); \
                __m512i REF = _mm512_and_si512(_mm512_srlv_epi64(R, S3), M255); \
                P = _mm512_or_si512(P, _mm512_slli_epi64(REF, 8 * (k))); \
                C = _mm512_and_si512(_mm512_srlv_epi64(H, _mm512_add_epi64(REF, REF)), M3); \
                R = _mm512_add_epi64(R, _mm512_sllv_epi64(ONE, S3)); }
            K4W_W(0) K4W_W(1) K4W_W(2) K4W_W(3) K4W_W(4) K4W_W(5) K4W_W(6) K4W_W(7)
            #undef K4W_W
            _mm512_store_si512((void *)(ps + 8 * g), P);
            _mm512_store_si512((void *)(rs + 8 * g), R);
            _mm512_store_si512((void *)(cs + 8 * g), C);
        }
        for (int k = 0; k < 16; k++) {
            if (done[k]) continue;
            int g = k >> 3, j = k & 7;
            __m256i S = _mm256_set_epi64x(
                (long long)Lg[g][3][j], (long long)Lg[g][2][j],
                (long long)Lg[g][1][j], (long long)Lg[g][0][j]);
            __m256i I = _mm256_castsi128_si256(_mm_cvtsi64_si128((long long)ps[k]));
            _mm_storel_epi64((__m128i *)(void *)o[k],
                _mm256_castsi256_si128(_mm256_permutexvar_epi8(I, S)));
            uint32_t d = (uint32_t)rs[k] - 0x18100800u;
            ap[k][0] += d & 255; ap[k][1] += (d >> 8) & 255;
            ap[k][2] += (d >> 16) & 255; ap[k][3] += d >> 24;
            o[k] += 8;
        }
    }
}
/* ===========================================================================
 * k4x -- register-arithmetic K=4 demux, SIMD across chains (x86 AVX-512).
 *
 * PROBLEM.  Re-interleave four per-class literal streams where the class of
 * the previous OUTPUT byte (top 2 bits, after remap) selects which stream the
 * next byte comes from.  The routing walk is inherently serial per chain; the
 * design runs 16 independent chains (striding segments) and executes their
 * walks lane-parallel: chain j of group g lives in u64 lane j of 512-bit
 * registers, so one vpsrlvq advances all 8 walks of a group at once.  This is
 * the one trick lookup-table kernels can never copy: a table step is a load,
 * and loads do not vectorize across chains; pure ALU does.
 *
 * THE CENTRAL IDENTITY.  Per chain, the 4 bucket windows are staged into one
 * contiguous 64-byte block SRC (bucket b at bytes [16b, 16b+16)).  Define
 *     ref = 16*b + n     (n = elements consumed from bucket b this iteration)
 * Then ref is simultaneously
 *     - the vpermb byte index into SRC (emit),
 *     - the bit index into both class-bit planes (walk), and
 *     - the value the per-bucket counter register produces directly
 *       (R holds one byte per bucket, bases 0x00/10/20/30, +1 per consume).
 * One number, three roles; the walk never touches memory.
 *
 * HEADER PLANES (PH-style bit-planes, not packed fields).  The two class
 * bits of the 64 staged source bytes are kept as two separate 64-bit planes:
 *     B7 bit ref = bit 7 of SRC[ref]      B6 bit ref = bit 6 of SRC[ref]
 * Built with 2 movemasks per 16B bucket load (paddb x,x lifts bit6 to the
 * sign position) and plain shift/or concatenation -- no pdep.  Packing into
 * 2-bit fields instead (k4v did) needs pdep headers and, at 16-deep, 128 bits
 * per chain; the planes fit one u64 lane each and index directly by ref.
 *
 * WHY 16-DEEP / 16 PER ITERATION.  Stocks hold 16 headers per bucket and the
 * walk runs exactly 16 steps, so an iteration can never stall or overshoot
 * (per-bucket consumption <= 16 = stock depth): every iteration emits exactly
 * 16 bytes, unconditionally -- no data-dependent yield, no terminal case, no
 * fallback branch.  Doubling the radius from 8 (k4v) halves every fixed
 * per-iteration cost (headers, emit, cursor updates, guards) per output byte.
 *
 * WALK STEP (12 ops, all lane-parallel; recurrence C -> S3 -> REF -> planes
 * -> ternlog -> C):
 *     ref-select  3   S3 = C<<3;  REF = (R >> S3) & 255
 *                     (the & 255 is IRREDUCIBLE: vpsrlvq zeroes on counts
 *                      > 63, so garbage high bytes in REF poison the plane
 *                      reads if unmasked)
 *     pattern     2   P |= REF << 8k         (k = step, unrolled constant)
 *     class       5   A2 = (B7 >> REF) << 1;  T6 = (B6 >> REF) & 1;
 *                     C = ternlog(A2, T6, 2, 0xE4)   -- bit1 from A2,
 *                     bit0 from T6, rest zero.
 *                     WARNING: the tempting fold "((B7>>REF)<<1 | (B6>>REF))
 *                     & 3" is WRONG -- bit 1 of the unmasked B6 term leaks
 *                     into the class (caught by selftest); both single-bit
 *                     isolations are load-bearing.
 *     counters    2   R += 1 << S3
 * Rejected cheaper-step candidates (all measured or arithmetically closed):
 * vpmultishiftqb ties vpsrlvq for one field/lane; GFNI is per-byte only;
 * vpmovqb pattern stores need a step-major->chain-major transpose the emit
 * cannot absorb; packed-2bit H with pre-doubled refs ~= wash vs pdep headers.
 *
 * STRUCTURE PER ITERATION: (1) tick -- drain/advance chains past their
 * segment limit (scalar; rare, predictable); (2) headers+staging per chain;
 * (3) the two 16-step lane-parallel walks; (4) per-chain emit: ONE
 * vpermb-512 over SRC using the 16 pattern bytes, one 16B store, cursor
 * advance from R deltas.  SIMD<->scalar state crosses through aligned arrays
 * (cs/h7s/h6s/p0s/p1s/rs): one store/load per zmm per iteration, and the
 * scalar drain/tick sees any chain's state naturally.
 *
 * HISTORY/PERF (x-ray/mozilla/dickens lit streams): scalar walk 0.67-0.90
 * ns/B -> k4v (8-deep, pdep headers, 2x vpermb-256) 0.51-0.70 -> k4x
 * 0.50-0.61 -> +vpternlogq 0.485-0.488 (Zen5) / 0.582-0.587 (GNR); u24
 * lane-groups ~wash (chain hidden at u16); vpgatherqq header build -45-52%
 * (gathers lose ~2x to scalar loads + movemask).  M4/Graviton have no
 * 8-lane-64b ALU: the 6-walk scalar kernel keeps those platforms.
 * ======================================================================== */
static void dec_k4x_u16(const ENC *E, uint8_t *out) {
    uint64_t ap[16][4] __attribute__((aligned(64)));   /* bucket read cursors */
    uint8_t *o[16], *lim[16]; int si[16], done[16];    /* per-chain segment st.*/
    uint64_t cs[16] __attribute__((aligned(64)));      /* current class/chain  */
    uint64_t h7s[16] __attribute__((aligned(64)));     /* bit-7 class plane    */
    uint64_t h6s[16] __attribute__((aligned(64)));     /* bit-6 class plane    */
    uint64_t p0s[16] __attribute__((aligned(64)));     /* pattern bytes 0..7   */
    uint64_t p1s[16] __attribute__((aligned(64)));     /* pattern bytes 8..15  */
    uint64_t rs[16] __attribute__((aligned(64)));      /* final R (cursor d's) */
    uint8_t SRC[16][64] __attribute__((aligned(64)));  /* staged 4x16B sources */
    /* chain k starts on segment k and strides by 16; class resets to 0 at
     * every segment boundary (matches the encoder's per-segment reset). */
    for (int k = 0; k < 16; k++) {
        si[k] = k; done[k] = 0; cs[k] = 0;
        const uint8_t *A4[4] = {0,0,0,0};
        k4v_loadseg(E, out, A4, &o[k], &lim[k], &cs[k], &si[k], &done[k]);
        for (int b = 0; b < 4; b++) ap[k][b] = (uint64_t)(uintptr_t)A4[b];
    }
    const __m512i M255 = _mm512_set1_epi64(255);   /* isolate REF byte        */
    const __m512i ONEZ = _mm512_set1_epi64(1);     /* bit isolate / increment */
    const __m512i TWOZ = _mm512_set1_epi64(2);     /* ternlog bit-1 selector  */
    const __m512i R0 = _mm512_set1_epi64(0x30201000); /* ref bases: byte b of
        each lane = 16*b, i.e. bucket b's first staged byte in SRC          */
    for (;;) {
        /* tick: a chain within MARGIN of its segment end finishes the tail
         * with the exact scalar walk (a 16B bulk store may not cross the
         * boundary), then hops to its next segment.  Rare and predictable. */
        int any = 0;
        for (int k = 0; k < 16; k++) {
            if (!done[k] && o[k] > lim[k]) {
                uint8_t *e_ = out + E->segoff[si[k] + 1];
                const uint8_t *A4[4] = {(const uint8_t *)(uintptr_t)ap[k][0],
                    (const uint8_t *)(uintptr_t)ap[k][1],
                    (const uint8_t *)(uintptr_t)ap[k][2],
                    (const uint8_t *)(uintptr_t)ap[k][3]};
                while (o[k] < e_) { uint8_t b_ = *A4[cs[k]]++; *o[k]++ = b_; cs[k] = b_ >> 6; }
                si[k] += 16;                          /* stride to next segment */
                k4v_loadseg(E, out, A4, &o[k], &lim[k], &cs[k], &si[k], &done[k]);
                for (int b = 0; b < 4; b++) ap[k][b] = (uint64_t)(uintptr_t)A4[b];
            }
            any |= !done[k];
        }
        if (!any) break;
        /* headers + staging: 4 contiguous 16B loads per chain (stream pads
         * guarantee readability past the true end; the walk provably never
         * consumes beyond it -- routing follows the true sequence).  paddb
         * x,x shifts each byte left 1 so movemask reads bit 6. */
        for (int k = 0; k < 16; k++) {
            if (done[k]) { h7s[k] = h6s[k] = 0; continue; }
            uint64_t b7 = 0, b6 = 0;
            for (int b = 0; b < 4; b++) {
                __m128i x = _mm_loadu_si128((const __m128i *)(const void *)(uintptr_t)ap[k][b]);
                _mm_store_si128((__m128i *)(void *)(SRC[k] + 16 * b), x);
                b7 |= (uint64_t)(uint32_t)_mm_movemask_epi8(x) << (16 * b);
                b6 |= (uint64_t)(uint32_t)_mm_movemask_epi8(_mm_add_epi8(x, x)) << (16 * b);
            }
            h7s[k] = b7; h6s[k] = b6;
        }
        /* the walk: two groups of 8 chains, chain j of group g in u64 lane
         * j; 16 fully unrolled lane-parallel steps; no memory access.  Loop-
         * carried recurrence per step: C -> S3 -> REF -> plane reads ->
         * ternlog -> C (~6 cy); two groups interleave to hide it. */
        for (int g = 0; g < 2; g++) {
            __m512i B7 = _mm512_load_si512((const void *)(h7s + 8 * g));
            __m512i B6 = _mm512_load_si512((const void *)(h6s + 8 * g));
            __m512i C = _mm512_load_si512((const void *)(cs + 8 * g));
            __m512i R = R0, P0 = _mm512_setzero_si512(), P1 = _mm512_setzero_si512();
            #define K4X_W(k) { \
                /* S3 = 8*class: byte offset of current bucket's counter   */ \
                __m512i S3 = _mm512_slli_epi64(C, 3); \
                /* REF = R's byte for that bucket = 16*b + n; the &255 is  */ \
                /* load-bearing: vpsrlvq zeroes on counts > 63, so garbage */ \
                /* high bytes would poison the plane reads below           */ \
                __m512i REF = _mm512_and_si512(_mm512_srlv_epi64(R, S3), M255); \
                /* record REF as pattern byte k (vpermb index into SRC)    */ \
                if ((k) < 8) P0 = _mm512_or_si512(P0, _mm512_slli_epi64(REF, 8 * ((k) & 7))); \
                else         P1 = _mm512_or_si512(P1, _mm512_slli_epi64(REF, 8 * ((k) & 7))); \
                /* next class = (bit7<<1)|bit6 of the consumed source byte, */ \
                /* read from the two planes at bit REF.  A2 keeps garbage   */ \
                /* above bit 1, T6 is isolated to bit 0; the 0xE4 ternlog   */ \
                /* takes bit 1 from A2, bit 0 from T6, zeroes the rest.     */ \
                /* NB do NOT fold to ((B7>>REF)<<1 | (B6>>REF)) & 3: bit 1  */ \
                /* of the unmasked B6 term leaks into the class.            */ \
                __m512i A2 = _mm512_slli_epi64(_mm512_srlv_epi64(B7, REF), 1); \
                __m512i T6 = _mm512_and_si512(_mm512_srlv_epi64(B6, REF), ONEZ); \
                C = _mm512_ternarylogic_epi64(A2, T6, TWOZ, 0xE4); \
                /* consume: +1 to the current bucket's counter byte         */ \
                R = _mm512_add_epi64(R, _mm512_sllv_epi64(ONEZ, S3)); }
            K4X_W(0) K4X_W(1) K4X_W(2) K4X_W(3) K4X_W(4) K4X_W(5) K4X_W(6) K4X_W(7)
            K4X_W(8) K4X_W(9) K4X_W(10) K4X_W(11) K4X_W(12) K4X_W(13) K4X_W(14) K4X_W(15)
            #undef K4X_W
            /* hand results back to the scalar side (emit + tick/drain)    */
            _mm512_store_si512((void *)(p0s + 8 * g), P0);
            _mm512_store_si512((void *)(p1s + 8 * g), P1);
            _mm512_store_si512((void *)(rs + 8 * g), R);
            _mm512_store_si512((void *)(cs + 8 * g), C);   /* class carries  */
        }
        /* emit: the 16 pattern bytes ARE vpermb indices into the staged
         * 64B source; one permute + one 16B store per chain.  Cursor deltas
         * fall out of R: byte b of (R - R0) = elements consumed from bucket
         * b this iteration. */
        for (int k = 0; k < 16; k++) {
            if (done[k]) continue;
            __m512i S = _mm512_load_si512((const void *)SRC[k]);
            __m512i I = _mm512_castsi128_si512(
                _mm_set_epi64x((long long)p1s[k], (long long)p0s[k]));
            _mm_storeu_si128((__m128i *)(void *)o[k],
                _mm512_castsi512_si128(_mm512_permutexvar_epi8(I, S)));
            uint32_t d = (uint32_t)rs[k] - 0x30201000u;
            ap[k][0] += d & 255; ap[k][1] += (d >> 8) & 255;
            ap[k][2] += (d >> 16) & 255; ap[k][3] += d >> 24;
            o[k] += 16;
        }
    }
}
#endif /* HAVE_K4V */

/* ---------------- harness ------------------------------------------------- */
static double now(void) {
    struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + 1e-9 * ts.tv_nsec;
}

typedef void (*decfn)(const ENC *, uint8_t *);

static void run(const char *name, decfn fn, const ENC *E, const uint8_t *v,
                uint8_t *out) {
    /* ONLY=<substr> runs just matching kernels (for perf-counter isolation:
       measure ONLY=zzz as setup baseline, subtract) */
    const char *only = getenv("ONLY");
    if (only && !strstr(name, only)) return;
    size_t N = E->N;
    memset(out, 0xAA, N);
    fn(E, out);
    if (memcmp(out, v, N)) { printf("  %-9s VERIFY FAIL\n", name); return; }
    int R = 1 + (int)(6e8 / N); if (R > 3000) R = 3000;
    double best = 1e30;
    for (int i = 0; i < R; i++) {
        double t0 = now(); fn(E, out); double t1 = now();
        if (t1 - t0 < best) best = t1 - t0;
    }
#ifdef PROF
    g_it = 0; memset(g_hist, 0, sizeof g_hist); fn(E, out);
    { double H = 0; uint64_t tot = 0; int nz = 0;
      for (int i = 0; i < 16384; i++) tot += g_hist[i];
      for (int i = 0; i < 16384; i++) if (g_hist[i]) { nz++; double p = (double)g_hist[i] / tot;
                                                       H -= p * __builtin_log2(p); }
      printf("  %-9s %7.3f ns/B  %6.2f GB/s  %5.2f B/iter  wset %4d  perplex %6.0f\n",
             name, 1e9 * best / N, N / best / 1e9, g_it ? (double)N / g_it : 0.0, nz,
             tot ? __builtin_exp2(H) : 0.0); }
#else
    printf("  %-9s %7.3f ns/B  %6.2f GB/s\n", name, 1e9 * best / N, N / best / 1e9);
#endif
}

static uint64_t rng = 0x243F6A8885A308D3ULL;
static uint32_t xr(void) { rng ^= rng << 13; rng ^= rng >> 7; rng ^= rng << 17; return (uint32_t)rng; }

static void selftest(void) {
    size_t N = 1 << 20;
    uint8_t *v = malloc(N), *out = malloc(N + 64);
    for (size_t i = 0; i < N; i++) v[i] = (uint8_t)xr();
    ENC *e4 = encode(v, N, 4, 8192, 6);
    ENC *e2 = encode(v, N, 2, 8192, 7);
    uint8_t *chk[9]; int nc = 0;
    struct { decfn f; const ENC *e; const char *n; } T[] = {
        {dec_h4_u1, e4, "h4u1"}, {dec_h4_u4, e4, "h4u4"}, {dec_h4_u8, e4, "h4u8"},
#ifdef HAVE_K4R
        {dec_k4r_u8, e4, "k4r8"}, {dec_k4r_u4, e4, "k4r4"},
#endif
#ifdef HAVE_K4V
        {dec_k4v_u8, e4, "k4v"}, {dec_k4v_u16, e4, "k4v16"},
        {dec_k4w_u16, e4, "k4w16"}, {dec_k4v_u24, e4, "k4v24"},
        {dec_k4x_u16, e4, "k4x16"},
#endif
        {dec_w3_u1, e4, "w3u1"}, {dec_w3_u4, e4, "w3u4"}, {dec_w3_u8, e4, "w3u8"},
        {dec_k2nt_u8, e2, "k2nt"}, {dec_k2t_u8, e2, "k2t"},
        {dec_k2t_u2, e2, "k2t2"}, {dec_k2t_u4, e2, "k2t4"},
        {dec_k2t_u6, e2, "k2t6"}, {dec_k2t_u12, e2, "k2t12"},
        {dec_k2t_u16, e2, "k2t16"},
        {dec_k2nt_u2, e2, "k2nt2"}, {dec_k2nt_u4, e2, "k2nt4"},
        {dec_k2nt_u6, e2, "k2nt6"}, {dec_k2nt_u12, e2, "k2nt12"},
        {dec_k2nt_u16, e2, "k2nt16"},
#if !defined(__aarch64__)
        {dec_k2m_u2, e2, "k2m2"}, {dec_k2m_u4, e2, "k2m4"},
        {dec_k2m_u6, e2, "k2m6"}, {dec_k2m_u12, e2, "k2m12"},
        {dec_k2m_u16, e2, "k2m16"},
#endif
#ifdef HAVE_K2G
        {dec_k2g_u2, e2, "k2g2"}, {dec_k2g_u4, e2, "k2g4"},
        {dec_k2g_u6, e2, "k2g6"}, {dec_k2g_u8, e2, "k2g8"},
        {dec_k2gc_u4, e2, "k2gc4"},
#endif
        {dec_k2ntc_u4, e2, "k2ntc4"}, {dec_k2tc_u4, e2, "k2tc4"},
#if !defined(__aarch64__)
        {dec_k2mc_u4, e2, "k2mc4"},
#endif
#ifdef HAVE_K2Q
        {dec_k2q_u4, e2, "k2q4"}, {dec_k2q_u6, e2, "k2q6"},
        {dec_k2qc_u4, e2, "k2qc4"},
#endif
#ifdef HAVE_K2P
        {dec_k2p_u4, e2, "k2p4"}, {dec_k2pc_u4, e2, "k2pc4"},
#endif
        {dec_k2d_u8, e2, "k2d"}, {dec_k2e_u8, e2, "k2e"}, {dec_k2f_u8, e2, "k2f"},
        {dec_sc4a_u8, e4, "sc4a8"}, {dec_sc4a_u16, e4, "sc4a16"},
        {dec_sc4al_u16, e4, "sc4al16"}, {dec_sc4r_u8, e4, "sc4r8"},
        {dec_sc4r_u16, e4, "sc4r16"}, {dec_sc2r_u8, e2, "sc2r8"},
        {dec_sc2r_u16, e2, "sc2r16"}, {dec_sc2rl_u16, e2, "sc2rl16"},
    };
    (void)chk; (void)nc;
    for (unsigned i = 0; i < sizeof T / sizeof *T; i++) {
        memset(out, 0x55, N);
        T[i].f(T[i].e, out);
        if (memcmp(out, v, N)) { printf("SELFTEST FAIL: %s\n", T[i].n); exit(1); }
    }
    enc_free(e4); enc_free(e2);
    free(v); free(out);
    printf("selftest ok\n");
}

int main(int argc, char **argv) {
    build_h4(); build_w3(); build_k2(); build_k2g(); build_k2p();
    build_k2d(); build_k2e(); build_k2f();
    for (int i = 0; i < 256; i++) { g_cls4[i] = (uint8_t)(i >> 6); g_cls2[i] = (uint8_t)(i >> 7); }
    /* gather constant identity-order check */
    { uint64_t w = 0x8000000000000080ULL;   /* byte0 and byte7 have top bit */
      uint32_t g = (uint32_t)((((w >> 7) & BB) * MUL) >> 56);
      if (g != 0x81) { printf("gather order BUG %02x\n", g); return 1; } }
    selftest();
    const char *defs[] = {"dickens", "webster", "xml", "samba", "x-ray", "mozilla"};
    int nf = argc > 1 ? argc - 1 : 6;
    for (int fi = 0; fi < nf; fi++) {
        const char *name = argc > 1 ? argv[fi + 1] : defs[fi];
        char path[256], mp4[256], mp2[256];
        snprintf(path, sizeof path, "/tmp/phd_%s/lit", name);
        snprintf(mp4, sizeof mp4, "/tmp/o1maps/%s.map4", name);
        snprintf(mp2, sizeof mp2, "/tmp/o1maps/%s.map2", name);
        FILE *f = fopen(path, "rb");
        if (!f) { printf("missing %s\n", path); continue; }
        fseek(f, 0, SEEK_END); size_t N = ftell(f); fseek(f, 0, SEEK_SET);
        uint8_t *v0 = malloc(N); size_t rd = fread(v0, 1, N, f); fclose(f);
        if (rd != N) { printf("read fail %s\n", path); free(v0); continue; }
        /* remap: class = top bit(s) AFTER bijective remap (identity if no map) */
        uint8_t map4[256], map2[256], map8[256], map16[256];
        int have4 = 0, have2 = 0;
        char mp8[256], mp16[256];
        snprintf(mp8, sizeof mp8, "/tmp/o1maps/%s.map8", name);
        snprintf(mp16, sizeof mp16, "/tmp/o1maps/%s.map16", name);
        for (int i = 0; i < 256; i++) map4[i] = map2[i] = map8[i] = map16[i] = (uint8_t)i;
        FILE *m = fopen(mp4, "rb");
        if (m) { have4 = fread(map4, 1, 256, m) == 256; fclose(m); }
        m = fopen(mp2, "rb");
        if (m) { have2 = fread(map2, 1, 256, m) == 256; fclose(m); }
        m = fopen(mp8, "rb");  if (m) { rd = fread(map8, 1, 256, m); fclose(m); }
        m = fopen(mp16, "rb"); if (m) { rd = fread(map16, 1, 256, m); fclose(m); }
        uint8_t *v = malloc(N), *v2 = malloc(N), *v8 = malloc(N), *v16 = malloc(N);
        for (size_t i = 0; i < N; i++) { v[i] = map4[v0[i]]; v2[i] = map2[v0[i]];
                                         v8[i] = map8[v0[i]]; v16[i] = map16[v0[i]]; }
        free(v0);
        uint8_t *out = malloc(N + 64);
        printf("%s (%zu bytes)%s%s\n", path, N,
               have4 ? " [map4]" : " [RAW top2]", have2 ? " [map2]" : " [RAW top1]");
        size_t segb = 16128;                /* NOT a 4K multiple: stagger the
                                               per-chain page offsets */
        if (segb > N / 32 && N / 32 >= 2048) segb = (N / 32) | 1;
        else if (N / 32 < 2048) segb = 2049;
        ENC *e4 = encode(v, N, 4, segb, 6);
        ENC *e2 = encode(v2, N, 2, segb, 7);
        ENC *e8 = encode(v8, N, 8, segb, 5);
        ENC *e16 = encode(v16, N, 16, segb, 4);
        /* scalar reference timing (K=4 encode, 8 seg) */
        { memset(out, 0, N); dec_ref(e4, out, 6);
          if (memcmp(out, v, N)) { printf("  ref VERIFY FAIL\n"); }
          else { int R = 1 + (int)(2e8 / N); double best = 1e30;
                 for (int i = 0; i < R; i++) { double t0 = now(); dec_ref(e4, out, 6); double t1 = now();
                                               if (t1 - t0 < best) best = t1 - t0; }
                 printf("  %-9s %7.3f ns/B  %6.2f GB/s\n", "ref4", 1e9 * best / N, N / best / 1e9); } }
        run("h4_u1", dec_h4_u1, e4, v, out);
        run("h4_u4", dec_h4_u4, e4, v, out);
        run("h4_u8", dec_h4_u8, e4, v, out);
#ifdef HAVE_K4R
        run("k4r_u4", dec_k4r_u4, e4, v, out);
        run("k4r_u8", dec_k4r_u8, e4, v, out);
#endif
#ifdef HAVE_K4V
        run("k4v_u8", dec_k4v_u8, e4, v, out);
        run("k4v_u16", dec_k4v_u16, e4, v, out);
        run("k4w_u16", dec_k4w_u16, e4, v, out);
        run("k4v_u24", dec_k4v_u24, e4, v, out);
        run("k4x_u16", dec_k4x_u16, e4, v, out);
#endif
        run("w3_u1", dec_w3_u1, e4, v, out);
        run("w3_u4", dec_w3_u4, e4, v, out);
        run("w3_u8", dec_w3_u8, e4, v, out);
        run("k2nt_u2", dec_k2nt_u2, e2, v2, out);
        run("k2nt_u4", dec_k2nt_u4, e2, v2, out);
        run("k2nt_u6", dec_k2nt_u6, e2, v2, out);
        run("k2nt_u8", dec_k2nt_u8, e2, v2, out);
        run("k2nt_u12", dec_k2nt_u12, e2, v2, out);
        run("k2nt_u16", dec_k2nt_u16, e2, v2, out);
        run("k2t_u2", dec_k2t_u2, e2, v2, out);
        run("k2t_u4", dec_k2t_u4, e2, v2, out);
        run("k2t_u6", dec_k2t_u6, e2, v2, out);
        run("k2t_u8", dec_k2t_u8, e2, v2, out);
        run("k2t_u12", dec_k2t_u12, e2, v2, out);
        run("k2t_u16", dec_k2t_u16, e2, v2, out);
#ifdef HAVE_K2G
        run("k2g_u2", dec_k2g_u2, e2, v2, out);
        run("k2g_u4", dec_k2g_u4, e2, v2, out);
        run("k2g_u6", dec_k2g_u6, e2, v2, out);
        run("k2g_u8", dec_k2g_u8, e2, v2, out);
        run("k2gc_u4", dec_k2gc_u4, e2, v2, out);
#endif
        run("k2ntc_u4", dec_k2ntc_u4, e2, v2, out);
        run("k2tc_u4", dec_k2tc_u4, e2, v2, out);
#if !defined(__aarch64__)
        run("k2mc_u4", dec_k2mc_u4, e2, v2, out);
#endif
#ifdef HAVE_K2Q
        run("k2q_u4", dec_k2q_u4, e2, v2, out);
        run("k2q_u6", dec_k2q_u6, e2, v2, out);
        run("k2qc_u4", dec_k2qc_u4, e2, v2, out);
#endif
#ifdef HAVE_K2P
        run("k2p_u4", dec_k2p_u4, e2, v2, out);
        run("k2pc_u4", dec_k2pc_u4, e2, v2, out);
#endif
        run("k2d_u8", dec_k2d_u8, e2, v2, out);
        run("k2e_u8", dec_k2e_u8, e2, v2, out);
        run("k2f_u8", dec_k2f_u8, e2, v2, out);
#if !defined(__aarch64__)
        run("k2m_u2", dec_k2m_u2, e2, v2, out);
        run("k2m_u4", dec_k2m_u4, e2, v2, out);
        run("k2m_u6", dec_k2m_u6, e2, v2, out);
        run("k2m_u8", dec_k2m_u8, e2, v2, out);
        run("k2m_u12", dec_k2m_u12, e2, v2, out);
        run("k2m_u16", dec_k2m_u16, e2, v2, out);
#endif
        run("sc4a_u2", dec_sc4a_u2, e4, v, out);
        run("sc4a_u4", dec_sc4a_u4c, e4, v, out);
        run("sc4a_u6", dec_sc4a_u6, e4, v, out);
        run("sc4a_u8", dec_sc4a_u8, e4, v, out);
        run("sc4a_u12", dec_sc4a_u12, e4, v, out);
        run("sc4a_u24", dec_sc4a_u24, e4, v, out);
        run("sc8a_u8", dec_sc8a_u8, e8, v8, out);
        { uint8_t *route = malloc(N);
          for (int s = 0; s < e8->nseg; s++) { uint32_t c_ = 0;
            for (size_t j = e8->segoff[s]; j < e8->segoff[s + 1]; j++) {
                route[j] = (uint8_t)c_; c_ = v8[j] >> 5; } }
          g_route = route;
          run("scr8_u1", dec_scr8_u1, e8, v8, out);
          run("scr8_u8", dec_scr8_u8, e8, v8, out);
          free(route); }
        run("sc16a_u8", dec_sc16a_u8, e16, v16, out);
        run("sc4al_u8", dec_sc4al_u8, e4, v, out);
        run("sc4a_u16", dec_sc4a_u16, e4, v, out);
        run("sc4al_u16", dec_sc4al_u16, e4, v, out);
        run("sc4r_u8", dec_sc4r_u8, e4, v, out);
        run("sc4r_u16", dec_sc4r_u16, e4, v, out);
        run("sc2r_u8", dec_sc2r_u8, e2, v2, out);
        run("sc2rl_u8", dec_sc2rl_u8, e2, v2, out);
        run("sc2r_u16", dec_sc2r_u16, e2, v2, out);
        run("sc2rl_u16", dec_sc2rl_u16, e2, v2, out);
        enc_free(e4); enc_free(e2); enc_free(e8); enc_free(e16);
        free(v); free(v2); free(v8); free(v16); free(out);
    }
    return 0;
}
