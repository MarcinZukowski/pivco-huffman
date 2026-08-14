/* bench_select — bulk value-position queries on the PH wire format
 * (wavelet-tree "select all occurrences", done with the merge kernels).
 *
 * Query: given an encoded block and a symbol s, produce the 0xff/0x00
 * byte mask of s's positions WITHOUT decoding: walk s's leaf-to-root
 * spine; at each level skip the off-spine sibling region (header walk
 * only), read the node's bitmap, and expand the child mask through the
 * existing merge primitives with the off-spine side constant 0x00.
 * A leaf inside a flat subtree seeds the mask via prim_merge_flat with
 * a code->{0xff,0} table.  PH mode only (FSE markers assumed 0).
 *
 * Baseline: full decode (production entry) + autovectorized compare.
 *
 * Uses the canonical wire readers (pivco_huffman_wire.h) and the
 * arch-selected primitives router -- no wire or kernel replication.
 */
#include "pivco_huffman.h"
#include "bench_ctx.h"
#include "pivco_huffman_primitives.h"
#include "pivco_huffman_wire.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static double now_ns(void) {
    struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1e9 + ts.tv_nsec;
}

extern void bench_init(void);
extern int bench_num_distributions(void);
extern const char *bench_dist_name(int idx);
extern const uint64_t *bench_dist_freq(int idx);

#define BLK   PIVCO_BLOCK_SIZE   /* library default (16K Apple / 32K elsewhere) */

/* hand-tuned SIMD raw scan: byte-compare `n` bytes to `sym`, 0xff/0 mask */
#if defined(__aarch64__) || defined(__ARM_NEON)
#include <arm_neon.h>
static void simd_scan(const uint8_t *in, int n, uint8_t sym, uint8_t *mask) {
    uint8x16_t v = vdupq_n_u8(sym);
    int i = 0;
    for (; i + 64 <= n; i += 64) {
        vst1q_u8(mask+i,    vceqq_u8(vld1q_u8(in+i),    v));
        vst1q_u8(mask+i+16, vceqq_u8(vld1q_u8(in+i+16), v));
        vst1q_u8(mask+i+32, vceqq_u8(vld1q_u8(in+i+32), v));
        vst1q_u8(mask+i+48, vceqq_u8(vld1q_u8(in+i+48), v));
    }
    for (; i < n; i++) mask[i] = (in[i] == sym) ? 0xff : 0x00;
}
#elif defined(__AVX512BW__)
#include <immintrin.h>
static void simd_scan(const uint8_t *in, int n, uint8_t sym, uint8_t *mask) {
    __m512i v = _mm512_set1_epi8((char)sym);
    int i = 0;
    for (; i + 64 <= n; i += 64)
        _mm512_storeu_si512((void*)(mask+i),
            _mm512_movm_epi8(_mm512_cmpeq_epi8_mask(
                _mm512_loadu_si512((const void*)(in+i)), v)));
    for (; i < n; i++) mask[i] = (in[i] == sym) ? 0xff : 0x00;
}
#elif defined(__SSE2__)
#include <emmintrin.h>
static void simd_scan(const uint8_t *in, int n, uint8_t sym, uint8_t *mask) {
    __m128i v = _mm_set1_epi8((char)sym);
    int i = 0;
    for (; i + 16 <= n; i += 16)
        _mm_storeu_si128((__m128i*)(mask+i),
            _mm_cmpeq_epi8(_mm_loadu_si128((const __m128i*)(in+i)), v));
    for (; i < n; i++) mask[i] = (in[i] == sym) ? 0xff : 0x00;
}
#else
static void simd_scan(const uint8_t *in, int n, uint8_t sym, uint8_t *mask) {
    for (int i = 0; i < n; i++) mask[i] = (in[i] == sym) ? 0xff : 0x00;
}
#endif
#define TOTAL (4u << 20)

/* ---- wire region skipper: mirrors codec_decode_subtree's stream order ---- */
static void skip_marker_bitmap(const uint8_t **p, int K) {
    uint8_t marker = **p; (*p)++;
    if (marker == 0) { *p += bitmap_bytes(K); return; }
    uint16_t fse_len; memcpy(&fse_len, *p, 2); *p += 2 + fse_len;
}

static void skip_region(const pivco_table_t *t, int16_t id, int K,
                        const uint8_t **p) {
    const pivco_tree_node_t *n = &t->tree[id];
    if (K == 0) return;
    if (n->symbol >= 0) return;                       /* leaf: no bytes */
    if (t->flat_depth[id] >= 2) { *p += (K * t->flat_depth[id] + 7) >> 3; return; }
    switch (t->node_type[id]) {
    case PIVCO_NODE_BOTH_LEAVES:
        skip_marker_bitmap(p, K);
        return;
    case PIVCO_NODE_LEAF_LEFT: {
        int Kr = wire_read_kr_header(t, id, p);
        skip_region(t, n->right, Kr, p);
        skip_marker_bitmap(p, K);
        return;
    }
    default: {                                        /* FULL */
        int Kr = wire_read_kr_header(t, id, p);
        int Kl = K - Kr;
        if (Kr > Kl) { skip_region(t, n->right, Kr, p); skip_region(t, n->left, Kl, p); }
        else         { skip_region(t, n->left, Kl, p);  skip_region(t, n->right, Kr, p); }
        skip_marker_bitmap(p, K);
        return;
    }
    }
}

/* ---- select walk ---- */
typedef struct {
    const pivco_table_t *t;
    uint8_t on_spine[PIVCO_MAX_TREE_NODES];   /* node is an ancestor of target */
    int16_t target;                            /* leaf node, or flat root      */
    uint8_t flat_mask_c2s[256];                /* for flat target: code->mask  */
    uint8_t flat_sym;                          /* for D=8 flat: the symbol     */
    const uint8_t *zeros;                      /* BLK zero bytes               */
} selq_t;

/* returns mask of target occurrences among this node's K symbols in `out` */
static void select_rec(const selq_t *q, int16_t id, int K,
                       uint8_t *out, uint8_t *tmp, const uint8_t **p) {
    const pivco_table_t *t = q->t;
    const pivco_tree_node_t *n = &t->tree[id];
    if (K == 0) return;
    if (id == q->target) {
        if (t->flat_depth[id] >= 2) {          /* flat root: mask via c2s */
            int D = t->flat_depth[id];
            const uint8_t *bm = *p; *p += (K * D + 7) >> 3;
            if (D == 8) {
                /* D=8 packed codes ARE the symbols (the production
                 * kernel assumes identity c2s): direct compare */
                uint8_t sym = q->flat_sym;
                for (int i = 0; i < K; i++)
                    out[i] = (bm[i] == sym) ? 0xff : 0x00;
            } else {
                prim_merge_flat(out, K, bm, D, q->flat_mask_c2s);
            }
        } else {
            memset(out, 0xff, (size_t)K);      /* the leaf itself */
        }
        return;
    }
    uint8_t bm_scratch[(size_t)bitmap_bytes(BLK) + 16];
    switch (t->node_type[id]) {
    case PIVCO_NODE_BOTH_LEAVES: {
        const uint8_t *bm = wire_read_bitmap(p, K, bm_scratch);
        prim_merge_cst_cst(bm, K,
                           q->target == n->left  ? 0xff : 0x00,
                           q->target == n->right ? 0xff : 0x00, out);
        return;
    }
    case PIVCO_NODE_LEAF_LEFT: {
        int Kr = wire_read_kr_header(t, id, p);
        if (q->on_spine[n->right] || q->target == n->right) {
            select_rec(q, n->right, Kr, tmp, out, p);
            const uint8_t *bm = wire_read_bitmap(p, K, bm_scratch);
            prim_merge_cst_vec(bm, K, 0x00, tmp, out);
        } else {                               /* target is the left leaf */
            skip_region(t, n->right, Kr, p);
            const uint8_t *bm = wire_read_bitmap(p, K, bm_scratch);
            prim_merge_cst_vec(bm, K, 0xff, q->zeros, out);
        }
        return;
    }
    default: {                                 /* FULL */
        int Kr = wire_read_kr_header(t, id, p);
        int Kl = K - Kr;
        int right_spine = q->on_spine[n->right] || q->target == n->right;
        /* stream order: larger-K child first */
        if (Kr > Kl) {
            if (right_spine) select_rec(q, n->right, Kr, tmp, out, p);
            else             skip_region(t, n->right, Kr, p);
            if (right_spine) skip_region(t, n->left, Kl, p);
            else             select_rec(q, n->left, Kl, tmp, out, p);
        } else {
            if (right_spine) skip_region(t, n->left, Kl, p);
            else             select_rec(q, n->left, Kl, tmp, out, p);
            if (right_spine) select_rec(q, n->right, Kr, tmp, out, p);
            else             skip_region(t, n->right, Kr, p);
        }
        const uint8_t *bm = wire_read_bitmap(p, K, bm_scratch);
        if (right_spine) prim_merge_cst_vec(bm, K, 0x00, tmp, out);
        else             prim_merge_vec_vec(bm, K, tmp, q->zeros, out);
        return;
    }
    }
}

static int select_block(const selq_t *q, const uint8_t *enc,
                        uint8_t *mask, uint8_t *tmp) {
    const uint8_t *p = enc;
    uint16_t N; memcpy(&N, p, 2); p += 2;      /* block_N header */
    select_rec(q, q->t->tree_root, (int)N, mask, tmp, &p);
    return (int)N;
}


/* ---- multi-value (set) select: one walk over the UNION of spines ----
 * At nodes where several targets' subtrees meet, both children's masks
 * merge with vec_vec -- the shared ancestors (root above all) are paid
 * once for the whole set. */
typedef struct {
    const pivco_table_t *t;
    uint8_t contains[PIVCO_MAX_TREE_NODES];   /* subtree holds >=1 target */
    uint8_t leaf_in_set[PIVCO_MAX_TREE_NODES];
    uint8_t flat_c2s[256];                    /* multi-entry membership   */
    uint8_t in_set[256];                      /* symbol membership        */
    const uint8_t *zeros;
} mselq_t;

static uint8_t g_mbufL[16][65536], g_mbufR[16][65536];

static void mselect_rec(const mselq_t *q, int16_t id, int K, int depth,
                        uint8_t *out, const uint8_t **p) {
    const pivco_table_t *t = q->t;
    const pivco_tree_node_t *n = &t->tree[id];
    if (K == 0) return;
    if (n->symbol >= 0) { memset(out, q->leaf_in_set[id] ? 0xff : 0x00, (size_t)K); return; }
    if (t->flat_depth[id] >= 2) {
        int D = t->flat_depth[id];
        const uint8_t *bm = *p; *p += (K * D + 7) >> 3;
        if (D == 8) { for (int i = 0; i < K; i++) out[i] = q->in_set[bm[i]]; }
        else        prim_merge_flat(out, K, bm, D, q->flat_c2s);
        return;
    }
    uint8_t bm_scratch[(size_t)bitmap_bytes(BLK) + 16];
    switch (t->node_type[id]) {
    case PIVCO_NODE_BOTH_LEAVES: {
        const uint8_t *bm = wire_read_bitmap(p, K, bm_scratch);
        prim_merge_cst_cst(bm, K, q->leaf_in_set[n->left] ? 0xff : 0x00,
                           q->leaf_in_set[n->right] ? 0xff : 0x00, out);
        return;
    }
    case PIVCO_NODE_LEAF_LEFT: {
        int Kr = wire_read_kr_header(t, id, p);
        uint8_t cst = q->leaf_in_set[n->left] ? 0xff : 0x00;
        if (q->contains[n->right]) {
            mselect_rec(q, n->right, Kr, depth + 1, g_mbufR[depth], p);
            const uint8_t *bm = wire_read_bitmap(p, K, bm_scratch);
            prim_merge_cst_vec(bm, K, cst, g_mbufR[depth], out);
        } else {
            skip_region(t, n->right, Kr, p);
            const uint8_t *bm = wire_read_bitmap(p, K, bm_scratch);
            prim_merge_cst_vec(bm, K, cst, q->zeros, out);
        }
        return;
    }
    default: {                                 /* FULL */
        int Kr = wire_read_kr_header(t, id, p);
        int Kl = K - Kr;
        int cl = q->contains[n->left], cr = q->contains[n->right];
        const uint8_t *lm = q->zeros, *rm = q->zeros;
        /* stream order: larger-K child first */
        if (Kr > Kl) {
            if (cr) { mselect_rec(q, n->right, Kr, depth + 1, g_mbufR[depth], p); rm = g_mbufR[depth]; }
            else skip_region(t, n->right, Kr, p);
            if (cl) { mselect_rec(q, n->left, Kl, depth + 1, g_mbufL[depth], p); lm = g_mbufL[depth]; }
            else skip_region(t, n->left, Kl, p);
        } else {
            if (cl) { mselect_rec(q, n->left, Kl, depth + 1, g_mbufL[depth], p); lm = g_mbufL[depth]; }
            else skip_region(t, n->left, Kl, p);
            if (cr) { mselect_rec(q, n->right, Kr, depth + 1, g_mbufR[depth], p); rm = g_mbufR[depth]; }
            else skip_region(t, n->right, Kr, p);
        }
        const uint8_t *bm = wire_read_bitmap(p, K, bm_scratch);
        prim_merge_vec_vec(bm, K, lm, rm, out);
        return;
    }
    }
}

static void mbuild(mselq_t *q, const pivco_table_t *t,
                   const int *syms, int nsym) {
    memset(q, 0, sizeof(*q));
    q->t = t;
    for (int si = 0; si < nsym; si++) {
        int sym = syms[si];
        q->in_set[sym] = 0xff;
        uint16_t code = t->code[sym];
        int len = t->code_len[sym];
        int16_t cur = t->tree_root;
        for (int b = len - 1; ; b--) {
            q->contains[cur] = 1;
            if (t->flat_depth[cur] >= 2) {
                q->flat_c2s[code & ((1u << t->flat_depth[cur]) - 1)] = 0xff;
                break;
            }
            if (t->tree[cur].symbol >= 0) { q->leaf_in_set[cur] = 1; break; }
            if (b < 0) { q->leaf_in_set[cur] = 1; break; }
            cur = ((code >> b) & 1) ? t->tree[cur].right : t->tree[cur].left;
        }
    }
}

/* build spine[] by walking the code bits of sym from the root */
static void build_spine(selq_t *q, const pivco_table_t *t, int sym) {
    memset(q->on_spine, 0, sizeof(q->on_spine));
    memset(q->flat_mask_c2s, 0, sizeof(q->flat_mask_c2s));
    q->t = t;
    uint16_t code = t->code[sym];
    int len = t->code_len[sym];
    int16_t cur = t->tree_root;
    for (int b = len - 1; ; b--) {
        q->on_spine[cur] = 1;
        if (t->flat_depth[cur] >= 2) {         /* target inside flat subtree */
            int D = t->flat_depth[cur];
            q->target = cur;
            q->flat_mask_c2s[code & ((1u << D) - 1)] = 0xff;
            q->flat_sym = (uint8_t)(code & ((1u << D) - 1));
            return;
        }
        if (t->tree[cur].symbol >= 0) { q->target = cur; return; }
        if (b < 0) { q->target = cur; return; }
        cur = ((code >> b) & 1) ? t->tree[cur].right : t->tree[cur].left;
    }
}

static volatile uint8_t g_sink;
int main(void) {
    bench_init();
    prim_codec_init();                         /* merge shuffle tables */
    bench_cfg()->fse_enabled = (0);          /* PH mode: raw bitmaps */
    static uint8_t zeros[BLK];
    printf("bench_select: %u MiB, %d-sym blocks, best of 5  (GB/s of input covered)\n",
           TOTAL >> 20, BLK);
    printf("%-14s %-5s %6s | %8s %8s %8s | %8s %8s\n",
           "dist", "sym", "P%", "select", "dec+cmp", "speedup", "dec", "rawsimd");
    int nd = bench_num_distributions();
    for (int di = 0; di < nd; di++) {
        const uint64_t *freq = bench_dist_freq(di);
        static pivco_table_t t;
        if (pivco_build_table(bench_cfg(), freq, &t) != PIVCO_OK) continue;
        /* generate data */
        uint64_t tot = 0; for (int s = 0; s < 256; s++) tot += freq[s];
        uint8_t *data = malloc(TOTAL);
        uint64_t rng = 0x12345678 + (unsigned)di;
        {   uint64_t acc[257]; acc[0] = 0;
            for (int s = 0; s < 256; s++) acc[s+1] = acc[s] + freq[s];
            for (size_t i = 0; i < TOTAL; i++) {
                rng ^= rng << 13; rng ^= rng >> 7; rng ^= rng << 17;
                uint64_t r = rng % tot;
                int lo = 0, hi = 256;
                while (lo + 1 < hi) { int mid = (lo+hi)/2; if (acc[mid] <= r) lo = mid; else hi = mid; }
                data[i] = (uint8_t)lo;
            }
        }
        /* encode blocks */
        int nblk = TOTAL / BLK;
        uint8_t **enc = malloc(sizeof(void*) * nblk);
        size_t *elen = malloc(sizeof(size_t) * nblk);
        for (int b = 0; b < nblk; b++) {
            enc[b] = malloc(2 * BLK + 64); elen[b] = 2 * BLK + 64;
            if (pivco_encode(bench_enc_ctx(), &t, data + (size_t)b * BLK, BLK, enc[b], &elen[b])) {
                printf("%-14s encode failed\n", bench_dist_name(di)); nblk = 0; break;
            }
        }
        if (!nblk) continue;
        /* wire-walk validation: skip_region and select must consume exactly
         * what decode consumes */
        {   size_t cons; static uint8_t d0[BLK];
            pivco_decode(bench_dec_ctx(), &t, enc[0], elen[0], d0, &cons);
            const uint8_t *p = enc[0] + 2;
            skip_region(&t, t.tree_root, BLK, &p);
            if ((size_t)(p - enc[0]) != cons)
                fprintf(stderr, "DBG %s skip consumed %zu vs decode %zu\n",
                        bench_dist_name(di), (size_t)(p - enc[0]), cons);
        }
        /* pick frequent / median / rare present symbols */
        int syms[3] = {-1, -1, -1};
        {   int present[256], np = 0;
            for (int s = 0; s < 256; s++) if (freq[s]) present[np++] = s;
            for (int i = 1; i < np; i++) {     /* sort by freq desc */
                int v = present[i], j = i - 1;
                while (j >= 0 && freq[present[j]] < freq[v]) { present[j+1] = present[j]; j--; }
                present[j+1] = v;
            }
            syms[0] = present[0]; syms[1] = present[np/2]; syms[2] = present[np-1];
        }
        static uint8_t mask[BLK], tmp[BLK], dec[BLK], ref[BLK];
        for (int qi = 0; qi < 3; qi++) {
            int sym = syms[qi];
            if (qi && sym == syms[qi-1]) continue;
            selq_t q; q.zeros = zeros;
            build_spine(&q, &t, sym);
            double t_sel = 1e30, t_dc = 1e30, t_dec = 1e30, t_cmp = 1e30;
            int ok = 1;
            for (int rep = 0; rep < 5; rep++) {
                double t0 = now_ns();
                for (int b = 0; b < nblk; b++) select_block(&q, enc[b], mask, tmp);
                double e = now_ns() - t0; if (e < t_sel) t_sel = e;
                t0 = now_ns();
                for (int b = 0; b < nblk; b++) {
                    size_t cons;
                    pivco_decode(bench_dec_ctx(), &t, enc[b], elen[b], dec, &cons);
                    for (int i = 0; i < BLK; i++) ref[i] = (dec[i] == sym) ? 0xff : 0x00;
                    g_sink = ref[BLK - 1];
                }
                e = now_ns() - t0; if (e < t_dc) t_dc = e;
                t0 = now_ns();
                for (int b = 0; b < nblk; b++) { size_t cons;
                    pivco_decode(bench_dec_ctx(), &t, enc[b], elen[b], dec, &cons); }
                e = now_ns() - t0; if (e < t_dec) t_dec = e;
                t0 = now_ns();
                for (int b = 0; b < nblk; b++) {
                    simd_scan(dec, BLK, (uint8_t)sym, ref);
                    g_sink = ref[BLK - 1];
                }
                e = now_ns() - t0; if (e < t_cmp) t_cmp = e;
            }
            /* verify on the last block */
            {   size_t cons;
                select_block(&q, enc[nblk-1], mask, tmp);
                pivco_decode(bench_dec_ctx(), &t, enc[nblk-1], elen[nblk-1], dec, &cons);
                int non = 0, nref = 0, first = -1;
                for (int i = 0; i < BLK; i++) {
                    if (mask[i]) non++;
                    if (dec[i] == sym) nref++;
                    if (mask[i] != ((dec[i] == sym) ? 0xff : 0x00) && first < 0) first = i;
                }
                if (first >= 0) { ok = 0;
                    fprintf(stderr, "DBG sym=%d mask_on=%d ref_on=%d first_bad=%d mask=%02x dec=%02x\n",
                            sym, non, nref, first, mask[first], dec[first]); }
            }
            printf("%-14s %-5d %6.2f | %8.2f %8.2f %7.2fx | %8.2f %8.2f %s\n",
                   qi ? "" : bench_dist_name(di), sym,
                   100.0 * freq[sym] / tot,
                   TOTAL / t_sel, TOTAL / t_dc, t_dc / t_sel,
                   TOTAL / t_dec, TOTAL / t_cmp, ok ? "" : "MISMATCH");
        }
        for (int b = 0; b < nblk; b++) free(enc[b]);
        free(enc); free(elen); free(data);
    }

    /* ---- 3-value OR on prose: amortization vs 3 singles vs raw ---- */
    for (int di = 0; di < nd; di++) {
        if (strcmp(bench_dist_name(di), "prose_pride")) continue;
        const uint64_t *freq = bench_dist_freq(di);
        static pivco_table_t t;
        if (pivco_build_table(bench_cfg(), freq, &t) != PIVCO_OK) break;
        uint64_t tot = 0; for (int s2 = 0; s2 < 256; s2++) tot += freq[s2];
        uint8_t *data = malloc(TOTAL);
        uint64_t rng = 0xfeedbeef;
        {   uint64_t acc[257]; acc[0] = 0;
            for (int s2 = 0; s2 < 256; s2++) acc[s2+1] = acc[s2] + freq[s2];
            for (size_t i = 0; i < TOTAL; i++) {
                rng ^= rng << 13; rng ^= rng >> 7; rng ^= rng << 17;
                uint64_t r = rng % tot;
                int lo = 0, hi = 256;
                while (lo + 1 < hi) { int mid = (lo+hi)/2; if (acc[mid] <= r) lo = mid; else hi = mid; }
                data[i] = (uint8_t)lo;
            }
        }
        int nblk = TOTAL / BLK;
        uint8_t **enc = malloc(sizeof(void*) * nblk);
        size_t *elen = malloc(sizeof(size_t) * nblk);
        for (int b = 0; b < nblk; b++) {
            enc[b] = malloc(2 * BLK + 64); elen[b] = 2 * BLK + 64;
            pivco_encode(bench_enc_ctx(), &t, data + (size_t)b * BLK, BLK, enc[b], &elen[b]);
        }
        int syms3[3] = {'m', 'y', 'k'};        /* semi-random mid-freq letters */
        double psum = 0;
        for (int i2 = 0; i2 < 3; i2++) psum += 100.0 * freq[syms3[i2]] / tot;
        mselq_t mq; mbuild(&mq, &t, syms3, 3);
        selq_t q1[3];
        for (int i2 = 0; i2 < 3; i2++) { q1[i2].zeros = zeros; build_spine(&q1[i2], &t, syms3[i2]); }
        mq.zeros = zeros;
        static uint8_t mask[BLK], m2[BLK], tmp[BLK], dec[BLK], ref[BLK];
        double t_multi = 1e30, t_3single = 1e30, t_raw = 1e30, t_dc = 1e30;
        for (int rep = 0; rep < 5; rep++) {
            double t0 = now_ns();
            for (int b = 0; b < nblk; b++) {
                const uint8_t *p = enc[b]; uint16_t N; memcpy(&N, p, 2); p += 2;
                mselect_rec(&mq, t.tree_root, (int)N, 0, mask, &p);
            }
            double e = now_ns() - t0; if (e < t_multi) t_multi = e;
            t0 = now_ns();
            for (int b = 0; b < nblk; b++) {
                select_block(&q1[0], enc[b], mask, tmp);
                select_block(&q1[1], enc[b], m2, tmp);
                for (int i2 = 0; i2 < BLK; i2++) mask[i2] |= m2[i2];
                select_block(&q1[2], enc[b], m2, tmp);
                for (int i2 = 0; i2 < BLK; i2++) mask[i2] |= m2[i2];
            }
            e = now_ns() - t0; if (e < t_3single) t_3single = e;
            t0 = now_ns();
            for (int b = 0; b < nblk; b++) {
                simd_scan(dec, BLK, (uint8_t)syms3[0], ref);
                simd_scan(dec, BLK, (uint8_t)syms3[1], m2);
                for (int i2 = 0; i2 < BLK; i2++) ref[i2] |= m2[i2];
                simd_scan(dec, BLK, (uint8_t)syms3[2], m2);
                for (int i2 = 0; i2 < BLK; i2++) ref[i2] |= m2[i2];
                g_sink = ref[BLK - 1];
            }
            e = now_ns() - t0; if (e < t_raw) t_raw = e;
            t0 = now_ns();
            for (int b = 0; b < nblk; b++) { size_t cons;
                pivco_decode(bench_dec_ctx(), &t, enc[b], elen[b], dec, &cons);
                simd_scan(dec, BLK, (uint8_t)syms3[0], ref);
                simd_scan(dec, BLK, (uint8_t)syms3[1], m2);
                for (int i2 = 0; i2 < BLK; i2++) ref[i2] |= m2[i2];
                simd_scan(dec, BLK, (uint8_t)syms3[2], m2);
                for (int i2 = 0; i2 < BLK; i2++) ref[i2] |= m2[i2];
                g_sink = ref[BLK - 1];
            }
            e = now_ns() - t0; if (e < t_dc) t_dc = e;
        }
        int ok = 1;
        {   size_t cons;
            const uint8_t *p = enc[nblk-1]; uint16_t N; memcpy(&N, p, 2); p += 2;
            mselect_rec(&mq, t.tree_root, (int)N, 0, mask, &p);
            pivco_decode(bench_dec_ctx(), &t, enc[nblk-1], elen[nblk-1], dec, &cons);
            for (int i2 = 0; i2 < BLK; i2++) {
                uint8_t want = (dec[i2] == syms3[0] || dec[i2] == syms3[1]
                             || dec[i2] == syms3[2]) ? 0xff : 0x00;
                if (mask[i2] != want) { ok = 0; break; }
            }
        }
        printf("\n3-value OR on prose_pride ('m','y','k', P=%.1f%%):  (GB/s)\n", psum);
        printf("  multi-select (1 walk)  %8.2f %s\n", TOTAL / t_multi, ok ? "" : "MISMATCH");
        printf("  3x single select + OR  %8.2f   (amortization x%.2f)\n",
               TOTAL / t_3single, t_3single / t_multi);
        printf("  raw SIMD 3-scan + OR   %8.2f   (multi vs raw x%.2f)\n",
               TOTAL / t_raw, t_raw / t_multi);
        printf("  decode + 3-scan + OR   %8.2f   (multi vs dec x%.2f)\n",
               TOTAL / t_dc, t_dc / t_multi);
        for (int b = 0; b < nblk; b++) free(enc[b]);
        free(enc); free(elen); free(data);
        break;
    }
    return 0;
}
