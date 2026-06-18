/* bench/prim_variants/prims-merge.h — merge-family variant graveyard.
 *
 * Logical primitives: merge_vec_vec (ST_MERGE_VEC_VEC), merge_cst_vec,
 * merge_vec_cst, merge_cst_cst, merge_flat.  See prims.h for the contract +
 * naming (PV_ = constants/macros, pv_ = plumbing, prim_ = kernels).
 *
 * Uses the production neon merge tables (expand_tab / expand_tab_pre /
 * expand_popcnt — built by prim_codec_init(), in scope here), same as the
 * partition graveyard uses compress_tab.  MERGE_LEFT_SYM / MERGE_RIGHT_SYM
 * (the cst_cst test symbols) also come from bench_prim.c.
 *
 * NOTE: the 128/64/16-wide merge_vec_vec kernels process whole blocks only;
 * they verify at n a multiple of their stride (the bench default n=8192 is).
 */
#ifndef PIVCO_PRIM_VARIANTS_MERGE_H
#define PIVCO_PRIM_VARIANTS_MERGE_H

#if defined(USE_NEON_KERNELS)

/* ============================================================================
 * merge_vec_vec : asof-4dd08e3 — the genesis bottom-up tree_merge (2026-05-10)
 *   First-ever BU NEON merge (4dd08e3, "beats top-down on 22/29 dists").  Pure
 *   stride-8: one vqtbl1 over vcombine(L8,R8) per 8 outputs, serial cursor.
 *   Superseded by cursor16 (2c8606e), then COM64.
 * ========================================================================== */
static inline void prim_merge_vv_asof_4dd08e3_neon(const uint8_t *bm, int K,
                                                   const uint8_t *left, const uint8_t *right, uint8_t *out) {
    int lc = 0, rc = 0, j = 0;
    for (; j + 8 <= K; j += 8) {
        uint8_t m = bm[j >> 3];
        uint8x16_t both = vcombine_u8(vld1_u8(left + lc), vld1_u8(right + rc));
        vst1_u8(out + j, vqtbl1_u8(both, vld1_u8(expand_tab[m])));
        int nr = expand_popcnt[m];
        rc += nr; lc += (8 - nr);
    }
    for (; j < K; j++) { int mb = (bm[j >> 3] >> (j & 7)) & 1; out[j] = mb ? right[rc++] : left[lc++]; }
}
static void prim_merge_vv_asof_4dd08e3(const ctx_t *c){
    prim_merge_vv_asof_4dd08e3_neon(c->bm, c->n, c->merge_left, c->merge_right, c->out);
}

/* ============================================================================
 * merge_vec_vec : cursor16 — pre-COM64 shipped stride-16 merge (= asof-2c8606e)
 *   The production merge_vec_vec before the COM64 rework (5cccccc).  Cursor
 *   advanced by the byte popcounts — the loop-carried dependency COM removed.
 * ========================================================================== */
static inline void prim_merge_vv_cursor16_neon(const uint8_t *bm, int K,
                                               const uint8_t *left, const uint8_t *right, uint8_t *out) {
    int lc = 0, rc = 0, j = 0;
    for (; j + 16 <= K; j += 16) {
        uint8x16_t L_full = vld1q_u8(left + lc), R_full = vld1q_u8(right + rc);
        uint8_t m0 = bm[j >> 3];
        uint8x16_t both0 = vcombine_u8(vget_low_u8(L_full), vget_low_u8(R_full));
        vst1_u8(out + j, vqtbl1_u8(both0, vld1_u8(expand_tab[m0])));
        int nr0 = expand_popcnt[m0];
        uint8_t m1 = bm[(j >> 3) + 1];
        uint8x16x2_t src = {{ L_full, R_full }};
        vst1_u8(out + j + 8, vqtbl2_u8(src, vld1_u8(expand_tab_pre[nr0][m1])));
        int nr1 = expand_popcnt[m1];
        rc += nr0 + nr1; lc += (16 - nr0 - nr1);
    }
}
static void prim_merge_vv_cursor16(const ctx_t *c){
    prim_merge_vv_cursor16_neon(c->bm, c->n, c->merge_left, c->merge_right, c->out);
}

/* ============================================================================
 * merge_vec_vec : prefix128 / prefix64 — COM with vpaddl(q) popcount fold +
 *   u16 SWAR prefix sum + half-level cursors (Jeff Plaisance, PR #4).
 *   prefix128 is the 128/iter form (lo/hi split); prefix64 the 64/iter form.
 *   Measured on M1 Max; need a test-c8g run before any promotion.
 * ========================================================================== */
static inline void prim_merge_vv_prefix128_neon(const uint8_t *bm, int K,
                                                const uint8_t *left, const uint8_t *right, uint8_t *out) {
    int lc = 0, rc = 0, j = 0;
    for (; j + 128 <= K; j += 128) {
        uint8x16_t bmv = vld1q_u8(bm + (j >> 3));
        uint8x16_t pcv = vcntq_u8(bmv);
        uint64x2_t bm64 = vreinterpretq_u64_u8(bmv);
        uint64_t bm_lo = vgetq_lane_u64(bm64, 0), bm_hi = vgetq_lane_u64(bm64, 1);
        uint64x2_t pc64 = vreinterpretq_u64_u8(pcv);
        uint64_t pc_lo = vgetq_lane_u64(pc64, 0), pc_hi = vgetq_lane_u64(pc64, 1);
        uint16x8_t pair = vpaddlq_u8(pcv);
        uint64x2_t pr64 = vreinterpretq_u64_u16(pair);
        uint64_t pref_lo = vgetq_lane_u64(pr64, 0) * 0x0001000100010001ULL;
        uint64_t pref_hi = vgetq_lane_u64(pr64, 1) * 0x0001000100010001ULL;
        #define PV_PFX_BLOCK(BMW, PCW, K_, EXCL, EBASE)                         \
        do {                                                                    \
            uint32_t excl_ = (EXCL);                                            \
            uint8_t  m0  = (uint8_t)((BMW) >> (16 * (K_)));                      \
            uint8_t  m1  = (uint8_t)((BMW) >> (16 * (K_) + 8));                  \
            uint8_t  nr0 = (uint8_t)((PCW) >> (16 * (K_)));                      \
            uint8x16_t L = vld1q_u8(left  + lc + (16 * (K_) - excl_));           \
            uint8x16_t R = vld1q_u8(right + rc + excl_);                         \
            uint8x16_t both0 = vcombine_u8(vget_low_u8(L), vget_low_u8(R));      \
            vst1_u8(out + j + (EBASE) + 16 * (K_), vqtbl1_u8(both0, vld1_u8(expand_tab[m0]))); \
            uint8x16x2_t src = {{ L, R }};                                      \
            vst1_u8(out + j + (EBASE) + 16 * (K_) + 8, vqtbl2_u8(src, vld1_u8(expand_tab_pre[nr0][m1]))); \
        } while (0)
        PV_PFX_BLOCK(bm_lo, pc_lo, 0, 0,                        0);
        PV_PFX_BLOCK(bm_lo, pc_lo, 1, (pref_lo)       & 0xFFFF, 0);
        PV_PFX_BLOCK(bm_lo, pc_lo, 2, (pref_lo >> 16) & 0xFFFF, 0);
        PV_PFX_BLOCK(bm_lo, pc_lo, 3, (pref_lo >> 32) & 0xFFFF, 0);
        uint32_t r_lo = (uint32_t)(pref_lo >> 48); rc += r_lo; lc += 64 - r_lo;
        PV_PFX_BLOCK(bm_hi, pc_hi, 0, 0,                        64);
        PV_PFX_BLOCK(bm_hi, pc_hi, 1, (pref_hi)       & 0xFFFF, 64);
        PV_PFX_BLOCK(bm_hi, pc_hi, 2, (pref_hi >> 16) & 0xFFFF, 64);
        PV_PFX_BLOCK(bm_hi, pc_hi, 3, (pref_hi >> 32) & 0xFFFF, 64);
        uint32_t r_hi = (uint32_t)(pref_hi >> 48); rc += r_hi; lc += 64 - r_hi;
        #undef PV_PFX_BLOCK
    }
}
static void prim_merge_vv_prefix128(const ctx_t *c){
    prim_merge_vv_prefix128_neon(c->bm, c->n, c->merge_left, c->merge_right, c->out);
}

static inline void prim_merge_vv_prefix64_neon(const uint8_t *bm, int K,
                                               const uint8_t *left, const uint8_t *right, uint8_t *out) {
    int lc = 0, rc = 0, j = 0;
    for (; j + 64 <= K; j += 64) {
        uint64_t bm_word; memcpy(&bm_word, bm + (j >> 3), 8);
        uint8x8_t bm_v = vcreate_u8(bm_word);
        uint8x8_t pc_v = vcnt_u8(bm_v);
        uint64_t pc_word = vget_lane_u64(vreinterpret_u64_u8(pc_v), 0);
        uint16x4_t pair = vpaddl_u8(pc_v);
        uint64_t pref = vget_lane_u64(vreinterpret_u64_u16(pair), 0) * 0x0001000100010001ULL;
        #define PV_PFX_BLOCK64(K_, EXCL)                                        \
        do {                                                                    \
            uint32_t excl_ = (EXCL);                                            \
            uint8_t  m0  = (uint8_t)(bm_word >> (16 * (K_)));                    \
            uint8_t  m1  = (uint8_t)(bm_word >> (16 * (K_) + 8));                \
            uint8_t  nr0 = (uint8_t)(pc_word >> (16 * (K_)));                    \
            uint8x16_t L = vld1q_u8(left  + lc + (16 * (K_) - excl_));           \
            uint8x16_t R = vld1q_u8(right + rc + excl_);                         \
            uint8x16_t both0 = vcombine_u8(vget_low_u8(L), vget_low_u8(R));      \
            vst1_u8(out + j + 16 * (K_), vqtbl1_u8(both0, vld1_u8(expand_tab[m0]))); \
            uint8x16x2_t src = {{ L, R }};                                      \
            vst1_u8(out + j + 16 * (K_) + 8, vqtbl2_u8(src, vld1_u8(expand_tab_pre[nr0][m1]))); \
        } while (0)
        PV_PFX_BLOCK64(0, 0);
        PV_PFX_BLOCK64(1, (pref)       & 0xFFFF);
        PV_PFX_BLOCK64(2, (pref >> 16) & 0xFFFF);
        PV_PFX_BLOCK64(3, (pref >> 32) & 0xFFFF);
        uint32_t r = (uint32_t)(pref >> 48); rc += r; lc += 64 - r;
        #undef PV_PFX_BLOCK64
    }
}
static void prim_merge_vv_prefix64(const ctx_t *c){
    prim_merge_vv_prefix64_neon(c->bm, c->n, c->merge_left, c->merge_right, c->out);
}

/* ============================================================================
 * merge_vec_vec : pcpc / pcpc_full / unroll8 — precomputed/in-register popcount
 *   Hypothesis: the per-iter expand_popcnt[] load is on the cursor critical
 *   path; replace it with a sequential pv_bm_popcnt[] read (pcpc) or in-lane
 *   vcnt values (unroll8).  pcpc_full pays the table-fill inside the timed
 *   region; pcpc assumes it's pre-filled (bench fills it once, see bench_prim).
 *   M4: wash vs COM64, store-bound.  Originated in bench_prim.
 * ========================================================================== */
static uint8_t pv_bm_popcnt[16384] __attribute__((aligned(64)));
static inline void pv_fill_bm_popcnt(const uint8_t *bm, int K) {
    int bm_bytes = (K + 7) >> 3, i = 0;
    for (; i + 16 <= bm_bytes; i += 16) vst1q_u8(pv_bm_popcnt + i, vcntq_u8(vld1q_u8(bm + i)));
    for (; i < bm_bytes; i++) pv_bm_popcnt[i] = (uint8_t)__builtin_popcount(bm[i]);
}
static inline void prim_merge_vv_pcpc_neon(const uint8_t *bm, int K,
                                           const uint8_t *left, const uint8_t *right, uint8_t *out) {
    int lc = 0, rc = 0, j = 0;
    for (; j + 16 <= K; j += 16) {
        uint8x16_t L_full = vld1q_u8(left + lc), R_full = vld1q_u8(right + rc);
        uint8_t m0 = bm[j >> 3]; int nr0 = pv_bm_popcnt[j >> 3];
        uint8x16_t both0 = vcombine_u8(vget_low_u8(L_full), vget_low_u8(R_full));
        vst1_u8(out + j, vqtbl1_u8(both0, vld1_u8(expand_tab[m0])));
        uint8_t m1 = bm[(j >> 3) + 1]; int nr1 = pv_bm_popcnt[(j >> 3) + 1];
        uint8x16x2_t src = {{ L_full, R_full }};
        vst1_u8(out + j + 8, vqtbl2_u8(src, vld1_u8(expand_tab_pre[nr0][m1])));
        rc += nr0 + nr1; lc += (16 - nr0 - nr1);
    }
    for (; j < K; j++) { int mb = (bm[j >> 3] >> (j & 7)) & 1; out[j] = mb ? right[rc++] : left[lc++]; }
}
static void prim_merge_vv_pcpc(const ctx_t *c){     /* bm_popcnt pre-filled by bench */
    prim_merge_vv_pcpc_neon(c->bm, c->n, c->merge_left, c->merge_right, c->out);
}
static void prim_merge_vv_pcpc_full(const ctx_t *c){
    pv_fill_bm_popcnt(c->bm, c->n);
    prim_merge_vv_pcpc_neon(c->bm, c->n, c->merge_left, c->merge_right, c->out);
}
static inline void prim_merge_vv_unroll8_neon(const uint8_t *bm, int K,
                                              const uint8_t *left, const uint8_t *right, uint8_t *out) {
    int lc = 0, rc = 0, j = 0;
    for (; j + 64 <= K; j += 64) {
        uint8x8_t bm_vec = vld1_u8(bm + (j >> 3)), pc_vec = vcnt_u8(bm_vec);
        #define PV_UNROLL_STEP(K_) do {                              \
            uint8_t  m  = vget_lane_u8(bm_vec, K_);                  \
            int      nr = vget_lane_u8(pc_vec, K_);                  \
            uint8x8_t  L = vld1_u8(left + lc), R = vld1_u8(right + rc); \
            uint8x16_t both = vcombine_u8(L, R);                     \
            vst1_u8(out + j + 8 * K_, vqtbl1_u8(both, vld1_u8(expand_tab[m]))); \
            rc += nr; lc += (8 - nr);                                \
        } while (0)
        PV_UNROLL_STEP(0); PV_UNROLL_STEP(1); PV_UNROLL_STEP(2); PV_UNROLL_STEP(3);
        PV_UNROLL_STEP(4); PV_UNROLL_STEP(5); PV_UNROLL_STEP(6); PV_UNROLL_STEP(7);
        #undef PV_UNROLL_STEP
    }
    for (; j < K; j++) { int mb = (bm[j >> 3] >> (j & 7)) & 1; out[j] = mb ? right[rc++] : left[lc++]; }
}
static void prim_merge_vv_unroll8(const ctx_t *c){
    prim_merge_vv_unroll8_neon(c->bm, c->n, c->merge_left, c->merge_right, c->out);
}

/* ============================================================================
 * merge_vec_vec : com32 / com128 — COM prefix-sum cursor-decouple at 32- and
 *   128-code stride (the COM family from bench_merge_neon.c).  com64 is the
 *   shipped production merge_vec_vec; com32 (2 chunks/iter) and com128 (8
 *   chunks/iter, lo/hi u64 split + cross-half bias) bracket it on stride.
 *   Each chunk is vqtbl1 (iter0) + vqtbl2 over expand_tab_pre (iter1), with
 *   per-chunk start cursors precomputed from a *0x0101.. byte prefix sum so
 *   the chunks are independent (no loop-carried cursor).  Finding: com64
 *   dominates the cross-uarch matrix; com128 stresses NEON regalloc + adds a
 *   cross-half recombine; com32 leaves ILP on the table.
 * ========================================================================== */
static inline void prim_merge_vv_com32_neon(const uint8_t *bm, int K,
                                            const uint8_t *left, const uint8_t *right, uint8_t *out) {
    int lc = 0, rc = 0, j = 0;
    for (; j + 32 <= K; j += 32) {
        uint32_t mask_u32; memcpy(&mask_u32, bm + (j >> 3), 4);
        uint8x8_t bm_v = vcreate_u8((uint64_t)mask_u32);
        uint8x8_t pc_v = vcnt_u8(bm_v);
        uint64_t pc_u64 = vget_lane_u64(vreinterpret_u64_u8(pc_v), 0);
        uint64_t pfx = pc_u64 * 0x0101010101010101ULL;
        uint8_t cnt_in_chunk0 = (uint8_t)(pc_u64);
        uint8_t cnt_chunk1_r  = (uint8_t)(pfx >> 8);
        uint8_t cnt_in_chunk1 = (uint8_t)(pc_u64 >> 16);
        uint8_t m0 = (uint8_t)mask_u32,       m1 = (uint8_t)(mask_u32 >> 8);
        uint8_t m2 = (uint8_t)(mask_u32 >> 16), m3 = (uint8_t)(mask_u32 >> 24);
        uint8x16_t L0 = vld1q_u8(left + lc), R0 = vld1q_u8(right + rc);
        uint8x16_t both00 = vcombine_u8(vget_low_u8(L0), vget_low_u8(R0));
        vst1_u8(out + j,     vqtbl1_u8(both00, vld1_u8(expand_tab[m0])));
        uint8x16x2_t src0 = {{ L0, R0 }};
        vst1_u8(out + j + 8, vqtbl2_u8(src0, vld1_u8(expand_tab_pre[cnt_in_chunk0][m1])));
        uint8_t cl1 = (uint8_t)(16 - cnt_chunk1_r);
        uint8x16_t L1 = vld1q_u8(left + lc + cl1), R1 = vld1q_u8(right + rc + cnt_chunk1_r);
        uint8x16_t both10 = vcombine_u8(vget_low_u8(L1), vget_low_u8(R1));
        vst1_u8(out + j + 16, vqtbl1_u8(both10, vld1_u8(expand_tab[m2])));
        uint8x16x2_t src1 = {{ L1, R1 }};
        vst1_u8(out + j + 24, vqtbl2_u8(src1, vld1_u8(expand_tab_pre[cnt_in_chunk1][m3])));
        uint8_t total_r = (uint8_t)(pfx >> 24);
        rc += total_r; lc += 32 - total_r;
    }
}
static void prim_merge_vv_com32(const ctx_t *c){
    prim_merge_vv_com32_neon(c->bm, c->n, c->merge_left, c->merge_right, c->out);
}

static inline void prim_merge_vv_com128_neon(const uint8_t *bm, int K,
                                             const uint8_t *left, const uint8_t *right, uint8_t *out) {
    int lc = 0, rc = 0, j = 0;
    for (; j + 128 <= K; j += 128) {
        uint8x16_t bm_v = vld1q_u8(bm + (j >> 3));
        uint8x16_t pc_v = vcntq_u8(bm_v);
        uint64_t mask_lo = vgetq_lane_u64(vreinterpretq_u64_u8(bm_v), 0);
        uint64_t mask_hi = vgetq_lane_u64(vreinterpretq_u64_u8(bm_v), 1);
        uint64_t pc_lo   = vgetq_lane_u64(vreinterpretq_u64_u8(pc_v), 0);
        uint64_t pc_hi   = vgetq_lane_u64(vreinterpretq_u64_u8(pc_v), 1);
        uint64_t pfx_lo = pc_lo * 0x0101010101010101ULL;
        uint64_t pfx_hi = pc_hi * 0x0101010101010101ULL;
        uint64_t lo_total = (pfx_lo >> 56) * 0x0101010101010101ULL;
        pfx_hi += lo_total;
        uint8_t cr0 = 0,                       cr1 = (uint8_t)(pfx_lo >>  8);
        uint8_t cr2 = (uint8_t)(pfx_lo >> 24), cr3 = (uint8_t)(pfx_lo >> 40);
        uint8_t cr4 = (uint8_t)(pfx_lo >> 56), cr5 = (uint8_t)(pfx_hi >>  8);
        uint8_t cr6 = (uint8_t)(pfx_hi >> 24), cr7 = (uint8_t)(pfx_hi >> 40);
        uint8_t in0 = (uint8_t)pc_lo,         in1 = (uint8_t)(pc_lo >> 16);
        uint8_t in2 = (uint8_t)(pc_lo >> 32), in3 = (uint8_t)(pc_lo >> 48);
        uint8_t in4 = (uint8_t)pc_hi,         in5 = (uint8_t)(pc_hi >> 16);
        uint8_t in6 = (uint8_t)(pc_hi >> 32), in7 = (uint8_t)(pc_hi >> 48);
        uint8_t m0  = (uint8_t)mask_lo,        m1  = (uint8_t)(mask_lo >>  8);
        uint8_t m2  = (uint8_t)(mask_lo >> 16), m3  = (uint8_t)(mask_lo >> 24);
        uint8_t m4  = (uint8_t)(mask_lo >> 32), m5  = (uint8_t)(mask_lo >> 40);
        uint8_t m6  = (uint8_t)(mask_lo >> 48), m7  = (uint8_t)(mask_lo >> 56);
        uint8_t m8  = (uint8_t)mask_hi,        m9  = (uint8_t)(mask_hi >>  8);
        uint8_t m10 = (uint8_t)(mask_hi >> 16), m11 = (uint8_t)(mask_hi >> 24);
        uint8_t m12 = (uint8_t)(mask_hi >> 32), m13 = (uint8_t)(mask_hi >> 40);
        uint8_t m14 = (uint8_t)(mask_hi >> 48), m15 = (uint8_t)(mask_hi >> 56);
        #define PV_COM128_CHUNK(idx, cr, in, ma, mb) do {                       \
            uint8_t cl = (uint8_t)((idx)*16 - (cr));                            \
            uint8x16_t L = vld1q_u8(left + lc + cl);                            \
            uint8x16_t R = vld1q_u8(right + rc + (cr));                         \
            uint8x16_t both = vcombine_u8(vget_low_u8(L), vget_low_u8(R));      \
            vst1_u8(out + j + (idx)*16,     vqtbl1_u8(both, vld1_u8(expand_tab[ma]))); \
            uint8x16x2_t src = {{ L, R }};                                      \
            vst1_u8(out + j + (idx)*16 + 8, vqtbl2_u8(src, vld1_u8(expand_tab_pre[in][mb]))); \
        } while (0)
        PV_COM128_CHUNK(0, cr0, in0, m0,  m1);
        PV_COM128_CHUNK(1, cr1, in1, m2,  m3);
        PV_COM128_CHUNK(2, cr2, in2, m4,  m5);
        PV_COM128_CHUNK(3, cr3, in3, m6,  m7);
        PV_COM128_CHUNK(4, cr4, in4, m8,  m9);
        PV_COM128_CHUNK(5, cr5, in5, m10, m11);
        PV_COM128_CHUNK(6, cr6, in6, m12, m13);
        PV_COM128_CHUNK(7, cr7, in7, m14, m15);
        #undef PV_COM128_CHUNK
        uint8_t total_r = (uint8_t)(pfx_hi >> 56);
        rc += total_r; lc += 128 - total_r;
    }
}
static void prim_merge_vv_com128(const ctx_t *c){
    prim_merge_vv_com128_neon(c->bm, c->n, c->merge_left, c->merge_right, c->out);
}

/* ============================================================================
 * merge_cst_cst : tbl / blendtab / vtblq / vtbl / d1flat
 *   Alternate two-constant-symbol merges tried against the production cst_cst.
 *   tbl/blendtab use precomputed 256x8 LUTs (mask->bits / mask->blended out);
 *   vtblq/vtbl use vtst+vqtbl1 at 16- and 8-lane width; d1flat reuses the D=1
 *   flat-decode shape.  Originated in bench_prim.
 * ========================================================================== */
static uint8_t pv_mask_to_bits[256][8];
static uint8_t pv_blend_tab[256][8] __attribute__((aligned(64)));
static int     pv_cc_tables_built = 0;
static void pv_build_cc_tables(void) {
    if (pv_cc_tables_built) return;
    for (int m = 0; m < 256; m++)
        for (int i = 0; i < 8; i++) {
            pv_mask_to_bits[m][i] = ((m >> i) & 1) ? 0xFF : 0x00;
            pv_blend_tab[m][i]    = ((m >> i) & 1) ? MERGE_RIGHT_SYM : MERGE_LEFT_SYM;
        }
    pv_cc_tables_built = 1;
}
static void prim_merge_cc_tbl(const ctx_t *c) {
    const uint8_t *bm = c->bm; uint8_t *out = c->out; int K = c->n;
    uint8x16_t vleft = vdupq_n_u8(MERGE_LEFT_SYM), vdelta = vdupq_n_u8(MERGE_LEFT_SYM ^ MERGE_RIGHT_SYM);
    int j = 0;
    for (; j + 16 <= K; j += 16) {
        uint8x16_t bits = vcombine_u8(vld1_u8(pv_mask_to_bits[bm[j >> 3]]),
                                      vld1_u8(pv_mask_to_bits[bm[(j >> 3) + 1]]));
        vst1q_u8(out + j, veorq_u8(vleft, vandq_u8(vdelta, bits)));
    }
    for (; j < K; j++) out[j] = ((bm[j >> 3] >> (j & 7)) & 1) ? MERGE_RIGHT_SYM : MERGE_LEFT_SYM;
}
static void prim_merge_cc_blendtab(const ctx_t *c) {
    const uint8_t *bm = c->bm; uint8_t *out = c->out; int K = c->n; int j = 0;
    for (; j + 16 <= K; j += 16)
        vst1q_u8(out + j, vcombine_u8(vld1_u8(pv_blend_tab[bm[j >> 3]]),
                                      vld1_u8(pv_blend_tab[bm[(j >> 3) + 1]])));
    for (; j < K; j++) out[j] = ((bm[j >> 3] >> (j & 7)) & 1) ? MERGE_RIGHT_SYM : MERGE_LEFT_SYM;
}
static void prim_merge_cc_vtblq(const ctx_t *c) {
    const uint8_t *bm = c->bm; uint8_t *out = c->out; int K = c->n;
    static const uint8_t bit_pos16[16] = {1,2,4,8,16,32,64,128,1,2,4,8,16,32,64,128};
    uint8x16_t vbits = vld1q_u8(bit_pos16), one = vdupq_n_u8(1);
    uint16_t lr = (uint16_t)MERGE_LEFT_SYM | ((uint16_t)MERGE_RIGHT_SYM << 8);
    uint8x16_t c2s_vec = vreinterpretq_u8_u16(vdupq_n_u16(lr));
    int j = 0;
    for (; j + 16 <= K; j += 16) {
        uint8x16_t bm_dup = vcombine_u8(vdup_n_u8(bm[j >> 3]), vdup_n_u8(bm[(j >> 3) + 1]));
        uint8x16_t idx = vandq_u8(vtstq_u8(bm_dup, vbits), one);
        vst1q_u8(out + j, vqtbl1q_u8(c2s_vec, idx));
    }
    for (; j < K; j++) out[j] = ((bm[j >> 3] >> (j & 7)) & 1) ? MERGE_RIGHT_SYM : MERGE_LEFT_SYM;
}
static void prim_merge_cc_vtbl(const ctx_t *c) {
    const uint8_t *bm = c->bm; uint8_t *out = c->out; int K = c->n;
    static const uint8_t bit_pos8[8] = {1,2,4,8,16,32,64,128};
    uint8x8_t vbits = vld1_u8(bit_pos8), one = vdup_n_u8(1);
    uint16_t lr = (uint16_t)MERGE_LEFT_SYM | ((uint16_t)MERGE_RIGHT_SYM << 8);
    uint8x8_t c2s8 = vreinterpret_u8_u16(vdup_n_u16(lr));
    int j = 0;
    for (; j + 16 <= K; j += 16) {
        uint8x8_t m0 = vtst_u8(vdup_n_u8(bm[j >> 3]),       vbits);
        uint8x8_t m1 = vtst_u8(vdup_n_u8(bm[(j >> 3) + 1]), vbits);
        vst1_u8(out + j,     vtbl1_u8(c2s8, vand_u8(m0, one)));
        vst1_u8(out + j + 8, vtbl1_u8(c2s8, vand_u8(m1, one)));
    }
    for (; j < K; j++) out[j] = ((bm[j >> 3] >> (j & 7)) & 1) ? MERGE_RIGHT_SYM : MERGE_LEFT_SYM;
}
static void prim_merge_cc_d1flat(const ctx_t *c) {
    const uint8_t *bm = c->bm; uint8_t *out = c->out; int K = c->n;
    uint16_t lr_word = (uint16_t)MERGE_LEFT_SYM | ((uint16_t)MERGE_RIGHT_SYM << 8);
    uint8x16_t c2s_vec = vreinterpretq_u8_u16(vdupq_n_u16(lr_word));
    static const uint8_t dup_tab[16]  = {0,0,0,0,0,0,0,0, 1,1,1,1,1,1,1,1};
    static const int8_t  shift_tab[16]= {0,-1,-2,-3,-4,-5,-6,-7, 0,-1,-2,-3,-4,-5,-6,-7};
    uint8x16_t dup_v = vld1q_u8(dup_tab), one_v = vdupq_n_u8(1);
    int8x16_t  shift_v = vld1q_s8(shift_tab);
    int j = 0;
    for (; j + 16 <= K; j += 16) {
        uint16_t bm_word; memcpy(&bm_word, bm + (j >> 3), 2);
        uint8x16_t bm_lo = vreinterpretq_u8_u16(vsetq_lane_u16(bm_word, vdupq_n_u16(0), 0));
        uint8x16_t idx = vandq_u8(vshlq_u8(vqtbl1q_u8(bm_lo, dup_v), shift_v), one_v);
        vst1q_u8(out + j, vqtbl1q_u8(c2s_vec, idx));
    }
    for (; j < K; j++) out[j] = ((bm[j >> 3] >> (j & 7)) & 1) ? MERGE_RIGHT_SYM : MERGE_LEFT_SYM;
}

#endif /* USE_NEON_KERNELS */

/* ============================================================================
 * x86 (SSE4.1 / AVX2) merge_vec_vec variants — extracted verbatim from
 * extras/bench/bench_merge_{x86,avx2}.c.  The COM / prefix-sum forms reuse the
 * production expand_tab / expand_popcnt (in scope here, built by
 * prim_codec_init); merge_prepop derives its pshufb controls on the fly and
 * needs no table.  Same (bm,K,left,right,out) contract as the production merge.
 * ========================================================================== */
#if defined(__SSE4_1__) && !defined(__AVX512VBMI2__)

/* Shared SWAR per-byte popcount helper — guarded so the partition x86 section
 * (included first) and this section define it exactly once. */
#ifndef PV_X86_POPCNT_BYTES_U64
#define PV_X86_POPCNT_BYTES_U64 1
static inline uint64_t pv_popcnt_bytes_u64(uint64_t x) {
    x = x - ((x >> 1) & 0x5555555555555555ULL);
    x = (x & 0x3333333333333333ULL) + ((x >> 2) & 0x3333333333333333ULL);
    x = (x + (x >> 4)) & 0x0f0f0f0f0f0f0f0fULL;
    return x;
}
#endif

/* Muła pshufb nibble-LUT bytewise popcount (per-byte popcounts, 0..8). */
static inline __m128i pv_popcnt_bytes_xmm(__m128i v) {
    const __m128i lut  = _mm_setr_epi8(0,1,1,2,1,2,2,3,1,2,2,3,2,3,3,4);
    const __m128i mask = _mm_set1_epi8(0x0f);
    __m128i lo = _mm_and_si128(v, mask);
    __m128i hi = _mm_and_si128(_mm_srli_epi16(v, 4), mask);
    return _mm_add_epi8(_mm_shuffle_epi8(lut, lo), _mm_shuffle_epi8(lut, hi));
}
static inline uint64_t pv_popcnt_bytes_u64_pshufb(uint64_t x) {
    __m128i v = _mm_cvtsi64_si128((long long)x);
    return (uint64_t)_mm_cvtsi128_si64(pv_popcnt_bytes_xmm(v));
}

/* sse_com: 64 codes/iter, 8 independent 8-code pshufb merges, cursors from a
 * SWAR bytewise popcount prefix sum.  bench_merge_x86.c::sse_com_merge. */
static inline void prim_merge_vv_sse_com_x86(const uint8_t *bm, int K,
                      const uint8_t *left, const uint8_t *right, uint8_t *out) {
    int lc = 0, rc = 0, j = 0;
    for (; j + 64 <= K; j += 64) {
        uint64_t mask_u64; memcpy(&mask_u64, bm + (j >> 3), 8);
        uint64_t pfx = pv_popcnt_bytes_u64(mask_u64) * 0x0101010101010101ULL;
        #define PV_CH(K_) do {                                                   \
            uint8_t m  = (uint8_t)(mask_u64 >> (8*(K_)));                        \
            uint32_t cr = (K_)==0 ? 0 : (uint32_t)((pfx >> (8*((K_)-1))) & 0xFF);\
            uint32_t cl = 8*(K_) - cr;                                          \
            __m128i L = _mm_loadl_epi64((const __m128i *)(left + lc + cl));      \
            __m128i R = _mm_loadl_epi64((const __m128i *)(right + rc + cr));     \
            __m128i both = _mm_unpacklo_epi64(L, R);                            \
            __m128i shuf = _mm_loadl_epi64((const __m128i *)expand_tab[m]);      \
            _mm_storel_epi64((__m128i *)(out + j + 8*(K_)),                     \
                             _mm_shuffle_epi8(both, shuf));                      \
        } while (0)
        PV_CH(0); PV_CH(1); PV_CH(2); PV_CH(3); PV_CH(4); PV_CH(5); PV_CH(6); PV_CH(7);
        #undef PV_CH
        uint32_t total_r = (uint32_t)(pfx >> 56);
        rc += total_r; lc += 64 - total_r;
    }
    for (; j < K; j++) { int b = (bm[j>>3] >> (j&7)) & 1; out[j] = b ? right[rc++] : left[lc++]; }
}
static void prim_merge_vv_sse_com(const ctx_t *c){
    prim_merge_vv_sse_com_x86(c->bm, c->n, c->merge_left, c->merge_right, c->out);
}

/* sse_com_pshufb: sse_com but pshufb (Muła) bytewise popcount. */
static inline void prim_merge_vv_sse_com_pshufb_x86(const uint8_t *bm, int K,
                      const uint8_t *left, const uint8_t *right, uint8_t *out) {
    int lc = 0, rc = 0, j = 0;
    for (; j + 64 <= K; j += 64) {
        uint64_t mask_u64; memcpy(&mask_u64, bm + (j >> 3), 8);
        uint64_t pfx = pv_popcnt_bytes_u64_pshufb(mask_u64) * 0x0101010101010101ULL;
        #define PV_CHP(K_) do {                                                  \
            uint8_t m  = (uint8_t)(mask_u64 >> (8*(K_)));                        \
            uint32_t cr = (K_)==0 ? 0 : (uint32_t)((pfx >> (8*((K_)-1))) & 0xFF);\
            uint32_t cl = 8*(K_) - cr;                                          \
            __m128i L = _mm_loadl_epi64((const __m128i *)(left + lc + cl));      \
            __m128i R = _mm_loadl_epi64((const __m128i *)(right + rc + cr));     \
            __m128i both = _mm_unpacklo_epi64(L, R);                            \
            __m128i shuf = _mm_loadl_epi64((const __m128i *)expand_tab[m]);      \
            _mm_storel_epi64((__m128i *)(out + j + 8*(K_)),                     \
                             _mm_shuffle_epi8(both, shuf));                      \
        } while (0)
        PV_CHP(0); PV_CHP(1); PV_CHP(2); PV_CHP(3); PV_CHP(4); PV_CHP(5); PV_CHP(6); PV_CHP(7);
        #undef PV_CHP
        uint32_t total_r = (uint32_t)(pfx >> 56);
        rc += total_r; lc += 64 - total_r;
    }
    for (; j < K; j++) { int b = (bm[j>>3] >> (j&7)) & 1; out[j] = b ? right[rc++] : left[lc++]; }
}
static void prim_merge_vv_sse_com_pshufb(const ctx_t *c){
    prim_merge_vv_sse_com_pshufb_x86(c->bm, c->n, c->merge_left, c->merge_right, c->out);
}

/* sse_com128: 128 codes/iter, 16 chunks, pshufb popcount of all 16 bm bytes
 * in one shot, two u64 prefix sums (hi biased by lo total). */
static inline void prim_merge_vv_sse_com128_x86(const uint8_t *bm, int K,
                      const uint8_t *left, const uint8_t *right, uint8_t *out) {
    int lc = 0, rc = 0, j = 0;
    for (; j + 128 <= K; j += 128) {
        __m128i bmv = _mm_loadu_si128((const __m128i *)(bm + (j >> 3)));
        __m128i pcv = pv_popcnt_bytes_xmm(bmv);
        uint64_t mask_lo = (uint64_t)_mm_cvtsi128_si64(bmv);
        uint64_t mask_hi = (uint64_t)_mm_extract_epi64(bmv, 1);
        uint64_t pc_lo   = (uint64_t)_mm_cvtsi128_si64(pcv);
        uint64_t pc_hi   = (uint64_t)_mm_extract_epi64(pcv, 1);
        uint64_t pfx_lo = pc_lo * 0x0101010101010101ULL;
        uint64_t pfx_hi = pc_hi * 0x0101010101010101ULL;
        pfx_hi += (pfx_lo >> 56) * 0x0101010101010101ULL;
        #define PV_CH16(K_, BMW, PFX, EBASE) do {                               \
            uint8_t m  = (uint8_t)((BMW) >> (8*(K_)));                           \
            uint32_t cr = ((K_)==0 && (EBASE)==0) ? 0                            \
                        : ((K_)==0 ? (uint32_t)((pfx_lo>>56)&0xFF)              \
                        : (uint32_t)(((PFX) >> (8*((K_)-1))) & 0xFF));          \
            uint32_t cl = ((EBASE)+8*(K_)) - cr;                               \
            __m128i L = _mm_loadl_epi64((const __m128i *)(left + lc + cl));      \
            __m128i R = _mm_loadl_epi64((const __m128i *)(right + rc + cr));     \
            __m128i both = _mm_unpacklo_epi64(L, R);                            \
            __m128i shuf = _mm_loadl_epi64((const __m128i *)expand_tab[m]);      \
            _mm_storel_epi64((__m128i *)(out + j + (EBASE) + 8*(K_)),           \
                             _mm_shuffle_epi8(both, shuf));                      \
        } while (0)
        PV_CH16(0,mask_lo,pfx_lo,0); PV_CH16(1,mask_lo,pfx_lo,0); PV_CH16(2,mask_lo,pfx_lo,0); PV_CH16(3,mask_lo,pfx_lo,0);
        PV_CH16(4,mask_lo,pfx_lo,0); PV_CH16(5,mask_lo,pfx_lo,0); PV_CH16(6,mask_lo,pfx_lo,0); PV_CH16(7,mask_lo,pfx_lo,0);
        PV_CH16(0,mask_hi,pfx_hi,64); PV_CH16(1,mask_hi,pfx_hi,64); PV_CH16(2,mask_hi,pfx_hi,64); PV_CH16(3,mask_hi,pfx_hi,64);
        PV_CH16(4,mask_hi,pfx_hi,64); PV_CH16(5,mask_hi,pfx_hi,64); PV_CH16(6,mask_hi,pfx_hi,64); PV_CH16(7,mask_hi,pfx_hi,64);
        #undef PV_CH16
        uint32_t total_r = (uint32_t)(pfx_hi >> 56);
        rc += total_r; lc += 128 - total_r;
    }
    for (; j < K; j++) { int b = (bm[j>>3] >> (j&7)) & 1; out[j] = b ? right[rc++] : left[lc++]; }
}
static void prim_merge_vv_sse_com128(const ctx_t *c){
    prim_merge_vv_sse_com128_x86(c->bm, c->n, c->merge_left, c->merge_right, c->out);
}

#if defined(__AVX2__)
/* avx2_com: 16 codes per _mm256_shuffle_epi8 (two independent 128-bit lanes),
 * COM cursors.  bench_merge_x86.c::avx2_com_merge. */
static inline void prim_merge_vv_avx2_com_x86(const uint8_t *bm, int K,
                      const uint8_t *left, const uint8_t *right, uint8_t *out) {
    int lc = 0, rc = 0, j = 0;
    for (; j + 64 <= K; j += 64) {
        uint64_t mask_u64; memcpy(&mask_u64, bm + (j >> 3), 8);
        uint64_t pfx = pv_popcnt_bytes_u64(mask_u64) * 0x0101010101010101ULL;
        #define PV_PAIR(P) do {                                                 \
            int k0 = 2*(P), k1 = 2*(P)+1;                                       \
            uint8_t ma = (uint8_t)(mask_u64 >> (8*k0));                         \
            uint8_t mb = (uint8_t)(mask_u64 >> (8*k1));                         \
            uint32_t cra = k0==0 ? 0 : (uint32_t)((pfx >> (8*(k0-1))) & 0xFF);  \
            uint32_t crb =          (uint32_t)((pfx >> (8*(k1-1))) & 0xFF);     \
            uint32_t cla = 8*k0 - cra, clb = 8*k1 - crb;                        \
            __m128i La = _mm_loadl_epi64((const __m128i *)(left + lc + cla));    \
            __m128i Ra = _mm_loadl_epi64((const __m128i *)(right + rc + cra));   \
            __m128i Lb = _mm_loadl_epi64((const __m128i *)(left + lc + clb));    \
            __m128i Rb = _mm_loadl_epi64((const __m128i *)(right + rc + crb));   \
            __m256i both = _mm256_set_m128i(_mm_unpacklo_epi64(Lb, Rb),         \
                                            _mm_unpacklo_epi64(La, Ra));        \
            __m128i sa = _mm_loadl_epi64((const __m128i *)expand_tab[ma]);       \
            __m128i sb = _mm_loadl_epi64((const __m128i *)expand_tab[mb]);       \
            __m256i shuf = _mm256_set_m128i(sb, sa);                            \
            __m256i o = _mm256_shuffle_epi8(both, shuf);                        \
            _mm_storel_epi64((__m128i *)(out + j + 8*k0), _mm256_castsi256_si128(o)); \
            _mm_storel_epi64((__m128i *)(out + j + 8*k1), _mm256_extracti128_si256(o,1)); \
        } while (0)
        PV_PAIR(0); PV_PAIR(1); PV_PAIR(2); PV_PAIR(3);
        #undef PV_PAIR
        uint32_t total_r = (uint32_t)(pfx >> 56);
        rc += total_r; lc += 64 - total_r;
    }
    for (; j < K; j++) { int b = (bm[j>>3] >> (j&7)) & 1; out[j] = b ? right[rc++] : left[lc++]; }
}
static void prim_merge_vv_avx2_com(const ctx_t *c){
    prim_merge_vv_avx2_com_x86(c->bm, c->n, c->merge_left, c->merge_right, c->out);
}

/* prepop: 16 bytes/iter, prefix-popcount pshufb controls derived on the fly
 * (no expand_tab).  bench_merge_avx2.c::merge_prepop. */
static inline __m256i pv_popcnt_epi16_avx2(__m256i v) {
    const __m256i lookup = _mm256_setr_epi8(
        0,1,1,2, 1,2,2,3, 1,2,2,3, 2,3,3,4,
        0,1,1,2, 1,2,2,3, 1,2,2,3, 2,3,3,4);
    const __m256i nibble_mask = _mm256_set1_epi8(0x0F);
    __m256i lo = _mm256_and_si256(v, nibble_mask);
    __m256i hi = _mm256_and_si256(_mm256_srli_epi16(v, 4), nibble_mask);
    __m256i byte_pop = _mm256_add_epi8(
        _mm256_shuffle_epi8(lookup, lo),
        _mm256_shuffle_epi8(lookup, hi));
    return _mm256_maddubs_epi16(byte_pop, _mm256_set1_epi8(1));
}
static inline void prim_merge_vv_prepop_x86(const uint8_t *bm, int K,
                      const uint8_t *left, const uint8_t *right, uint8_t *out) {
    int lc = 0, rc = 0, j = 0;
    const __m256i prefix_masks_16 = _mm256_setr_epi16(
        0x0001, 0x0003, 0x0007, 0x000F, 0x001F, 0x003F, 0x007F, 0x00FF,
        0x01FF, 0x03FF, 0x07FF, 0x0FFF, 0x1FFF, 0x3FFF, 0x7FFF, (short)0xFFFF);
    const __m128i ones        = _mm_set1_epi8(1);
    const __m128i indices_16  = _mm_setr_epi8(0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15);
    const __m128i bcast_shuf  = _mm_setr_epi8(0,0,0,0,0,0,0,0, 1,1,1,1,1,1,1,1);
    const __m128i bit_pos     = _mm_setr_epi8(1,2,4,8,16,32,64,(char)128,
                                                1,2,4,8,16,32,64,(char)128);
    for (; j + 16 <= K; j += 16) {
        uint16_t m; memcpy(&m, bm + (j >> 3), 2);
        __m256i mvec = _mm256_set1_epi16((short)m);
        __m256i masked = _mm256_and_si256(mvec, prefix_masks_16);
        __m256i prefix = pv_popcnt_epi16_avx2(masked);
        __m256i packed = _mm256_packus_epi16(prefix, _mm256_setzero_si256());
        __m256i prefix_compact = _mm256_permute4x64_epi64(packed, 0xD8);
        __m128i prefix_u8 = _mm256_castsi256_si128(prefix_compact);
        __m128i shuf_right = _mm_sub_epi8(prefix_u8, ones);
        __m128i shuf_left  = _mm_sub_epi8(indices_16, prefix_u8);
        __m128i left_data  = _mm_loadu_si128((const __m128i *)(left  + lc));
        __m128i right_data = _mm_loadu_si128((const __m128i *)(right + rc));
        __m128i g_left  = _mm_shuffle_epi8(left_data,  shuf_left);
        __m128i g_right = _mm_shuffle_epi8(right_data, shuf_right);
        __m128i mvec_b = _mm_cvtsi32_si128((int32_t)m);
        __m128i mbcast = _mm_shuffle_epi8(mvec_b, bcast_shuf);
        __m128i mask_vec = _mm_cmpeq_epi8(_mm_and_si128(mbcast, bit_pos), bit_pos);
        __m128i out_vec = _mm_blendv_epi8(g_left, g_right, mask_vec);
        _mm_storeu_si128((__m128i *)(out + j), out_vec);
        int nr = __builtin_popcount((uint32_t)m);
        rc += nr; lc += 16 - nr;
    }
    for (; j < K; j++) { int b = (bm[j>>3] >> (j&7)) & 1; out[j] = b ? right[rc++] : left[lc++]; }
}
static void prim_merge_vv_prepop(const ctx_t *c){
    prim_merge_vv_prepop_x86(c->bm, c->n, c->merge_left, c->merge_right, c->out);
}
#endif /* __AVX2__ */
#endif /* __SSE4_1__ */

/* ============================================================================
 * merge_flat : asof-e5a199a — pre-widening 16-codes/iter AVX-512 merge_flat
 *   merge_flat_d{2..7}_avx512 from e5a199a~1:src/pivco_huffman_primitives_
 *   avx512.h (before the 64-codes/iter widening in e5a199a).  Each per-D
 *   kernel unpacks 16 D-bit codes via the shared flat_d*_unpack_avx512
 *   helpers, then applies the c2s table: pshufb (D<=4), vpermb on an ymm/zmm
 *   c2s (D=5/6), or vpermi2b across two zmm tables (D=7).  Reuses the
 *   production flat unpack helpers via pivco_huffman_avx512_flat.h.
 * ========================================================================== */
#if defined(__AVX512VBMI2__) && defined(__AVX512VBMI__)
#include <immintrin.h>
#include "pivco_huffman_avx512_flat.h"  /* flat_d{2..7}_unpack_avx512* */

/* Extract D bits at bit_pos from in (D<=16) — scalar tail. */
static inline uint32_t pv_extract_D_bits_avx512(const uint8_t *in, int bit_pos, int D) {
    int byte_idx = bit_pos >> 3, bit_off = bit_pos & 7;
    uint32_t val = (uint32_t)in[byte_idx];
    if (bit_off + D > 8)  val |= ((uint32_t)in[byte_idx + 1]) << 8;
    if (bit_off + D > 16) val |= ((uint32_t)in[byte_idx + 2]) << 16;
    return (val >> bit_off) & ((1u << D) - 1);
}

static void pv_merge_flat_e5a199a_d2(uint8_t *symbols, int n, const uint8_t *bm, const uint8_t *c2s) {
    uint32_t c2s_lo; memcpy(&c2s_lo, c2s, 4);
    __m128i c2s_vec = _mm_set1_epi32((int32_t)c2s_lo);
    int i = 0;
    for (; i + 16 <= n; i += 16) {
        __m128i codes = flat_d2_unpack_avx512(bm + (i >> 2));
        _mm_storeu_si128((__m128i *)(symbols + i), _mm_shuffle_epi8(c2s_vec, codes));
    }
    for (; i + 4 <= n; i += 4) {
        uint8_t b = bm[i >> 2];
        symbols[i] = c2s[b & 3]; symbols[i+1] = c2s[(b>>2)&3];
        symbols[i+2] = c2s[(b>>4)&3]; symbols[i+3] = c2s[(b>>6)&3];
    }
    for (; i < n; i++) symbols[i] = c2s[pv_extract_D_bits_avx512(bm, i*2, 2)];
}
static void pv_merge_flat_e5a199a_d3(uint8_t *symbols, int n, const uint8_t *bm, const uint8_t *c2s) {
    uint64_t c2s_lo; memcpy(&c2s_lo, c2s, 8);
    __m128i c2s_vec = _mm_cvtsi64_si128((int64_t)c2s_lo);
    int i = 0, fast_end = n >= 16 ? n - 16 : 0;
    for (; i + 16 <= fast_end; i += 16) {
        __m128i codes = flat_d3_unpack_avx512_fast(bm + ((i * 3) >> 3));
        _mm_storeu_si128((__m128i *)(symbols + i), _mm_shuffle_epi8(c2s_vec, codes));
    }
    if (i + 16 <= n) {
        __m128i codes = flat_d3_unpack_avx512_safe(bm + ((i * 3) >> 3));
        _mm_storeu_si128((__m128i *)(symbols + i), _mm_shuffle_epi8(c2s_vec, codes)); i += 16;
    }
    for (; i < n; i++) symbols[i] = c2s[pv_extract_D_bits_avx512(bm, i*3, 3)];
}
static void pv_merge_flat_e5a199a_d4(uint8_t *symbols, int n, const uint8_t *bm, const uint8_t *c2s) {
    __m128i c2s_vec = _mm_loadu_si128((const __m128i *)c2s);
    int i = 0;
    for (; i + 16 <= n; i += 16) {
        __m128i codes = flat_d4_unpack_avx512(bm + (i >> 1));
        _mm_storeu_si128((__m128i *)(symbols + i), _mm_shuffle_epi8(c2s_vec, codes));
    }
    for (; i < n; i++) symbols[i] = c2s[pv_extract_D_bits_avx512(bm, i*4, 4)];
}
static void pv_merge_flat_e5a199a_d5(uint8_t *symbols, int n, const uint8_t *bm, const uint8_t *c2s) {
    __m256i c2s_vec = _mm256_loadu_si256((const __m256i *)c2s);
    int i = 0, fast_end = n >= 16 ? n - 16 : 0;
    for (; i + 16 <= fast_end; i += 16) {
        __m128i codes = flat_d5_unpack_avx512_fast(bm + ((i * 5) >> 3));
        __m256i syms = _mm256_permutexvar_epi8(_mm256_zextsi128_si256(codes), c2s_vec);
        _mm_storeu_si128((__m128i *)(symbols + i), _mm256_castsi256_si128(syms));
    }
    if (i + 16 <= n) {
        __m128i codes = flat_d5_unpack_avx512_safe(bm + ((i * 5) >> 3));
        __m256i syms = _mm256_permutexvar_epi8(_mm256_zextsi128_si256(codes), c2s_vec);
        _mm_storeu_si128((__m128i *)(symbols + i), _mm256_castsi256_si128(syms)); i += 16;
    }
    for (; i < n; i++) symbols[i] = c2s[pv_extract_D_bits_avx512(bm, i*5, 5)];
}
static void pv_merge_flat_e5a199a_d6(uint8_t *symbols, int n, const uint8_t *bm, const uint8_t *c2s) {
    __m512i c2s_vec = _mm512_loadu_si512((const __m512i *)c2s);
    int i = 0, fast_end = n >= 16 ? n - 16 : 0;
    for (; i + 16 <= fast_end; i += 16) {
        __m128i codes = flat_d6_unpack_avx512_fast(bm + ((i * 6) >> 3));
        __m512i syms = _mm512_permutexvar_epi8(_mm512_castsi128_si512(codes), c2s_vec);
        _mm_storeu_si128((__m128i *)(symbols + i), _mm512_castsi512_si128(syms));
    }
    if (i + 16 <= n) {
        __m128i codes = flat_d6_unpack_avx512_safe(bm + ((i * 6) >> 3));
        __m512i syms = _mm512_permutexvar_epi8(_mm512_castsi128_si512(codes), c2s_vec);
        _mm_storeu_si128((__m128i *)(symbols + i), _mm512_castsi512_si128(syms)); i += 16;
    }
    for (; i < n; i++) symbols[i] = c2s[pv_extract_D_bits_avx512(bm, i*6, 6)];
}
static void pv_merge_flat_e5a199a_d7(uint8_t *symbols, int n, const uint8_t *bm, const uint8_t *c2s) {
    __m512i c2s_lo = _mm512_loadu_si512((const __m512i *)c2s);
    __m512i c2s_hi = _mm512_loadu_si512((const __m512i *)(c2s + 64));
    int i = 0, fast_end = n >= 16 ? n - 16 : 0;
    for (; i + 16 <= fast_end; i += 16) {
        __m128i codes = flat_d7_unpack_avx512_fast(bm + ((i * 7) >> 3));
        __m512i syms = _mm512_permutex2var_epi8(c2s_lo, _mm512_castsi128_si512(codes), c2s_hi);
        _mm_storeu_si128((__m128i *)(symbols + i), _mm512_castsi512_si128(syms));
    }
    if (i + 16 <= n) {
        __m128i codes = flat_d7_unpack_avx512_safe(bm + ((i * 7) >> 3));
        __m512i syms = _mm512_permutex2var_epi8(c2s_lo, _mm512_castsi128_si512(codes), c2s_hi);
        _mm_storeu_si128((__m128i *)(symbols + i), _mm512_castsi512_si128(syms)); i += 16;
    }
    for (; i < n; i++) symbols[i] = c2s[pv_extract_D_bits_avx512(bm, i*7, 7)];
}

static void prim_merge_flat_e5a199a(const ctx_t *c) {
    switch (c->D) {
    case 2: pv_merge_flat_e5a199a_d2(c->out, c->n, c->bm, c->c2s); break;
    case 3: pv_merge_flat_e5a199a_d3(c->out, c->n, c->bm, c->c2s); break;
    case 4: pv_merge_flat_e5a199a_d4(c->out, c->n, c->bm, c->c2s); break;
    case 5: pv_merge_flat_e5a199a_d5(c->out, c->n, c->bm, c->c2s); break;
    case 6: pv_merge_flat_e5a199a_d6(c->out, c->n, c->bm, c->c2s); break;
    case 7: pv_merge_flat_e5a199a_d7(c->out, c->n, c->bm, c->c2s); break;
    default: break;
    }
}
#endif /* __AVX512VBMI2__ && __AVX512VBMI__ */

/* ============================================================================
 * Registry — merge family (no-op where the ISA is unavailable)
 * ========================================================================== */
static void pv_register_merge(void) {
#if defined(__AVX512VBMI2__) && defined(__AVX512VBMI__)
    for (int d = 2; d <= 7; d++)
        PV_VARIANT_D(ST_MERGE_FLAT, "asof-e5a199a", d, PV_ISA_AVX512,
                     "e5a199a~1 merge_flat_dN_avx512",
                     "pre-widening 16 codes/iter", 0, prim_merge_flat_e5a199a);
#endif
#if defined(USE_NEON_KERNELS)
    pv_build_cc_tables();
    /* merge_vec_vec */
    PV_VARIANT(ST_MERGE_VEC_VEC, "asof-4dd08e3", PV_ISA_NEON, "4dd08e3 (2026-05-10)",
               "genesis bottom-up tree_merge: stride-8, 1 vqtbl1/8, serial cursor", 0, prim_merge_vv_asof_4dd08e3);
    PV_VARIANT(ST_MERGE_VEC_VEC, "cursor16",  PV_ISA_NEON, "historical 2c8606e (pre-5cccccc)",
               "stride-16 shipped merge before COM64; loop-carried cursor", 0, prim_merge_vv_cursor16);
    PV_VARIANT(ST_MERGE_VEC_VEC, "prefix128", PV_ISA_NEON, "Jeff Plaisance / PR #4",
               "vpaddlq fold + u16 SWAR prefix, 128/iter; M1 Max win, ARM-untested", 0, prim_merge_vv_prefix128);
    PV_VARIANT(ST_MERGE_VEC_VEC, "prefix64",  PV_ISA_NEON, "Jeff Plaisance / PR #4",
               "prefix128 block style at 64-code stride", 0, prim_merge_vv_prefix64);
    PV_VARIANT(ST_MERGE_VEC_VEC, "pcpc",      PV_ISA_NEON, "bench_prim experiment",
               "precomputed-popcount (pre-filled table); M4 wash, store-bound", 0, prim_merge_vv_pcpc);
    PV_VARIANT(ST_MERGE_VEC_VEC, "pcpc_full", PV_ISA_NEON, "bench_prim experiment",
               "pcpc + table fill inside the timed region", 0, prim_merge_vv_pcpc_full);
    PV_VARIANT(ST_MERGE_VEC_VEC, "unroll8",   PV_ISA_NEON, "bench_prim experiment",
               "8-way unroll, in-register vcnt popcount; doubles L/R load count", 0, prim_merge_vv_unroll8);
    PV_VARIANT(ST_MERGE_VEC_VEC, "com32",     PV_ISA_NEON, "bench_merge_neon.c",
               "COM prefix-sum cursor-decouple, 32/iter (2 chunks); narrower than shipped com64", 0, prim_merge_vv_com32);
    PV_VARIANT(ST_MERGE_VEC_VEC, "com128",    PV_ISA_NEON, "bench_merge_neon.c",
               "COM 128/iter (8 chunks, lo/hi u64 split + cross-half bias); regalloc-heavy, loses on Graviton", 0, prim_merge_vv_com128);
    /* merge_cst_cst */
    PV_VARIANT(ST_MERGE_CST_CST, "tbl",       PV_ISA_NEON, "bench_prim experiment",
               "256x8 mask->0xFF/0x00 LUT + vand/veor blend", 0, prim_merge_cc_tbl);
    PV_VARIANT(ST_MERGE_CST_CST, "blendtab",  PV_ISA_NEON, "bench_prim experiment",
               "precomputed 256x8 blended-output LUT (no hot-loop compute)", 0, prim_merge_cc_blendtab);
    PV_VARIANT(ST_MERGE_CST_CST, "vtblq",     PV_ISA_NEON, "bench_prim experiment",
               "16-lane vtstq + vqtbl1q", 0, prim_merge_cc_vtblq);
    PV_VARIANT(ST_MERGE_CST_CST, "vtbl",      PV_ISA_NEON, "bench_prim experiment",
               "8-lane vtst + vtbl1 x2", 0, prim_merge_cc_vtbl);
    PV_VARIANT(ST_MERGE_CST_CST, "d1flat",    PV_ISA_NEON, "bench_prim experiment",
               "D=1 flat-decode shape (vshl + vqtbl1q)", 0, prim_merge_cc_d1flat);
#endif
#if defined(__SSE4_1__) && !defined(__AVX512VBMI2__)
    /* merge_vec_vec — x86 COM / prefix-sum forms (IDEAS: x86 COM merge). */
    PV_VARIANT(ST_MERGE_VEC_VEC, "sse_com",        PV_ISA_SSE4, "bench_merge_x86.c",
               "64 codes/iter, 8 pshufb merges, SWAR-popcnt prefix cursors", 0,
               prim_merge_vv_sse_com);
    PV_VARIANT(ST_MERGE_VEC_VEC, "sse_com_pshufb", PV_ISA_SSE4, "bench_merge_x86.c",
               "sse_com but Mula pshufb bytewise popcount", 0,
               prim_merge_vv_sse_com_pshufb);
    PV_VARIANT(ST_MERGE_VEC_VEC, "sse_com128",     PV_ISA_SSE4, "bench_merge_x86.c",
               "128 codes/iter, full-width pshufb popcount", 0,
               prim_merge_vv_sse_com128);
#if defined(__AVX2__)
    PV_VARIANT(ST_MERGE_VEC_VEC, "avx2_com",       PV_ISA_AVX2, "bench_merge_x86.c",
               "16 codes/_mm256_shuffle_epi8 (2 lanes), COM cursors", 0,
               prim_merge_vv_avx2_com);
    PV_VARIANT(ST_MERGE_VEC_VEC, "prepop",         PV_ISA_AVX2, "bench_merge_avx2.c",
               "16 B/iter, on-the-fly prefix-popcount pshufb controls (no tab)", 0,
               prim_merge_vv_prepop);
#endif
#endif
}

#endif /* PIVCO_PRIM_VARIANTS_MERGE_H */
