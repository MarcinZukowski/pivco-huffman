/* bench/prim_variants/prims-merge.h — merge-family variant graveyard.
 *
 * Logical primitives: merge_vec_vec (ST_MERGE_VEC_VEC), merge_cst_vec,
 * merge_cst_cst, merge_flat.  See prims.h for the contract +
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
 * merge_vec_vec : aqrit-notab — table-free shuffle-index synthesis (issue #26)
 *
 * From aqrit's gist (aa27a7930d0cab9ab97112236aae2182), verbatim except the
 * signature const/int match.  The shuffle control is COMPUTED, not looked up:
 * an inclusive per-lane prefix popcount comes from vcnt over the bitmap byte
 * broadcast AND'ed with progressive masks {0x1,0x3,..,0xFF}, the right-source
 * index is the exclusive prefix (vext by 15), the left index is (16+lane) -
 * r_shuf, and the L/R select mask falls out as exclusive - inclusive (0x00 or
 * 0xFF per lane) — one vsub, no table.  ~12 ALU ops per 16 outputs vs the
 * production two-table SABD merge's ~6 + two 16 B index-table loads; the trade
 * is pure-ALU (no 8 KiB g_merge_shuf0/1 footprint, no bitmap->table load-load
 * dependency) vs more ops.  aqrit self-closed the issue doubting x86 (byte
 * popcount needs pshufb there); this NEON form he could not run.
 * ==========================================================================*/
static const uint8_t pv_aqrit_popcnt_mask[16] = { /* inclusive prefix scan over packed bits */
    0x1, 0x3, 0x7, 0x0F, 0x1F, 0x3F, 0x7F, 0xFF,
    0x1, 0x3, 0x7, 0x0F, 0x1F, 0x3F, 0x7F, 0xFF
};
static const uint8_t pv_aqrit_left_id[16] = { 16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31 };

/* `left` and `right` must end with 16 bytes of padding (bench buffers have
 * the canary slack the production speculative loads already rely on). */
static inline void pv_merge_vec_vec_aqrit_neon(const uint8_t *bm, int K,
                                               const uint8_t *left,
                                               const uint8_t *right,
                                               uint8_t *out)
{
    size_t i = 0;
    if (K >= 16) {
        const uint8x16_t popcnt_mask_v = vld1q_u8(pv_aqrit_popcnt_mask);
        const uint8x16_t left_id_v = vld1q_u8(pv_aqrit_left_id);

        const size_t end = ((size_t)K >> 3) & ~(size_t)1;
        do {
            uint8x8_t bm_lo = vdup_n_u8(bm[i]);
            uint8x8_t bm_hi = vdup_n_u8(bm[i + 1]);
            uint8x16x2_t src;
            src.val[0] = vld1q_u8(right);
            src.val[1] = vld1q_u8(left);

            uint8x16_t prefix_half = vcntq_u8(vandq_u8(vcombine_u8(bm_lo, bm_hi), popcnt_mask_v));
            uint8x16_t carry = vcombine_u8(vdup_n_u8(0), vcnt_u8(bm_lo));
            uint8x16_t prefix = vaddq_u8(prefix_half, carry);

            uint8_t r_popcnt = vgetq_lane_u8(prefix, 15);
            right += r_popcnt;
            left += 16 - r_popcnt;

            uint8x16_t r_shuf = vextq_u8(vdupq_n_u8(0), prefix, 15); /* exclusive */
            uint8x16_t l_shuf = vsubq_u8(left_id_v, r_shuf);
            uint8x16_t r_select = vsubq_u8(r_shuf, prefix);
            uint8x16_t shuf = vbslq_u8(r_select, r_shuf, l_shuf);

            vst1q_u8(&out[i * 8], vqtbl2q_u8(src, shuf));
            i += 2;
        } while (i < end);
    }

    if (K & 8) { /* just for kicks */
        const uint64_t mul = UINT64_C(0x0002040810204080);
        const uint64_t kx01 = UINT64_C(0x0101010101010101);
        const uint64_t left_adj = UINT64_C(0x0F0E0D0C0B0A0908);

        uint64_t x = bm[i];
        uint8x8x2_t src;
        src.val[0] = vld1_u8(right);
        src.val[1] = vld1_u8(left);

        uint64_t unpacked = (((x & 0xFE) * mul) ^ x) & kx01; /* unpack bits */
        uint64_t prefix = unpacked * kx01;
        size_t r_popcnt = (size_t)(prefix >> 56);
        right += r_popcnt;
        left += ((size_t)8) - r_popcnt;

        uint64_t r_shuf = prefix - unpacked;
        uint64_t l_shuf = left_adj - r_shuf;
        uint64_t r_select = (unpacked << 8) - unpacked;
        uint64_t shuf = (r_select & r_shuf) | (~r_select & l_shuf);
        vst1_u8(&out[i * 8], vtbl2_u8(src, vcreate_u8(shuf)));
        i++;
    }

    if (K & 7) {
        size_t x = bm[i];
        uint8_t *end = &out[K];
        out += i * 8;
        do {
            size_t b = x & 1; x >>= 1;
            uint8_t r = *right; right += b;
            uint8_t l = *left; left += b ^ 1;
            *out = b ? r : l;
        } while (++out < end);
    }
}
static void prim_merge_vv_aqrit(const ctx_t *c){
    pv_merge_vec_vec_aqrit_neon(c->bm, c->n, c->merge_left, c->merge_right, c->out);
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

/* com64 — the V5 stride-64 COM merge that was production before the
 * two-table merge landed (saved here for A/B). */
static void prim_merge_vv_com64_neon(const uint8_t *bm, int K,
                                     const uint8_t *left,
                                     const uint8_t *right,
                                     uint8_t *out)
{
    int lc = 0, rc = 0;
    int j = 0;

    /* V5 main: 64 codes per iter, 4 independent chunks. */
    for (; j + 64 <= K; j += 64) {
        uint64_t mask_u64;
        memcpy(&mask_u64, bm + (j >> 3), 8);
        uint8x8_t bm_v = vcreate_u8(mask_u64);
        uint8x8_t pc_v = vcnt_u8(bm_v);
        uint64_t pc_u64 = vget_lane_u64(vreinterpret_u64_u8(pc_v), 0);
        uint64_t pfx = pc_u64 * 0x0101010101010101ULL;

        uint8_t cr0 = 0;
        uint8_t cr1 = (uint8_t)(pfx >>  8);
        uint8_t cr2 = (uint8_t)(pfx >> 24);
        uint8_t cr3 = (uint8_t)(pfx >> 40);
        uint8_t in0 = (uint8_t)pc_u64;
        uint8_t in1 = (uint8_t)(pc_u64 >> 16);
        uint8_t in2 = (uint8_t)(pc_u64 >> 32);
        uint8_t in3 = (uint8_t)(pc_u64 >> 48);

        uint8_t m0 = (uint8_t) mask_u64;
        uint8_t m1 = (uint8_t)(mask_u64 >>  8);
        uint8_t m2 = (uint8_t)(mask_u64 >> 16);
        uint8_t m3 = (uint8_t)(mask_u64 >> 24);
        uint8_t m4 = (uint8_t)(mask_u64 >> 32);
        uint8_t m5 = (uint8_t)(mask_u64 >> 40);
        uint8_t m6 = (uint8_t)(mask_u64 >> 48);
        uint8_t m7 = (uint8_t)(mask_u64 >> 56);

#define _V5_CHUNK(idx, cr, in, ma, mb) do {                                  \
            uint8_t cl = (uint8_t)((idx)*16 - (cr));                         \
            uint8x16_t L = vld1q_u8(left  + lc + cl);                        \
            uint8x16_t R = vld1q_u8(right + rc + (cr));                      \
            uint8x16_t both = vcombine_u8(vget_low_u8(L), vget_low_u8(R));   \
            uint8x8_t  s0   = vld1_u8(expand_tab[ma]);                       \
            vst1_u8(out + j + (idx)*16,     vqtbl1_u8(both, s0));            \
            uint8x16x2_t src = {{ L, R }};                                   \
            uint8x8_t s1 = vld1_u8(expand_tab_pre[in][mb]);                  \
            vst1_u8(out + j + (idx)*16 + 8, vqtbl2_u8(src, s1));             \
        } while (0)
        _V5_CHUNK(0, cr0, in0, m0, m1);
        _V5_CHUNK(1, cr1, in1, m2, m3);
        _V5_CHUNK(2, cr2, in2, m4, m5);
        _V5_CHUNK(3, cr3, in3, m6, m7);
#undef _V5_CHUNK

        uint8_t total_r = (uint8_t)(pfx >> 56);
        rc += total_r;
        lc += 64 - total_r;
    }

    /* V4 stride-16 fallback for the residual (handles K mod 128). */
    for (; j + 16 <= K; j += 16) {
        uint8x16_t L_full = vld1q_u8(left  + lc);
        uint8x16_t R_full = vld1q_u8(right + rc);

        uint8_t m0 = bm[j >> 3];
        uint8x16_t both0 = vcombine_u8(vget_low_u8(L_full),
                                        vget_low_u8(R_full));
        uint8x8_t  shuf0 = vld1_u8(expand_tab[m0]);
        uint8x8_t  o0    = vqtbl1_u8(both0, shuf0);
        vst1_u8(out + j, o0);
        int nr0 = expand_popcnt[m0];

        uint8_t m1 = bm[(j >> 3) + 1];
        uint8x16x2_t src = {{ L_full, R_full }};
        uint8x8_t shuf1  = vld1_u8(expand_tab_pre[nr0][m1]);
        uint8x8_t o1     = vqtbl2_u8(src, shuf1);
        vst1_u8(out + j + 8, o1);
        int nr1 = expand_popcnt[m1];

        rc += nr0 + nr1;
        lc += (16 - nr0 - nr1);
    }
    for (; j + 8 <= K; j += 8) {
        uint8_t m  = bm[j >> 3];
        uint8x8_t  L    = vld1_u8(left + lc);
        uint8x8_t  R    = vld1_u8(right + rc);
        uint8x16_t both = vcombine_u8(L, R);
        uint8x8_t  shuf = vld1_u8(expand_tab[m]);
        uint8x8_t  o    = vqtbl1_u8(both, shuf);
        vst1_u8(out + j, o);
        int nr = expand_popcnt[m];
        rc += nr;
        lc += (8 - nr);
    }
    /* Scalar tail (1..7 leftover). */
    for (; j < K; j++) {
        int mb = (bm[j >> 3] >> (j & 7)) & 1;
        out[j] = mb ? right[rc++] : left[lc++];
    }
}
static void prim_merge_vv_com64(const ctx_t *c){
    prim_merge_vv_com64_neon(c->bm, c->n, c->merge_left, c->merge_right, c->out);
}

/* com64 — V5 COM cst_vec (broadcast L), prior production (A/B). */
static inline void prim_merge_cv_com64_neon(const uint8_t *bm, int K,
                                                uint8_t left_sym,
                                                const uint8_t *right,
                                                uint8_t *out)
{
    int rc = 0;
    int j = 0;
    uint8x8_t  Lbcast   = vdup_n_u8(left_sym);
    uint8x16_t Lbcast_q = vdupq_n_u8(left_sym);

    /* V5 main: 64 codes per iter, 4 independent chunks. */
    for (; j + 64 <= K; j += 64) {
        uint64_t mask_u64;
        memcpy(&mask_u64, bm + (j >> 3), 8);
        uint8x8_t bm_v = vcreate_u8(mask_u64);
        uint8x8_t pc_v = vcnt_u8(bm_v);
        uint64_t pc_u64 = vget_lane_u64(vreinterpret_u64_u8(pc_v), 0);
        uint64_t pfx = pc_u64 * 0x0101010101010101ULL;

        uint8_t cr0=0,             cr1=(uint8_t)(pfx>> 8);
        uint8_t cr2=(uint8_t)(pfx>>24), cr3=(uint8_t)(pfx>>40);
        uint8_t in0=(uint8_t)pc_u64,     in1=(uint8_t)(pc_u64>>16);
        uint8_t in2=(uint8_t)(pc_u64>>32), in3=(uint8_t)(pc_u64>>48);

        uint8_t m0=(uint8_t) mask_u64,        m1=(uint8_t)(mask_u64>> 8);
        uint8_t m2=(uint8_t)(mask_u64>>16),   m3=(uint8_t)(mask_u64>>24);
        uint8_t m4=(uint8_t)(mask_u64>>32),   m5=(uint8_t)(mask_u64>>40);
        uint8_t m6=(uint8_t)(mask_u64>>48),   m7=(uint8_t)(mask_u64>>56);

#define _V5_CHUNK_CV(idx, cr, in, ma, mb) do {                               \
            uint8x16_t R = vld1q_u8(right + rc + (cr));                      \
            uint8x16_t both = vcombine_u8(Lbcast, vget_low_u8(R));           \
            uint8x8_t  s0   = vld1_u8(expand_tab[ma]);                       \
            vst1_u8(out + j + (idx)*16,     vqtbl1_u8(both, s0));            \
            uint8x16x2_t src = {{ Lbcast_q, R }};                            \
            uint8x8_t s1 = vld1_u8(expand_tab_pre[in][mb]);                  \
            vst1_u8(out + j + (idx)*16 + 8, vqtbl2_u8(src, s1));             \
        } while (0)
        _V5_CHUNK_CV(0, cr0, in0, m0, m1);
        _V5_CHUNK_CV(1, cr1, in1, m2, m3);
        _V5_CHUNK_CV(2, cr2, in2, m4, m5);
        _V5_CHUNK_CV(3, cr3, in3, m6, m7);
#undef _V5_CHUNK_CV

        rc += (uint8_t)(pfx >> 56);
    }

    for (; j + 16 <= K; j += 16) {
        uint8x16_t R_full = vld1q_u8(right + rc);
        uint8_t m0 = bm[j >> 3];
        uint8x16_t both0 = vcombine_u8(Lbcast, vget_low_u8(R_full));
        uint8x8_t  shuf0 = vld1_u8(expand_tab[m0]);
        uint8x8_t  o0    = vqtbl1_u8(both0, shuf0);
        vst1_u8(out + j, o0);
        int nr0 = expand_popcnt[m0];

        uint8_t m1 = bm[(j >> 3) + 1];
        uint8x16x2_t src = {{ Lbcast_q, R_full }};
        uint8x8_t shuf1  = vld1_u8(expand_tab_pre[nr0][m1]);
        uint8x8_t o1     = vqtbl2_u8(src, shuf1);
        vst1_u8(out + j + 8, o1);

        rc += nr0 + expand_popcnt[m1];
    }
    for (; j + 8 <= K; j += 8) {
        uint8_t m = bm[j >> 3];
        uint8x8_t  R    = vld1_u8(right + rc);
        uint8x16_t both = vcombine_u8(Lbcast, R);
        uint8x8_t  shuf = vld1_u8(expand_tab[m]);
        uint8x8_t  o    = vqtbl1_u8(both, shuf);
        vst1_u8(out + j, o);
        rc += expand_popcnt[m];
    }
    for (; j < K; j++) {
        int mb = (bm[j >> 3] >> (j & 7)) & 1;
        out[j] = mb ? right[rc++] : left_sym;
    }
}
static void prim_merge_cv_com64(const ctx_t *c){ prim_merge_cv_com64_neon(c->bm, c->n, MERGE_LEFT_SYM, c->merge_right, c->out); }

/* ============================================================================
 * merge_vec_vec : asof-f9974f5 — pre-PR#22 COM64 main loop (prior production)
 *   The production merge before the PR #22 software pipelining: nothing is
 *   started for the next iteration, and the byte-popcounts cross to a GPR
 *   before the prefix-sum multiply.  Frozen here verbatim (PROF markers
 *   dropped per graveyard convention); reuses the production merge_neon_16B /
 *   merge_neon_8B_lo / g_merge_shuf0/1 / expand_popcnt, all unchanged by #22.
 * ========================================================================== */
static inline void prim_merge_vv_asof_f9974f5_neon(const uint8_t *bm, int K,
                                     const uint8_t *left,
                                     const uint8_t *right,
                                     uint8_t *out)
{
    const uint8_t *l_list = left, *r_list = right;
    intptr_t i = 0;
    for (; i + 64 <= K; i += 64) {
        uint64_t mask; memcpy(&mask, bm + (i >> 3), 8);
        uint8x8_t vmask = vcreate_u8(mask);
        uint8x8_t pop8  = vcnt_u8(vmask);
        /* Per-chunk cursor splits: move the 8 byte-popcounts to a GPR and
         * prefix-sum them with a single 64-bit multiply (offloads to the scalar
         * pipe; cheaper than the SIMD vpadd+vmul fold).  Byte k of the product
         * holds sum(pop8[0..k]), so bytes 1/3/5/7 are the 16-bit chunk
         * boundaries c0, c0+c1, c0+c1+c2, total. */
        uint64_t all_pop = vget_lane_u64(vreinterpret_u64_u8(pop8), 0);
        uint64_t pfx = all_pop * 0x0101010101010101ull;
        intptr_t pop0 = (pfx >> 8)  & 0xff;
        intptr_t pop1 = (pfx >> 24) & 0xff;
        intptr_t pop2 = (pfx >> 40) & 0xff;
        intptr_t pop3 =  pfx >> 56;
        merge_neon_16B(out + i,      l_list,             r_list,        mask,       g_merge_shuf0, g_merge_shuf1);
        merge_neon_16B(out + i + 16, l_list + 16 - pop0, r_list + pop0, mask >> 16, g_merge_shuf0, g_merge_shuf1);
        merge_neon_16B(out + i + 32, l_list + 32 - pop1, r_list + pop1, mask >> 32, g_merge_shuf0, g_merge_shuf1);
        merge_neon_16B(out + i + 48, l_list + 48 - pop2, r_list + pop2, mask >> 48, g_merge_shuf0, g_merge_shuf1);
        r_list += pop3; l_list += 64 - pop3;
    }
    int j = (int)i;

    /* Residue on the main tables: 16-wide, then 8-wide (low half). */
    for (; j + 16 <= K; j += 16) {
        uint16_t m16; memcpy(&m16, bm + (j >> 3), 2);
        merge_neon_16B(out + j, l_list, r_list, (intptr_t)m16,
                       g_merge_shuf0, g_merge_shuf1);
        /* expand_popcnt, not __builtin_popcount: aarch64 has no GPR
         * popcount (fmov+cnt+addv+fmov, ~7cy) and this sits on the
         * serial cursor chain between tail iterations. */
        int pop = expand_popcnt[m16 & 0xff] + expand_popcnt[m16 >> 8];
        r_list += pop; l_list += 16 - pop;
    }
    if (j + 8 <= K) {
        intptr_t m8 = bm[j >> 3];
        merge_neon_8B_lo(out + j, l_list, r_list, m8,
                         g_merge_shuf0, g_merge_shuf1);
        int pop = expand_popcnt[m8];
        r_list += pop; l_list += 8 - pop;
        j += 8;
    }
    /* Scalar tail (1..7 leftover). */
    int lc = (int)(l_list - left), rc = (int)(r_list - right);
    for (; j < K; j++) {
        int mb = (bm[j >> 3] >> (j & 7)) & 1;
        out[j] = mb ? right[rc++] : left[lc++];
    }
}
static void prim_merge_vv_asof_f9974f5(const ctx_t *c){
    prim_merge_vv_asof_f9974f5_neon(c->bm, c->n, c->merge_left, c->merge_right, c->out);
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

/* ============================================================================
 * merge_cst_vec : c3roll-asm-p0/p16 — c3/IVB loop-top phase pair (2026-07)
 *
 * The c3 (Ivy Bridge) "binary lottery": the production two-table pshufb
 * cst_vec loop (113 bytes) runs +11% slower when its loop top lands at
 * phase 16 mod 32 (32-byte uop-cache window split; phases {0,8,24} are
 * fast) — which phase a build gets is a link-layout roll (bench_c3jit.c
 * placement sweep, 2026-07-06).  These two cores are the verbatim
 * clang-20 -march=native c3 production roll with the loop top pinned to
 * phase 0 (fast) and phase 16 (slow), so any host can measure its
 * window-packing sensitivity.  VEX-encoded (needs AVX1 at runtime —
 * true of the whole SSE-tier fleet, c3..c6a); references the production
 * g_x86_merge_shuf0/1 tables via the same absolute-disp32 forms, so the
 * loop bytes match production exactly (byte-verified on c3).
 * Production-side cure candidate: -falign-loops=32 (fleet A/A' pending).
 * ========================================================================== */
__attribute__((naked,noinline,used))
static long pv_cv_ivb_p0_core(uint8_t *out, const uint8_t *right,
                              const uint8_t *bm, long n, long left_sym)
{
    __asm__ volatile(
        "pushq %rbx\n"
        "movq %rdx,%rax\n"          /* bm */
        "movq %rsi,%rdx\n"          /* right */
        "movq %rcx,%rsi\n"          /* n */
        "movq %rdi,%rcx\n"          /* out */
        "leaq 0x1(%rax),%r9\n"      /* bm cursor (m1 side) */
        "imull $0x01010101,%r8d,%r8d\n"
        "vmovd %r8d,%xmm1\n"
        "vpshufd $0x0,%xmm1,%xmm1\n" /* left_sym broadcast */
        "vpcmpeqd %xmm0,%xmm0,%xmm0\n"
        "xorl %edi,%edi\n"          /* rc */
        "xorl %r8d,%r8d\n"          /* j */
        ".p2align 5\n"              /* loop top at phase 0 mod 32 */
                                    /* NO .skip HERE !!! */
        "1:\n"                      /* ---- verbatim c3 clang-20 roll ---- */
        "movl %edi,%r10d\n"
        "movzbl -0x1(%r9),%r11d\n"
        "movzbl (%r9),%edi\n"
        "xorl %ebx,%ebx\n"
        "popcnt %r11d,%ebx\n"
        "shll $0x4,%r11d\n"
        "vmovq g_x86_merge_shuf1(,%rdi,8),%xmm2\n"
        "vpslldq $0x8,%xmm2,%xmm2\n"
        "vpaddb g_x86_merge_shuf0(%r11),%xmm2,%xmm2\n"
        "movq %r8,%r11\n"
        "movl %r10d,%r8d\n"
        "vmovdqu (%rdx,%r8,1),%xmm3\n"
        "vpshufb %xmm2,%xmm3,%xmm3\n"
        "vpxor %xmm0,%xmm2,%xmm2\n"
        "vpshufb %xmm2,%xmm1,%xmm2\n"
        "vpor %xmm3,%xmm2,%xmm2\n"
        "vmovdqu %xmm2,(%rcx,%r11,1)\n"
        "popcnt %edi,%edi\n"
        "addb %bl,%dil\n"
        "movzbl %dil,%edi\n"
        "addl %r10d,%edi\n"
        "leaq 0x10(%r11),%r8\n"
        "addq $0x2,%r9\n"
        "addq $0x20,%r11\n"
        "cmpq %rsi,%r11\n"
        "jbe 1b\n"
        "movl %edi,%eax\n"          /* return rc */
        "popq %rbx\n"
        "ret\n");
}
__attribute__((naked,noinline,used))
static long pv_cv_ivb_p16_core(uint8_t *out, const uint8_t *right,
                               const uint8_t *bm, long n, long left_sym)
{
    __asm__ volatile(
        "pushq %rbx\n"
        "movq %rdx,%rax\n"          /* bm */
        "movq %rsi,%rdx\n"          /* right */
        "movq %rcx,%rsi\n"          /* n */
        "movq %rdi,%rcx\n"          /* out */
        "leaq 0x1(%rax),%r9\n"      /* bm cursor (m1 side) */
        "imull $0x01010101,%r8d,%r8d\n"
        "vmovd %r8d,%xmm1\n"
        "vpshufd $0x0,%xmm1,%xmm1\n" /* left_sym broadcast */
        "vpcmpeqd %xmm0,%xmm0,%xmm0\n"
        "xorl %edi,%edi\n"          /* rc */
        "xorl %r8d,%r8d\n"          /* j */
        ".p2align 5\n"
        ".skip 16,0x90\n"           /* loop top at phase 16 mod 32 (the slow c3/IVB phase) */
        "1:\n"                      /* ---- same bytes as p0 ---- */
        "movl %edi,%r10d\n"
        "movzbl -0x1(%r9),%r11d\n"
        "movzbl (%r9),%edi\n"
        "xorl %ebx,%ebx\n"
        "popcnt %r11d,%ebx\n"
        "shll $0x4,%r11d\n"
        "vmovq g_x86_merge_shuf1(,%rdi,8),%xmm2\n"
        "vpslldq $0x8,%xmm2,%xmm2\n"
        "vpaddb g_x86_merge_shuf0(%r11),%xmm2,%xmm2\n"
        "movq %r8,%r11\n"
        "movl %r10d,%r8d\n"
        "vmovdqu (%rdx,%r8,1),%xmm3\n"
        "vpshufb %xmm2,%xmm3,%xmm3\n"
        "vpxor %xmm0,%xmm2,%xmm2\n"
        "vpshufb %xmm2,%xmm1,%xmm2\n"
        "vpor %xmm3,%xmm2,%xmm2\n"
        "vmovdqu %xmm2,(%rcx,%r11,1)\n"
        "popcnt %edi,%edi\n"
        "addb %bl,%dil\n"
        "movzbl %dil,%edi\n"
        "addl %r10d,%edi\n"
        "leaq 0x10(%r11),%r8\n"
        "addq $0x2,%r9\n"
        "addq $0x20,%r11\n"
        "cmpq %rsi,%r11\n"
        "jbe 1b\n"
        "movl %edi,%eax\n"          /* return rc */
        "popq %rbx\n"
        "ret\n");
}
static void pv_cv_phase(const uint8_t *bm, int K, uint8_t left_sym,
                        const uint8_t *right, uint8_t *out,
                        long (*core)(uint8_t*,const uint8_t*,const uint8_t*,long,long))
{
    int n16 = K >= 16 ? (K & ~15) : 0;   /* codes the asm loop consumes */
    long rc = 0;
    if (n16)
        rc = core(out, right, bm, (long)K, (long)left_sym);
    for (int j = n16; j < K; j++) {
        int b = (bm[j >> 3] >> (j & 7)) & 1;
        out[j] = b ? right[rc++] : left_sym;
    }
}
static void prim_merge_cv_ivb_p0 (const ctx_t *c){ pv_cv_phase(c->bm, c->n, MERGE_LEFT_SYM, c->merge_right, c->out, pv_cv_ivb_p0_core); }
static void prim_merge_cv_ivb_p16(const ctx_t *c){ pv_cv_phase(c->bm, c->n, MERGE_LEFT_SYM, c->merge_right, c->out, pv_cv_ivb_p16_core); }

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
/* unroll16 — the 2x-unrolled stride-16 expand_tab merge that was
 * production before the two-table merge landed (saved for A/B). */
static void prim_merge_vv_unroll16_x86(const uint8_t *bm, int K,
                                    const uint8_t *left,
                                    const uint8_t *right,
                                    uint8_t *out)
{
    int lc = 0, rc = 0;
    int j = 0;
    for (; j + 16 <= K; j += 16) {
        uint8_t m0 = bm[j >> 3];
        __m128i L0 = _mm_loadl_epi64((const __m128i *)(left + lc));
        __m128i R0 = _mm_loadl_epi64((const __m128i *)(right + rc));
        __m128i both0 = _mm_unpacklo_epi64(L0, R0);
        __m128i shuf0 = _mm_loadl_epi64((const __m128i *)expand_tab[m0]);
        __m128i o0    = _mm_shuffle_epi8(both0, shuf0);
        _mm_storel_epi64((__m128i *)(out + j), o0);
        int nr0 = expand_popcnt[m0];
        rc += nr0; lc += (8 - nr0);

        uint8_t m1 = bm[(j >> 3) + 1];
        __m128i L1 = _mm_loadl_epi64((const __m128i *)(left + lc));
        __m128i R1 = _mm_loadl_epi64((const __m128i *)(right + rc));
        __m128i both1 = _mm_unpacklo_epi64(L1, R1);
        __m128i shuf1 = _mm_loadl_epi64((const __m128i *)expand_tab[m1]);
        __m128i o1    = _mm_shuffle_epi8(both1, shuf1);
        _mm_storel_epi64((__m128i *)(out + j + 8), o1);
        int nr1 = expand_popcnt[m1];
        rc += nr1; lc += (8 - nr1);
    }
    for (; j + 8 <= K; j += 8) {
        uint8_t m = bm[j >> 3];
        __m128i L = _mm_loadl_epi64((const __m128i *)(left + lc));
        __m128i R = _mm_loadl_epi64((const __m128i *)(right + rc));
        __m128i both = _mm_unpacklo_epi64(L, R);
        __m128i shuf = _mm_loadl_epi64((const __m128i *)expand_tab[m]);
        __m128i o    = _mm_shuffle_epi8(both, shuf);
        _mm_storel_epi64((__m128i *)(out + j), o);
        int nr = expand_popcnt[m];
        rc += nr; lc += (8 - nr);
    }
    for (; j < K; j++) {
        int mb = (bm[j >> 3] >> (j & 7)) & 1;
        out[j] = mb ? right[rc++] : left[lc++];
    }
}
static void prim_merge_vv_unroll16(const ctx_t *c){
    prim_merge_vv_unroll16_x86(c->bm, c->n, c->merge_left, c->merge_right, c->out);
}

/* unroll16 — 2x-unrolled expand_tab cst_vec, prior production (A/B). */
static inline void prim_merge_cv_unroll16_x86(const uint8_t *bm, int K,
                                               uint8_t left_sym,
                                               const uint8_t *right,
                                               uint8_t *out)
{
    int rc = 0;
    int j = 0;
    __m128i Lbcast8 = _mm_set1_epi8((char)left_sym);
    for (; j + 16 <= K; j += 16) {
        uint8_t m0 = bm[j >> 3];
        __m128i R0 = _mm_loadl_epi64((const __m128i *)(right + rc));
        __m128i both0 = _mm_unpacklo_epi64(Lbcast8, R0);
        __m128i shuf0 = _mm_loadl_epi64((const __m128i *)expand_tab[m0]);
        __m128i o0    = _mm_shuffle_epi8(both0, shuf0);
        _mm_storel_epi64((__m128i *)(out + j), o0);
        rc += expand_popcnt[m0];

        uint8_t m1 = bm[(j >> 3) + 1];
        __m128i R1 = _mm_loadl_epi64((const __m128i *)(right + rc));
        __m128i both1 = _mm_unpacklo_epi64(Lbcast8, R1);
        __m128i shuf1 = _mm_loadl_epi64((const __m128i *)expand_tab[m1]);
        __m128i o1    = _mm_shuffle_epi8(both1, shuf1);
        _mm_storel_epi64((__m128i *)(out + j + 8), o1);
        rc += expand_popcnt[m1];
    }
    for (; j + 8 <= K; j += 8) {
        uint8_t m = bm[j >> 3];
        __m128i R = _mm_loadl_epi64((const __m128i *)(right + rc));
        __m128i both = _mm_unpacklo_epi64(Lbcast8, R);
        __m128i shuf = _mm_loadl_epi64((const __m128i *)expand_tab[m]);
        __m128i o    = _mm_shuffle_epi8(both, shuf);
        _mm_storel_epi64((__m128i *)(out + j), o);
        rc += expand_popcnt[m];
    }
    for (; j < K; j++) {
        int mb = (bm[j >> 3] >> (j & 7)) & 1;
        out[j] = mb ? right[rc++] : left_sym;
    }
}
static void prim_merge_cv_unroll16(const ctx_t *c){ prim_merge_cv_unroll16_x86(c->bm, c->n, MERGE_LEFT_SYM, c->merge_right, c->out); }


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

/* ============================================================================
 * merge_vec_vec / merge_cst_vec : asof-76dec21 -- the
 * zero-masked (maskz) expand forms, production until the issue-#11
 * merge-masking fix.  Zen 4/5 (c7a/c8a) have a false dependency on the maskz
 * destination that serializes the 64-wide loop at expand latency.
 * ========================================================================== */
static void pv_merge_vec_vec_avx512_maskz(const uint8_t *bm, int K,
                                       const uint8_t *left,
                                       const uint8_t *right,
                                       uint8_t *out)
{
    int lc = 0, rc = 0;
    int j = 0;
    for (; j + 64 <= K; j += 64) {
        uint64_t mask;
        memcpy(&mask, bm + (j >> 3), 8);
        __mmask64 m  = (__mmask64)mask;
        __mmask64 nm = ~m;
        __m512i L = _mm512_maskz_expandloadu_epi8(nm, left + lc);
        __m512i R = _mm512_maskz_expandloadu_epi8(m,  right + rc);
        __m512i o = _mm512_or_si512(L, R);
        _mm512_storeu_si512((__m512i *)(out + j), o);
        int nr = __builtin_popcountll(mask);
        rc += nr; lc += (64 - nr);
    }
    /* 2x-unrolled SSE stride-16: see primitives_x86.h. */
    for (; j + 16 <= K; j += 16) {
        uint8_t m0 = bm[j >> 3];
        __m128i L0 = _mm_loadl_epi64((const __m128i *)(left + lc));
        __m128i R0 = _mm_loadl_epi64((const __m128i *)(right + rc));
        __m128i both0 = _mm_unpacklo_epi64(L0, R0);
        __m128i shuf0 = _mm_loadl_epi64((const __m128i *)expand_tab[m0]);
        __m128i o0    = _mm_shuffle_epi8(both0, shuf0);
        _mm_storel_epi64((__m128i *)(out + j), o0);
        int nr0 = expand_popcnt[m0];
        rc += nr0; lc += (8 - nr0);

        uint8_t m1 = bm[(j >> 3) + 1];
        __m128i L1 = _mm_loadl_epi64((const __m128i *)(left + lc));
        __m128i R1 = _mm_loadl_epi64((const __m128i *)(right + rc));
        __m128i both1 = _mm_unpacklo_epi64(L1, R1);
        __m128i shuf1 = _mm_loadl_epi64((const __m128i *)expand_tab[m1]);
        __m128i o1    = _mm_shuffle_epi8(both1, shuf1);
        _mm_storel_epi64((__m128i *)(out + j + 8), o1);
        int nr1 = expand_popcnt[m1];
        rc += nr1; lc += (8 - nr1);
    }
    for (; j + 8 <= K; j += 8) {
        uint8_t m = bm[j >> 3];
        __m128i L = _mm_loadl_epi64((const __m128i *)(left + lc));
        __m128i R = _mm_loadl_epi64((const __m128i *)(right + rc));
        __m128i both = _mm_unpacklo_epi64(L, R);
        __m128i shuf = _mm_loadl_epi64((const __m128i *)expand_tab[m]);
        __m128i o    = _mm_shuffle_epi8(both, shuf);
        _mm_storel_epi64((__m128i *)(out + j), o);
        int nr = expand_popcnt[m];
        rc += nr; lc += (8 - nr);
    }
    for (; j < K; j++) {
        int mb = (bm[j >> 3] >> (j & 7)) & 1;
        out[j] = mb ? right[rc++] : left[lc++];
    }
}

static void pv_merge_cst_vec_avx512_maskz(const uint8_t *bm, int K,
                                                  uint8_t left_sym,
                                                  const uint8_t *right,
                                                  uint8_t *out)
{
    int rc = 0;
    int j = 0;
    __m128i Lbcast8 = _mm_set1_epi8((char)left_sym);
    __m512i Lbcast64 = _mm512_set1_epi8((char)left_sym);
    for (; j + 64 <= K; j += 64) {
        uint64_t mask;
        memcpy(&mask, bm + (j >> 3), 8);
        __mmask64 m  = (__mmask64)mask;
        __m512i R = _mm512_maskz_expandloadu_epi8(m, right + rc);
        __m512i o = _mm512_mask_mov_epi8(Lbcast64, m, R);
        _mm512_storeu_si512((__m512i *)(out + j), o);
        rc += __builtin_popcountll(mask);
    }
    for (; j + 16 <= K; j += 16) {
        uint8_t m0 = bm[j >> 3];
        __m128i R0 = _mm_loadl_epi64((const __m128i *)(right + rc));
        __m128i both0 = _mm_unpacklo_epi64(Lbcast8, R0);
        __m128i shuf0 = _mm_loadl_epi64((const __m128i *)expand_tab[m0]);
        __m128i o0    = _mm_shuffle_epi8(both0, shuf0);
        _mm_storel_epi64((__m128i *)(out + j), o0);
        rc += expand_popcnt[m0];

        uint8_t m1 = bm[(j >> 3) + 1];
        __m128i R1 = _mm_loadl_epi64((const __m128i *)(right + rc));
        __m128i both1 = _mm_unpacklo_epi64(Lbcast8, R1);
        __m128i shuf1 = _mm_loadl_epi64((const __m128i *)expand_tab[m1]);
        __m128i o1    = _mm_shuffle_epi8(both1, shuf1);
        _mm_storel_epi64((__m128i *)(out + j + 8), o1);
        rc += expand_popcnt[m1];
    }
    for (; j + 8 <= K; j += 8) {
        uint8_t m = bm[j >> 3];
        __m128i R = _mm_loadl_epi64((const __m128i *)(right + rc));
        __m128i both = _mm_unpacklo_epi64(Lbcast8, R);
        __m128i shuf = _mm_loadl_epi64((const __m128i *)expand_tab[m]);
        __m128i o    = _mm_shuffle_epi8(both, shuf);
        _mm_storel_epi64((__m128i *)(out + j), o);
        rc += expand_popcnt[m];
    }
    for (; j < K; j++) {
        int mb = (bm[j >> 3] >> (j & 7)) & 1;
        out[j] = mb ? right[rc++] : left_sym;
    }
}

static void prim_merge_vv_maskz(const ctx_t *c){ pv_merge_vec_vec_avx512_maskz(c->bm, c->n, c->merge_left, c->merge_right, c->out); }
/* ============================================================================
 * merge_cst_vec : iurii-asm-exp7b/exp8b — c8a/Zen 5 vpexpandb-encoding pair
 *   (2026-07-06 investigation)
 *
 * Two byte-exact re-creations of the production merge-masked cst_vec
 * inner loop that differ ONLY in the vpexpandb encoding: the expand's
 * base register chooses the 7-byte no-disp EVEX form (rcx) vs the
 * 8-byte disp8 form (r13, which architecturally requires the byte).
 * On Zen 5 (c8a) the no-disp form measures x1.15-1.21 SLOWER: the
 * disp8 changes op-cache entry packing -> dispatch-group formation ->
 * integer-scheduler steering (PMU: int-sched-0 token starvation,
 * +0.8c/iter).  A length-equalizing NOP does NOT recover it -- it is
 * the instruction's own form, not byte positions.  Which form the
 * compiler emits is a register-allocation roll: this pair pins both,
 * so any host can measure its sensitivity.  Expected ~0 on Intel.
 * ========================================================================== */
__attribute__((naked,noinline,used))
static long pv_cv_exp7b_core(uint8_t *out, const uint8_t *right,
                                  const uint8_t *bm, long n64, long left_sym)
{
    __asm__ volatile(
        "pushq %rbx\n"
        "pushq %r15\n"
        "movq %rdx,%r15\n"          /* bm  */
        "movq %rcx,%rdx\n"          /* n64 */
        "movq %rsi,%rcx\n"          /* right */
        "movq %rdi,%rbx\n"          /* out */
        "vpbroadcastb %r8d,%zmm1\n" /* left_sym */
        "xorl %eax,%eax\n"
        "xorl %esi,%esi\n"
        ".p2align 6\n"
        "1:\n"                      /* ---- verbatim no-disp form ---- */
        "movl %eax,%edi\n"
        "movslq %esi,%r11\n"
        "vmovdqa64 %zmm1,%zmm0\n"
        "sarl $0x3,%edi\n"
        "movslq %edi,%rdi\n"
        "movq (%r15,%rdi,1),%rdi\n"
        "kmovq %rdi,%k3\n"
        "vpexpandb (%rcx,%r11,1),%zmm0{%k3}\n"   /* 7B: no disp */
        "popcnt %rdi,%rdi\n"
        "addl %edi,%esi\n"
        "vmovdqu64 %zmm0,(%rbx,%rax,1)\n"
        "addq $0x40,%rax\n"
        "cmpq %rax,%rdx\n"
        "jne 1b\n"
        "movl %esi,%eax\n"          /* return rc */
        "popq %r15\n"
        "popq %rbx\n"
        "ret\n");
}
__attribute__((naked,noinline,used))
static long pv_cv_exp8b_core(uint8_t *out, const uint8_t *right,
                                 const uint8_t *bm, long n64, long left_sym)
{
    __asm__ volatile(
        "pushq %rbx\n"
        "pushq %r13\n"
        "movq %rdx,%r11\n"          /* bm  */
        "movq %rcx,%rdx\n"          /* n64 */
        "movq %rsi,%r13\n"          /* right */
        "movq %rdi,%rbx\n"          /* out */
        "vpbroadcastb %r8d,%zmm1\n"
        "xorl %eax,%eax\n"
        "xorl %ecx,%ecx\n"
        ".p2align 6\n"
        "1:\n"                      /* ---- verbatim disp8 form ---- */
        "movl %eax,%esi\n"
        "movslq %ecx,%r10\n"
        "vmovdqa64 %zmm1,%zmm0\n"
        "sarl $0x3,%esi\n"
        "movslq %esi,%rsi\n"
        "movq (%r11,%rsi,1),%rsi\n"
        "kmovq %rsi,%k3\n"
        "vpexpandb 0x0(%r13,%r10,1),%zmm0{%k3}\n" /* 8B: disp8 forced by r13 */
        "popcnt %rsi,%rsi\n"
        "addl %esi,%ecx\n"
        "vmovdqu64 %zmm0,(%rbx,%rax,1)\n"
        "addq $0x40,%rax\n"
        "cmpq %rax,%rdx\n"
        "jne 1b\n"
        "movl %ecx,%eax\n"          /* return rc */
        "popq %r13\n"
        "popq %rbx\n"
        "ret\n");
}
static void pv_merge_cst_vec_enc(const uint8_t *bm, int K, uint8_t left_sym,
                                 const uint8_t *right, uint8_t *out, int disp8)
{
    int n64 = K & ~63;
    long rc = 0;
    if (n64)
        rc = (disp8 ? pv_cv_exp8b_core : pv_cv_exp7b_core)
                 (out, right, bm, (long)n64, (long)left_sym);
    for (int j = n64; j < K; j++) {
        int b = (bm[j >> 3] >> (j & 7)) & 1;
        out[j] = b ? right[rc++] : left_sym;
    }
}
static void prim_merge_cv_exp7b(const ctx_t *c){ pv_merge_cst_vec_enc(c->bm, c->n, MERGE_LEFT_SYM, c->merge_right, c->out, 0); }
static void prim_merge_cv_exp8b (const ctx_t *c){ pv_merge_cst_vec_enc(c->bm, c->n, MERGE_LEFT_SYM, c->merge_right, c->out, 1); }

/* ============================================================================
 * merge_cst_vec : iurii-asm-gnr7b/gnr8b — c8i/GNR-tuned schedule pair (2026-07)
 *
 * gcc14 -march=native on Granite Rapids (a stray flag in test-c8i's
 * build cache) rolls a DIFFERENT schedule of the same issue-#11 loop:
 * the bm-word address chain issues first, the zmm copy + rc extend
 * are deferred, and the rc accumulate moves past the store.  On c8i/GNR
 * it beats the generic schedule (exp7b/exp8b) by ~6% at the same
 * vpexpandb encoding.  gnr7b is the verbatim byte-exact transcription
 * (7B no-disp expand off r9); gnr8b re-encodes only the expand to the
 * 8B disp8 form (r13 base) so Zen hosts can judge the schedule
 * without the known 7B-encoding handicap.
 * ========================================================================== */
__attribute__((naked,noinline,used))
static long pv_cv_gnr7b_core(uint8_t *out, const uint8_t *right,
                             const uint8_t *bm, long n64, long left_sym)
{
    __asm__ volatile(
        "movq %rsi,%r9\n"           /* right */
        "movq %rdi,%rsi\n"          /* out */
        "movq %rdx,%rdi\n"          /* bm */
        "movq %rcx,%r11\n"          /* n64 */
        "vpbroadcastb %r8d,%zmm1\n" /* left_sym */
        "xorl %eax,%eax\n"
        "xorl %edx,%edx\n"
        ".p2align 6\n"
        "1:\n"                      /* ---- verbatim c8i -march=native roll ---- */
        "movl %eax,%ecx\n"
        "sarl $0x3,%ecx\n"
        "movslq %ecx,%rcx\n"
        "movq (%rdi,%rcx,1),%rcx\n"
        "movslq %edx,%r8\n"
        "vmovdqa64 %zmm1,%zmm0\n"
        "kmovq %rcx,%k1\n"
        "vpexpandb (%r9,%r8,1),%zmm0{%k1}\n"   /* 7B: no disp */
        "popcnt %rcx,%rcx\n"
        "vmovdqu64 %zmm0,(%rsi,%rax,1)\n"
        "addq $0x40,%rax\n"
        "addl %ecx,%edx\n"
        "cmpq %rax,%r11\n"
        "jne 1b\n"
        "movl %edx,%eax\n"          /* return rc */
        "ret\n");
}
__attribute__((naked,noinline,used))
static long pv_cv_gnr8b_core(uint8_t *out, const uint8_t *right,
                             const uint8_t *bm, long n64, long left_sym)
{
    __asm__ volatile(
        "pushq %r13\n"
        "movq %rsi,%r13\n"          /* right */
        "movq %rdi,%rsi\n"          /* out */
        "movq %rdx,%rdi\n"          /* bm */
        "movq %rcx,%r11\n"          /* n64 */
        "vpbroadcastb %r8d,%zmm1\n" /* left_sym */
        "xorl %eax,%eax\n"
        "xorl %edx,%edx\n"
        ".p2align 6\n"
        "1:\n"                      /* ---- same schedule, disp8 expand ---- */
        "movl %eax,%ecx\n"
        "sarl $0x3,%ecx\n"
        "movslq %ecx,%rcx\n"
        "movq (%rdi,%rcx,1),%rcx\n"
        "movslq %edx,%r8\n"
        "vmovdqa64 %zmm1,%zmm0\n"
        "kmovq %rcx,%k1\n"
        "vpexpandb 0x0(%r13,%r8,1),%zmm0{%k1}\n" /* 8B: disp8 forced by r13 */
        "popcnt %rcx,%rcx\n"
        "vmovdqu64 %zmm0,(%rsi,%rax,1)\n"
        "addq $0x40,%rax\n"
        "addl %ecx,%edx\n"
        "cmpq %rax,%r11\n"
        "jne 1b\n"
        "movl %edx,%eax\n"          /* return rc */
        "popq %r13\n"
        "ret\n");
}
static void pv_cv_sched(const uint8_t *bm, int K, uint8_t left_sym,
                        const uint8_t *right, uint8_t *out,
                        long (*core)(uint8_t*,const uint8_t*,const uint8_t*,long,long))
{
    int n64 = K & ~63;
    long rc = 0;
    if (n64)
        rc = core(out, right, bm, (long)n64, (long)left_sym);
    for (int j = n64; j < K; j++) {
        int b = (bm[j >> 3] >> (j & 7)) & 1;
        out[j] = b ? right[rc++] : left_sym;
    }
}
static void prim_merge_cv_gnr7b(const ctx_t *c){ pv_cv_sched(c->bm, c->n, MERGE_LEFT_SYM, c->merge_right, c->out, pv_cv_gnr7b_core); }
static void prim_merge_cv_gnr8b(const ctx_t *c){ pv_cv_sched(c->bm, c->n, MERGE_LEFT_SYM, c->merge_right, c->out, pv_cv_gnr8b_core); }

/* ============================================================================
 * merge_cst_vec / merge_vec_vec : iurii2-c-* — issue-#11 follow-ups
 *   (Iurii, 2026-07)
 *
 * Full matrix benched 2026-07-07 on c6i/c7i/c8i/c7a/c8a; only the
 * information-carrying survivors are kept below.  Dropped (results in
 * the session log / #11): plain size_t (subsumed by vrb), my prefix-
 * cursor u2/u4/u2szt reconstructions (dominated by vrb for cst_vec;
 * for vec_vec only u4szt survives as the c8a/Zen 5 winner), and the
 * bumped-pointer forms (indistinguishable from vrb everywhere).
 *
 * Findings: (a) cst_vec is throughput-bound -> Iurii's SERIAL cursor
 * updates + separate bm cursor + ssize_t (vrb) win everywhere
 * (c7a/Zen 4 -16..19%, c8a/Zen 5 -27%, c8i/GNR -13%, c6i/ICL + c7i/SPR ~0) and beat every
 * pinned asm core; (b) vec_vec chains two expands -> latency-bound,
 * serial cursors collapse on c8a/Zen 5 (-1%) and PREFIX cursors (u4szt)
 * win there (-25%); (c) kld (kmovq mask from memory + popcnt from
 * memory) is the c8i/GNR vec_vec champion (-17%) and a c8a/Zen 5 cst_vec tie,
 * but consistently hurts c7a/Zen 4 (memory-form kmovq) and c6i/ICL.
 * Cross-checked: Zen 4/5 splits survive binary-swap between c7a/c8a.
 * ========================================================================== */
/* iurii2-c-vrb -- Iurii's verbatim follow-up from the #11 comment
 * (2026-07-06): ssize_t cursors, separate bm cursor k (no j>>3 shift in
 * the loop), 4x unroll with SERIAL rc updates -- rc += popcount after
 * each expand, unlike the prefix-cursor u4 above.  His Zen 5-mobile
 * numbers: 0.0093 -> 0.0080 ns/byte (clang 22). */
static void pv_merge_cv_vrb(const uint8_t *bm, int K_, uint8_t left_sym,
                            const uint8_t *right, uint8_t *out)
{
    ssize_t K = K_, rc = 0, j = 0, k = 0;
    __m512i Lbcast64 = _mm512_set1_epi8((char)left_sym);
    for (; j <= K - 256; j += 256, k += 32) {
        uint64_t mask1;
        memcpy(&mask1, bm + k, 8);
        __m512i o1 = _mm512_mask_expandloadu_epi8(Lbcast64, (__mmask64)mask1, right + rc);
        _mm512_storeu_si512((__m512i *)(out + j), o1);
        rc += __builtin_popcountll(mask1);

        uint64_t mask2;
        memcpy(&mask2, bm + k + 8, 8);
        __m512i o2 = _mm512_mask_expandloadu_epi8(Lbcast64, (__mmask64)mask2, right + rc);
        _mm512_storeu_si512((__m512i *)(out + j + 64), o2);
        rc += __builtin_popcountll(mask2);

        uint64_t mask3;
        memcpy(&mask3, bm + k + 16, 8);
        __m512i o3 = _mm512_mask_expandloadu_epi8(Lbcast64, (__mmask64)mask3, right + rc);
        _mm512_storeu_si512((__m512i *)(out + j + 128), o3);
        rc += __builtin_popcountll(mask3);

        uint64_t mask4;
        memcpy(&mask4, bm + k + 24, 8);
        __m512i o4 = _mm512_mask_expandloadu_epi8(Lbcast64, (__mmask64)mask4, right + rc);
        _mm512_storeu_si512((__m512i *)(out + j + 192), o4);
        rc += __builtin_popcountll(mask4);
    }
    for (; j + 64 <= K; j += 64, k += 8) {
        uint64_t mask; memcpy(&mask, bm + k, 8);
        __m512i o = _mm512_mask_expandloadu_epi8(Lbcast64, (__mmask64)mask, right + rc);
        _mm512_storeu_si512((__m512i *)(out + j), o);
        rc += __builtin_popcountll(mask);
    }
    for (; j < K; j++) {
        int b = (bm[j >> 3] >> (j & 7)) & 1;
        out[j] = b ? right[rc++] : left_sym;
    }
}
/* iurii2-c-kld -- vrb + Iurii's third suggestion: load the mask register
 * DIRECTLY from memory (kmovq mem-form, executes in the load pipes) and
 * re-load the same 8 bytes as popcnt's memory operand (micro-fused).
 * Duplicates the L1-hit load but drops the GPR->k transfer uop and
 * moves mask delivery off the crowded int schedulers.  The pointer
 * barrier keeps the compiler from CSEing the two loads back together. */
#define PV_KLD_BLOCK(OFF) do {                                              \
        const uint8_t *pk_ = bm + k + (OFF);                              \
        __mmask64 mk_ = _load_mask64((const __mmask64 *)pk_);               \
        __m512i o_ = _mm512_mask_expandloadu_epi8(Lb, mk_, right + rc);     \
        _mm512_storeu_si512((__m512i *)(out + j + (OFF)*8), o_);            \
        const uint8_t *pv_ = pk_; asm("" : "+r"(pv_));                      \
        uint64_t v_; memcpy(&v_, pv_, 8);                                   \
        rc += __builtin_popcountll(v_);                                     \
    } while (0)
static void pv_merge_cv_kld(const uint8_t *bm, int K_, uint8_t left_sym,
                            const uint8_t *right, uint8_t *out)
{
    ssize_t K = K_, rc = 0, j = 0, k = 0;
    __m512i Lb = _mm512_set1_epi8((char)left_sym);
    for (; j <= K - 256; j += 256, k += 32) {
        PV_KLD_BLOCK(0);
        PV_KLD_BLOCK(8);
        PV_KLD_BLOCK(16);
        PV_KLD_BLOCK(24);
    }
    for (; j + 64 <= K; j += 64, k += 8) {
        PV_KLD_BLOCK(0);
    }
    for (; j < K; j++) {
        int b = (bm[j >> 3] >> (j & 7)) & 1;
        out[j] = b ? right[rc++] : left_sym;
    }
}
#undef PV_KLD_BLOCK
/* iurii2-c-ptr -- same structure with j/k/rc as bumped POINTERS instead
 * of offsets (base+index addressing gone; the expand base is the moving
 * right pointer, which also re-rolls its encoding class per build). */
static void prim_merge_cv_c_kld  (const ctx_t *c){ pv_merge_cv_kld  (c->bm, c->n, MERGE_LEFT_SYM, c->merge_right, c->out); }
static void prim_merge_cv_c_vrb  (const ctx_t *c){ pv_merge_cv_vrb  (c->bm, c->n, MERGE_LEFT_SYM, c->merge_right, c->out); }

/* vec_vec versions of the same transforms (two-sided cursors; the
 * merge-masked zero + asm barrier follows production merge_vec_vec_avx512). */
static void pv_merge_vv_u4szt(const uint8_t *bm, int K, const uint8_t *left,
                              const uint8_t *right, uint8_t *out)
{
    size_t lc = 0, rc = 0, j = 0, n = (size_t)K;
    for (; j + 256 <= n; j += 256) {
        uint64_t m0, m1, m2, m3;
        memcpy(&m0, bm + (j >> 3),      8);
        memcpy(&m1, bm + (j >> 3) + 8,  8);
        memcpy(&m2, bm + (j >> 3) + 16, 8);
        memcpy(&m3, bm + (j >> 3) + 24, 8);
        size_t c1 = (size_t)__builtin_popcountll(m0);
        size_t c2 = c1 + (size_t)__builtin_popcountll(m1);
        size_t c3 = c2 + (size_t)__builtin_popcountll(m2);
        __m512i zero = _mm512_setzero_si512(); asm("":"+v"(zero));
        __m512i L0 = _mm512_mask_expandloadu_epi8(zero, ~(__mmask64)m0, left + lc);
        __m512i o0 = _mm512_mask_expandloadu_epi8(L0, (__mmask64)m0, right + rc);
        __m512i L1 = _mm512_mask_expandloadu_epi8(zero, ~(__mmask64)m1, left + lc + (64 - c1));
        __m512i o1 = _mm512_mask_expandloadu_epi8(L1, (__mmask64)m1, right + rc + c1);
        __m512i L2 = _mm512_mask_expandloadu_epi8(zero, ~(__mmask64)m2, left + lc + (128 - c2));
        __m512i o2 = _mm512_mask_expandloadu_epi8(L2, (__mmask64)m2, right + rc + c2);
        __m512i L3 = _mm512_mask_expandloadu_epi8(zero, ~(__mmask64)m3, left + lc + (192 - c3));
        __m512i o3 = _mm512_mask_expandloadu_epi8(L3, (__mmask64)m3, right + rc + c3);
        _mm512_storeu_si512((__m512i *)(out + j),       o0);
        _mm512_storeu_si512((__m512i *)(out + j + 64),  o1);
        _mm512_storeu_si512((__m512i *)(out + j + 128), o2);
        _mm512_storeu_si512((__m512i *)(out + j + 192), o3);
        size_t nr = c3 + (size_t)__builtin_popcountll(m3);
        rc += nr; lc += 256 - nr;
    }
    for (; j + 64 <= n; j += 64) {
        uint64_t mask; memcpy(&mask, bm + (j >> 3), 8);
        __mmask64 m = (__mmask64)mask;
        __m512i zero = _mm512_setzero_si512(); asm("":"+v"(zero));
        __m512i L = _mm512_mask_expandloadu_epi8(zero, ~m, left + lc);
        __m512i o = _mm512_mask_expandloadu_epi8(L, m, right + rc);
        _mm512_storeu_si512((__m512i *)(out + j), o);
        size_t nr = (size_t)__builtin_popcountll(mask);
        rc += nr; lc += 64 - nr;
    }
    for (; j < n; j++) {
        int b = (bm[j >> 3] >> (j & 7)) & 1;
        out[j] = b ? right[rc++] : left[lc++];
    }
}
/* vec_vec versions of Iurii's follow-up ideas: vrb structure (ssize_t,
 * separate bm cursor, 4x unroll, SERIAL lc/rc updates), bumped-pointer
 * form, and kld (kmovq mask from memory + knot for the L side + popcnt
 * from memory). */
#define PV_VV_VRB_BLOCK(BOFF) do {                                          \
        uint64_t m_; memcpy(&m_, bm + k + (BOFF), 8);                       \
        __m512i zero_ = _mm512_setzero_si512(); asm("":"+v"(zero_));        \
        __m512i L_ = _mm512_mask_expandloadu_epi8(zero_, ~(__mmask64)m_, left + lc); \
        __m512i o_ = _mm512_mask_expandloadu_epi8(L_, (__mmask64)m_, right + rc); \
        _mm512_storeu_si512((__m512i *)(out + j + (BOFF)*8), o_);           \
        ssize_t nr_ = __builtin_popcountll(m_);                             \
        rc += nr_; lc += 64 - nr_;                                          \
    } while (0)
static void pv_merge_vv_vrb(const uint8_t *bm, int K_, const uint8_t *left,
                            const uint8_t *right, uint8_t *out)
{
    ssize_t K = K_, lc = 0, rc = 0, j = 0, k = 0;
    for (; j <= K - 256; j += 256, k += 32) {
        PV_VV_VRB_BLOCK(0);
        PV_VV_VRB_BLOCK(8);
        PV_VV_VRB_BLOCK(16);
        PV_VV_VRB_BLOCK(24);
    }
    for (; j + 64 <= K; j += 64, k += 8) {
        PV_VV_VRB_BLOCK(0);
    }
    for (; j < K; j++) {
        int b = (bm[j >> 3] >> (j & 7)) & 1;
        out[j] = b ? right[rc++] : left[lc++];
    }
}
#undef PV_VV_VRB_BLOCK
#define PV_VV_KLD_BLOCK(BOFF) do {                                          \
        const uint8_t *pk_ = bm + k + (BOFF);                               \
        __mmask64 mk_ = _load_mask64((const __mmask64 *)pk_);               \
        __mmask64 nk_ = _knot_mask64(mk_);                                  \
        __m512i zero_ = _mm512_setzero_si512(); asm("":"+v"(zero_));        \
        __m512i L_ = _mm512_mask_expandloadu_epi8(zero_, nk_, left + lc);   \
        __m512i o_ = _mm512_mask_expandloadu_epi8(L_, mk_, right + rc);     \
        _mm512_storeu_si512((__m512i *)(out + j + (BOFF)*8), o_);           \
        const uint8_t *pv_ = pk_; asm("" : "+r"(pv_));                      \
        uint64_t v_; memcpy(&v_, pv_, 8);                                   \
        ssize_t nr_ = __builtin_popcountll(v_);                             \
        rc += nr_; lc += 64 - nr_;                                          \
    } while (0)
static void pv_merge_vv_kld(const uint8_t *bm, int K_, const uint8_t *left,
                            const uint8_t *right, uint8_t *out)
{
    ssize_t K = K_, lc = 0, rc = 0, j = 0, k = 0;
    for (; j <= K - 256; j += 256, k += 32) {
        PV_VV_KLD_BLOCK(0);
        PV_VV_KLD_BLOCK(8);
        PV_VV_KLD_BLOCK(16);
        PV_VV_KLD_BLOCK(24);
    }
    for (; j + 64 <= K; j += 64, k += 8) {
        PV_VV_KLD_BLOCK(0);
    }
    for (; j < K; j++) {
        int b = (bm[j >> 3] >> (j & 7)) & 1;
        out[j] = b ? right[rc++] : left[lc++];
    }
}
#undef PV_VV_KLD_BLOCK
static void prim_merge_vv_c_vrb  (const ctx_t *c){ pv_merge_vv_vrb  (c->bm, c->n, c->merge_left, c->merge_right, c->out); }
static void prim_merge_vv_c_kld  (const ctx_t *c){ pv_merge_vv_kld  (c->bm, c->n, c->merge_left, c->merge_right, c->out); }
static void prim_merge_vv_c_u4szt(const ctx_t *c){ pv_merge_vv_u4szt(c->bm, c->n, c->merge_left, c->merge_right, c->out); }

/* ============================================================================
 * merge_vec_vec : iurii-asm-l7r7/l7r8/l8r7/l8r8 — vec_vec vpexpandb encoding
 *   matrix (L-expand x R-expand, 7B no-disp vs 8B disp8 forms).
 * ========================================================================== */
__attribute__((naked,noinline,used))
static long pv_vv64_l7r7(uint8_t *out, const uint8_t *left,
                           const uint8_t *right, const uint8_t *bm, long n64)
{
    __asm__ volatile(
        "pushq %rbx\n"
        "pushq %r15\n"
        "pushq %r14\n"
        "movq %rdi,%rbx\n"
        "movq %rsi,%r15\n"
        "movq %rdx,%r10\n"
        "movq %rcx,%r11\n"
        "movq %r8,%rdi\n"
        "movl $0x40,%r8d\n"
        "vpxor %xmm1,%xmm1,%xmm1\n"
        "xorl %esi,%esi\n"
        "xorl %eax,%eax\n"
        "xorl %edx,%edx\n"
        ".p2align 4\n"
        "1:\n"
        "movl %esi,%ecx\n"
        "movslq %edx,%r14\n"
        "vmovdqa64 %zmm1,%zmm0\n"
        "sarl $0x3,%ecx\n"
        "movslq %ecx,%rcx\n"
        "movq (%r11,%rcx,1),%rcx\n"
        "kmovq %rcx,%k5\n"
        "knotq %k5,%k1\n"
        "vpexpandb (%r15,%r14,1),%zmm0{%k1}\n"
        "movslq %eax,%r14\n"
        "vpexpandb (%r10,%r14,1),%zmm0{%k5}\n"
        "movl %r8d,%r14d\n"
        "popcnt %rcx,%rcx\n"
        "addl %ecx,%eax\n"
        "subl %ecx,%r14d\n"
        "vmovdqu64 %zmm0,(%rbx,%rsi,1)\n"
        "addq $0x40,%rsi\n"
        "addl %r14d,%edx\n"
        "cmpq %rsi,%rdi\n"
        "jne 1b\n"
        "popq %r14\n"
        "popq %r15\n"
        "popq %rbx\n"
        "ret\n");
}
__attribute__((naked,noinline,used))
static long pv_vv64_l7r8(uint8_t *out, const uint8_t *left,
                           const uint8_t *right, const uint8_t *bm, long n64)
{
    __asm__ volatile(
        "pushq %rbx\n"
        "pushq %rbp\n"
        "pushq %r15\n"
        "pushq %r14\n"
        "movq %rdi,%rbx\n"
        "movq %rsi,%r15\n"
        "movq %rdx,%rbp\n"
        "movq %rcx,%r11\n"
        "movq %r8,%rdi\n"
        "movl $0x40,%r8d\n"
        "vpxor %xmm1,%xmm1,%xmm1\n"
        "xorl %esi,%esi\n"
        "xorl %eax,%eax\n"
        "xorl %edx,%edx\n"
        ".p2align 4\n"
        "1:\n"
        "movl %esi,%ecx\n"
        "movslq %edx,%r14\n"
        "vmovdqa64 %zmm1,%zmm0\n"
        "sarl $0x3,%ecx\n"
        "movslq %ecx,%rcx\n"
        "movq (%r11,%rcx,1),%rcx\n"
        "kmovq %rcx,%k5\n"
        "knotq %k5,%k1\n"
        "vpexpandb (%r15,%r14,1),%zmm0{%k1}\n"
        "movslq %eax,%r14\n"
        "vpexpandb 0x0(%rbp,%r14,1),%zmm0{%k5}\n"
        "movl %r8d,%r14d\n"
        "popcnt %rcx,%rcx\n"
        "addl %ecx,%eax\n"
        "subl %ecx,%r14d\n"
        "vmovdqu64 %zmm0,(%rbx,%rsi,1)\n"
        "addq $0x40,%rsi\n"
        "addl %r14d,%edx\n"
        "cmpq %rsi,%rdi\n"
        "jne 1b\n"
        "popq %r14\n"
        "popq %r15\n"
        "popq %rbp\n"
        "popq %rbx\n"
        "ret\n");
}
__attribute__((naked,noinline,used))
static long pv_vv64_l8r7(uint8_t *out, const uint8_t *left,
                           const uint8_t *right, const uint8_t *bm, long n64)
{
    __asm__ volatile(
        "pushq %rbx\n"
        "pushq %r13\n"
        "pushq %r14\n"
        "movq %rdi,%rbx\n"
        "movq %rsi,%r13\n"
        "movq %rdx,%r10\n"
        "movq %rcx,%r11\n"
        "movq %r8,%rdi\n"
        "movl $0x40,%r8d\n"
        "vpxor %xmm1,%xmm1,%xmm1\n"
        "xorl %esi,%esi\n"
        "xorl %eax,%eax\n"
        "xorl %edx,%edx\n"
        ".p2align 4\n"
        "1:\n"
        "movl %esi,%ecx\n"
        "movslq %edx,%r14\n"
        "vmovdqa64 %zmm1,%zmm0\n"
        "sarl $0x3,%ecx\n"
        "movslq %ecx,%rcx\n"
        "movq (%r11,%rcx,1),%rcx\n"
        "kmovq %rcx,%k5\n"
        "knotq %k5,%k1\n"
        "vpexpandb 0x0(%r13,%r14,1),%zmm0{%k1}\n"
        "movslq %eax,%r14\n"
        "vpexpandb (%r10,%r14,1),%zmm0{%k5}\n"
        "movl %r8d,%r14d\n"
        "popcnt %rcx,%rcx\n"
        "addl %ecx,%eax\n"
        "subl %ecx,%r14d\n"
        "vmovdqu64 %zmm0,(%rbx,%rsi,1)\n"
        "addq $0x40,%rsi\n"
        "addl %r14d,%edx\n"
        "cmpq %rsi,%rdi\n"
        "jne 1b\n"
        "popq %r14\n"
        "popq %r13\n"
        "popq %rbx\n"
        "ret\n");
}
__attribute__((naked,noinline,used))
static long pv_vv64_l8r8(uint8_t *out, const uint8_t *left,
                           const uint8_t *right, const uint8_t *bm, long n64)
{
    __asm__ volatile(
        "pushq %rbx\n"
        "pushq %rbp\n"
        "pushq %r13\n"
        "pushq %r14\n"
        "movq %rdi,%rbx\n"
        "movq %rsi,%r13\n"
        "movq %rdx,%rbp\n"
        "movq %rcx,%r11\n"
        "movq %r8,%rdi\n"
        "movl $0x40,%r8d\n"
        "vpxor %xmm1,%xmm1,%xmm1\n"
        "xorl %esi,%esi\n"
        "xorl %eax,%eax\n"
        "xorl %edx,%edx\n"
        ".p2align 4\n"
        "1:\n"
        "movl %esi,%ecx\n"
        "movslq %edx,%r14\n"
        "vmovdqa64 %zmm1,%zmm0\n"
        "sarl $0x3,%ecx\n"
        "movslq %ecx,%rcx\n"
        "movq (%r11,%rcx,1),%rcx\n"
        "kmovq %rcx,%k5\n"
        "knotq %k5,%k1\n"
        "vpexpandb 0x0(%r13,%r14,1),%zmm0{%k1}\n"
        "movslq %eax,%r14\n"
        "vpexpandb 0x0(%rbp,%r14,1),%zmm0{%k5}\n"
        "movl %r8d,%r14d\n"
        "popcnt %rcx,%rcx\n"
        "addl %ecx,%eax\n"
        "subl %ecx,%r14d\n"
        "vmovdqu64 %zmm0,(%rbx,%rsi,1)\n"
        "addq $0x40,%rsi\n"
        "addl %r14d,%edx\n"
        "cmpq %rsi,%rdi\n"
        "jne 1b\n"
        "popq %r14\n"
        "popq %r13\n"
        "popq %rbp\n"
        "popq %rbx\n"
        "ret\n");
}
static void pv_vv_matrix(const uint8_t *bm, int K, const uint8_t *left,
                         const uint8_t *right, uint8_t *out,
                         long (*core)(uint8_t*,const uint8_t*,const uint8_t*,const uint8_t*,long))
{
    int n64 = K & ~63;
    int rc = 0, lc = 0;
    if (n64) { rc = (int)core(out, left, right, bm, (long)n64); lc = n64 - rc; }
    for (int j = n64; j < K; j++) {
        int mb = (bm[j >> 3] >> (j & 7)) & 1;
        out[j] = mb ? right[rc++] : left[lc++];
    }
}
static void prim_merge_vv_l7r7(const ctx_t *c){ pv_vv_matrix(c->bm, c->n, c->merge_left, c->merge_right, c->out, pv_vv64_l7r7); }
static void prim_merge_vv_l7r8(const ctx_t *c){ pv_vv_matrix(c->bm, c->n, c->merge_left, c->merge_right, c->out, pv_vv64_l7r8); }
static void prim_merge_vv_l8r7(const ctx_t *c){ pv_vv_matrix(c->bm, c->n, c->merge_left, c->merge_right, c->out, pv_vv64_l8r7); }
static void prim_merge_vv_l8r8(const ctx_t *c){ pv_vv_matrix(c->bm, c->n, c->merge_left, c->merge_right, c->out, pv_vv64_l8r8); }

static void prim_merge_cv_maskz(const ctx_t *c){ pv_merge_cst_vec_avx512_maskz(c->bm, c->n, MERGE_LEFT_SYM, c->merge_right, c->out); }


#endif /* __AVX512VBMI2__ && __AVX512VBMI__ */

#if defined(USE_NEON_KERNELS)
/* skewfuse — fused per-word fast paths (no pre-pass, no lists): the production
 * linear loop grows two per-word fast paths, mask==0 (all-left: 64 B constant
 * fill, r_list unchanged) and mask==~0 (all-right: 64 B straight copy).  The
 * worst-case tax is one predictable branch pair per word; the risk is
 * mispredicts at run boundaries on run-structured bitmaps. */
static void prim_merge_cst_vec_skewfuse(const ctx_t *c)
{
    const uint8_t *bm = c->bm, *right = c->merge_right;
    uint8_t *out = c->out; int K = c->n;
    const uint8_t left_sym = MERGE_LEFT_SYM;
    uint8x16_t Lb = vdupq_n_u8(left_sym);
    const uint8_t *r_list = right;
    intptr_t i = 0;
    for (; i + 64 <= K; i += 64) {
        uint64_t mask; memcpy(&mask, bm + (i >> 3), 8);
        if (mask == 0) {                       /* all-left: constant fill */
            vst1q_u8(out + i,      Lb); vst1q_u8(out + i + 16, Lb);
            vst1q_u8(out + i + 32, Lb); vst1q_u8(out + i + 48, Lb);
            continue;
        }
        if (mask == ~0ull) {                   /* all-right: straight copy */
            vst1q_u8(out + i,      vld1q_u8(r_list));
            vst1q_u8(out + i + 16, vld1q_u8(r_list + 16));
            vst1q_u8(out + i + 32, vld1q_u8(r_list + 32));
            vst1q_u8(out + i + 48, vld1q_u8(r_list + 48));
            r_list += 64;
            continue;
        }
        uint8x8_t pop8 = vcnt_u8(vcreate_u8(mask));
        uint64_t pfx = vget_lane_u64(vreinterpret_u64_u8(pop8), 0) * 0x0101010101010101ull;
        intptr_t p0 = (pfx >> 8) & 0xff, p1 = (pfx >> 24) & 0xff, p2 = (pfx >> 40) & 0xff, p3 = pfx >> 56;
#define _MCVF(off, rd, mk) do {                                                  \
        int8x16_t s0 = vld1q_s8(&g_merge_shuf0[(((intptr_t)(mk)) << 4) & 0xff0]); \
        int8x16_t s1 = vld1q_s8(&g_merge_shuf1[(((intptr_t)(mk)) >> 4) & 0xff0]); \
        uint8x16_t sh = vreinterpretq_u8_s8(vabdq_s8(s0, s1));                    \
        uint8x16x2_t src; src.val[0] = vld1q_u8(rd); src.val[1] = Lb;            \
        vst1q_u8(out + i + (off), vqtbl2q_u8(src, sh));                          \
    } while (0)
        _MCVF(0,  r_list,      mask);
        _MCVF(16, r_list + p0, mask >> 16);
        _MCVF(32, r_list + p1, mask >> 32);
        _MCVF(48, r_list + p2, mask >> 48);
#undef _MCVF
        r_list += p3;
    }
    int j = (int)i;
    for (; j + 16 <= K; j += 16) {
        uint16_t m16; memcpy(&m16, bm + (j >> 3), 2);
        int8x16_t s0 = vld1q_s8(&g_merge_shuf0[((intptr_t)m16 << 4) & 0xff0]);
        int8x16_t s1 = vld1q_s8(&g_merge_shuf1[((intptr_t)m16 >> 4) & 0xff0]);
        uint8x16_t sh = vreinterpretq_u8_s8(vabdq_s8(s0, s1));
        uint8x16x2_t src; src.val[0] = vld1q_u8(r_list); src.val[1] = Lb;
        vst1q_u8(out + j, vqtbl2q_u8(src, sh));
        r_list += __builtin_popcount(m16);
    }
    int rc = (int)(r_list - right);
    for (; j < K; j++) { int mb = (bm[j >> 3] >> (j & 7)) & 1; out[j] = mb ? right[rc++] : left_sym; }
}
#endif /* USE_NEON_KERNELS */

/* ============================================================================
 * Registry — merge family (no-op where the ISA is unavailable)
 * ========================================================================== */

/* ============================================================================
 * merge_flat : avx2-ymm — PROMOTED to production (merge_flat_d{5,6,7}_ymm_x86
 *   in pivco_huffman_primitives_x86.h, 2026-07-16); this row calls the
 *   production ymm kernels directly (ymm blocks + scalar tail only, no
 *   128-bit mid-range) so it stays the isolated ymm measurement.
 * ========================================================================== */
#if defined(__AVX2__) && !defined(__AVX512VBMI2__)
static void prim_merge_flat_avx2ymm(const ctx_t *c) {
    int i = 0;
    switch (c->D) {
    case 5: i = merge_flat_d5_ymm_x86(c->out, c->n, c->bm, c->c2s); break;
    case 6: i = merge_flat_d6_ymm_x86(c->out, c->n, c->bm, c->c2s); break;
    case 7: i = merge_flat_d7_ymm_x86(c->out, c->n, c->bm, c->c2s); break;
    default: break;
    }
    merge_flat_tail_x86(c->out, i, c->n, c->bm, c->D, c->c2s);
}
#endif /* __AVX2__ && !__AVX512VBMI2__ */

static void pv_register_merge(void) {
#if defined(USE_NEON_KERNELS)
    pv_build_cc_tables();
#endif
    /* merge_flat — AVX-512 (asof-e5a199a, per-D) */
    for (int d = 2; d <= 7; d++)
        PV_VARIANT_D(ST_MERGE_FLAT, "asof-e5a199a", d, PV_ISA_AVX512,
                     "e5a199a~1 merge_flat_dN_avx512",
                     "pre-widening 16 codes/iter", 0, PV_FN_VBMI2(prim_merge_flat_e5a199a));
    /* merge_vec_vec / cst_vec — AVX-512 (asof-76dec21, pre issue-#11) */
    PV_VARIANT(ST_MERGE_VEC_VEC, "asof-76dec21", PV_ISA_AVX512, "76dec21 (prior production)",
               "maskz expands + OR; the issue-#11 merge-masked form wins -33% c7a / -27% c8a (Zen false dep), ~0 c7i/c8i", 0, PV_FN_VBMI2(prim_merge_vv_maskz));
    PV_VARIANT(ST_MERGE_CST_VEC, "asof-76dec21", PV_ISA_AVX512, "76dec21 (prior production)",
               "maskz expand + blend; merge-masked form wins -54% c7a / -51% c8a, -1% c7i / -10% c8i", 0, PV_FN_VBMI2(prim_merge_cv_maskz));
    PV_VARIANT(ST_MERGE_VEC_VEC, "iurii-asm-l7r7", PV_ISA_AVX512, "vec_vec vpexpandb encoding matrix (2026-07)",
               "issue-#11 loop, L-expand 7B / R-expand 7B", 0, PV_FN_VBMI2(prim_merge_vv_l7r7));
    PV_VARIANT(ST_MERGE_VEC_VEC, "iurii-asm-l7r8", PV_ISA_AVX512, "vec_vec vpexpandb encoding matrix (2026-07)",
               "issue-#11 loop, L-expand 7B / R-expand 8B (disp8)", 0, PV_FN_VBMI2(prim_merge_vv_l7r8));
    PV_VARIANT(ST_MERGE_VEC_VEC, "iurii-asm-l8r7", PV_ISA_AVX512, "vec_vec vpexpandb encoding matrix (2026-07)",
               "issue-#11 loop, L-expand 8B / R-expand 7B; fastest on c7a/Zen 4", 0, PV_FN_VBMI2(prim_merge_vv_l8r7));
    PV_VARIANT(ST_MERGE_VEC_VEC, "iurii-asm-l8r8", PV_ISA_AVX512, "vec_vec vpexpandb encoding matrix (2026-07)",
               "issue-#11 loop, L-expand 8B / R-expand 8B; fastest on c8a/Zen 5", 0, PV_FN_VBMI2(prim_merge_vv_l8r8));
    PV_VARIANT(ST_MERGE_CST_VEC, "iurii-asm-exp7b", PV_ISA_AVX512, "vpexpandb encoding study (2026-07)",
               "issue-#11 loop, asm-pinned to the 7-BYTE vpexpandb encoding (no displacement); x1.15-1.21 SLOWER on c8a/Zen 5 (op-cache packing -> int-sched steering), ~0 on Intel (c6i/c7i/c8i)", 0, PV_FN_VBMI2(prim_merge_cv_exp7b));
    PV_VARIANT(ST_MERGE_CST_VEC, "iurii-asm-exp8b", PV_ISA_AVX512, "vpexpandb encoding study (2026-07)",
               "same loop, asm-pinned to the 8-BYTE vpexpandb encoding (zero disp8 byte, r13-class base); the fast packing on c8a/Zen 5, ~0 on Intel (c6i/c7i/c8i)", 0, PV_FN_VBMI2(prim_merge_cv_exp8b));
    PV_VARIANT(ST_MERGE_CST_VEC, "iurii-asm-gnr7b", PV_ISA_AVX512, "gcc14 c8i/GNR-tuned schedule (2026-07)",
               "same loop, gcc14 -march=native-on-GNR schedule (bm chain first, rc accumulate after the store), verbatim transcription incl. the 7B no-disp expand; ~6% over the generic schedule on c8i", 0, PV_FN_VBMI2(prim_merge_cv_gnr7b));
    PV_VARIANT(ST_MERGE_CST_VEC, "iurii-asm-gnr8b", PV_ISA_AVX512, "gcc14 c8i/GNR-tuned schedule (2026-07)",
               "GNR schedule re-encoded with the 8B disp8 expand (r13 base) — isolates schedule from encoding on c7a+c8a/Zen", 0, PV_FN_VBMI2(prim_merge_cv_gnr8b));
    PV_VARIANT(ST_MERGE_CST_VEC, "iurii2-c-vrb",   PV_ISA_AVX512, "issue-#11 comment patch, verbatim (2026-07)",
               "Iurii's exact follow-up: ssize_t, separate bm cursor, 4x unroll with SERIAL rc updates (vs u4's prefix cursors)", 0, PV_FN_VBMI2(prim_merge_cv_c_vrb));
    PV_VARIANT(ST_MERGE_CST_VEC, "iurii2-c-kld",   PV_ISA_AVX512, "issue-#11 suggestion 3 (2026-07)",
               "kld = load the k mask register straight from memory (kmovq mem-form, no GPR detour) + popcnt from memory; duplicated L1 load, one uop fewer, mask via load pipes", 0, PV_FN_VBMI2(prim_merge_cv_c_kld));
    PV_VARIANT(ST_MERGE_VEC_VEC, "iurii2-c-u4szt", PV_ISA_AVX512, "issue-#11 follow-up (2026-07)",
               "4x unroll + size_t cursors combined", 0, PV_FN_VBMI2(prim_merge_vv_c_u4szt));
    PV_VARIANT(ST_MERGE_VEC_VEC, "iurii2-c-vrb",   PV_ISA_AVX512, "issue-#11 comment patch style (2026-07)",
               "vrb structure for vec_vec: ssize_t, separate bm cursor, 4x unroll, serial lc/rc updates", 0, PV_FN_VBMI2(prim_merge_vv_c_vrb));
    PV_VARIANT(ST_MERGE_VEC_VEC, "iurii2-c-kld",   PV_ISA_AVX512, "issue-#11 suggestion 3 (2026-07)",
               "kld = load the k mask register straight from memory (kmovq mem-form) + knot for the L side + popcnt from memory", 0, PV_FN_VBMI2(prim_merge_vv_c_kld));
    /* merge_flat — AVX2 ymm widening of the SSE d5/d6/d7 (see kernels above) */
    for (int d = 5; d <= 7; d++)
        PV_VARIANT_D(ST_MERGE_FLAT, "avx2-ymm", d, PV_ISA_AVX2,
                     "256-bit widening of the production SSE flat decoders",
                     "pair-gather + mul-as-shift unpack, pshufb scatter + blendv tree, 32 codes/iter", 0,
                     PV_FN_SSE_AVX2(prim_merge_flat_avx2ymm));

    /* merge_vec_vec — NEON */
    PV_VARIANT(ST_MERGE_VEC_VEC, "asof-f9974f5", PV_ISA_NEON, "f9974f5 (prior production)",
               "pre-PR#22 COM64 main loop (no software pipelining; GPR-side popcount)", 0, PV_FN_NEON(prim_merge_vv_asof_f9974f5));
    PV_VARIANT(ST_MERGE_VEC_VEC, "com64",     PV_ISA_NEON, "asof-d24c0eb (prior production)",
               "V5 stride-64 COM, 4 chunks, byte prefix-sum cursors; production before the two-table merge", 0, PV_FN_NEON(prim_merge_vv_com64));
    PV_VARIANT(ST_MERGE_VEC_VEC, "asof-4dd08e3", PV_ISA_NEON, "4dd08e3 (2026-05-10)",
               "genesis bottom-up tree_merge: stride-8, 1 vqtbl1/8, serial cursor", 0, PV_FN_NEON(prim_merge_vv_asof_4dd08e3));
    PV_VARIANT(ST_MERGE_VEC_VEC, "cursor16",  PV_ISA_NEON, "historical 2c8606e (pre-5cccccc)",
               "stride-16 shipped merge before COM64; loop-carried cursor", 0, PV_FN_NEON(prim_merge_vv_cursor16));
    PV_VARIANT(ST_MERGE_VEC_VEC, "prefix128", PV_ISA_NEON, "Jeff Plaisance / PR #4",
               "vpaddlq fold + u16 SWAR prefix, 128/iter; M1 Max win, ARM-untested", 0, PV_FN_NEON(prim_merge_vv_prefix128));
    PV_VARIANT(ST_MERGE_VEC_VEC, "prefix64",  PV_ISA_NEON, "Jeff Plaisance / PR #4",
               "prefix128 block style at 64-code stride", 0, PV_FN_NEON(prim_merge_vv_prefix64));
    PV_VARIANT(ST_MERGE_VEC_VEC, "pcpc",      PV_ISA_NEON, "bench_prim experiment",
               "precomputed-popcount (pre-filled table); M4 wash, store-bound", 0, PV_FN_NEON(prim_merge_vv_pcpc));
    PV_VARIANT(ST_MERGE_VEC_VEC, "pcpc_full", PV_ISA_NEON, "bench_prim experiment",
               "pcpc + table fill inside the timed region", 0, PV_FN_NEON(prim_merge_vv_pcpc_full));
    PV_VARIANT(ST_MERGE_VEC_VEC, "unroll8",   PV_ISA_NEON, "bench_prim experiment",
               "8-way unroll, in-register vcnt popcount; doubles L/R load count", 0, PV_FN_NEON(prim_merge_vv_unroll8));
    PV_VARIANT(ST_MERGE_VEC_VEC, "com32",     PV_ISA_NEON, "bench_merge_neon.c",
               "COM prefix-sum cursor-decouple, 32/iter (2 chunks); narrower than shipped com64", 0, PV_FN_NEON(prim_merge_vv_com32));
    PV_VARIANT(ST_MERGE_VEC_VEC, "com128",    PV_ISA_NEON, "bench_merge_neon.c",
               "COM 128/iter (8 chunks, lo/hi u64 split + cross-half bias); regalloc-heavy, loses on Graviton", 0, PV_FN_NEON(prim_merge_vv_com128));
    PV_VARIANT(ST_MERGE_VEC_VEC, "aqrit-notab", PV_ISA_NEON, "aqrit / issue #26 gist",
               "table-free shuffle synthesis: vcnt prefix popcount + exclusive-minus-inclusive select; ~12 ALU ops per 16, no 8KiB index tables", 0, PV_FN_NEON(prim_merge_vv_aqrit));
    /* merge_cst_cst — NEON */
    PV_VARIANT(ST_MERGE_CST_CST, "tbl",       PV_ISA_NEON, "bench_prim experiment",
               "256x8 mask->0xFF/0x00 LUT + vand/veor blend", 0, PV_FN_NEON(prim_merge_cc_tbl));
    PV_VARIANT(ST_MERGE_CST_CST, "blendtab",  PV_ISA_NEON, "bench_prim experiment",
               "precomputed 256x8 blended-output LUT (no hot-loop compute)", 0, PV_FN_NEON(prim_merge_cc_blendtab));
    PV_VARIANT(ST_MERGE_CST_CST, "vtblq",     PV_ISA_NEON, "bench_prim experiment",
               "16-lane vtstq + vqtbl1q", 0, PV_FN_NEON(prim_merge_cc_vtblq));
    PV_VARIANT(ST_MERGE_CST_CST, "vtbl",      PV_ISA_NEON, "bench_prim experiment",
               "8-lane vtst + vtbl1 x2", 0, PV_FN_NEON(prim_merge_cc_vtbl));
    PV_VARIANT(ST_MERGE_CST_CST, "d1flat",    PV_ISA_NEON, "bench_prim experiment",
               "D=1 flat-decode shape (vshl + vqtbl1q)", 0, PV_FN_NEON(prim_merge_cc_d1flat));
    PV_VARIANT(ST_MERGE_CST_VEC, "com64",     PV_ISA_NEON, "asof-d24c0eb (prior production)",
               "V5 COM cst_vec, broadcast L; production before the two-table merge", 0, PV_FN_NEON(prim_merge_cv_com64));
    PV_VARIANT(ST_MERGE_CST_VEC, "skewfuse",  PV_ISA_NEON, "fused per-word fast paths (no pre-pass)",
               "production linear loop + two per-word branches: mask==0 -> 64B constant fill, mask==~0 -> 64B straight copy; worst-case tax is a predictable branch pair per word, risk is mispredicts at run boundaries (bench with --bm=runs)", 0,
               PV_FN_NEON(prim_merge_cst_vec_skewfuse));
    PV_VARIANT(ST_MERGE_CST_VEC, "unroll16",  PV_ISA_SSE4, "asof-d24c0eb (prior production)",
               "2x-unrolled expand_tab cst_vec; production before the two-table merge", 0, PV_FN_SSE(prim_merge_cv_unroll16));
    PV_VARIANT(ST_MERGE_CST_VEC, "c3roll-asm-p0",  PV_ISA_SSE4, "c3/IVB loop-top phase study (2026-07)",
               "clang-20 c3 production two-table pshufb loop, verbatim (VEX, needs AVX1), loop top pinned to phase 0 mod 32 — a fast IVB phase", 0, PV_FN_SSE(prim_merge_cv_ivb_p0));
    PV_VARIANT(ST_MERGE_CST_VEC, "c3roll-asm-p16", PV_ISA_SSE4, "c3/IVB loop-top phase study (2026-07)",
               "same bytes, loop top at phase 16 mod 32 — the slow c3/IVB phase (32B uop-window split, +11% on c3); the phase roll is the c3 binary lottery", 0, PV_FN_SSE(prim_merge_cv_ivb_p16));
    /* merge_vec_vec — x86 COM / prefix-sum forms (IDEAS: x86 COM merge). */
    PV_VARIANT(ST_MERGE_VEC_VEC, "sse_com",        PV_ISA_SSE4, "bench_merge_x86.c",
               "64 codes/iter, 8 pshufb merges, SWAR-popcnt prefix cursors", 0,
               PV_FN_SSE(prim_merge_vv_sse_com));
    PV_VARIANT(ST_MERGE_VEC_VEC, "sse_com_pshufb", PV_ISA_SSE4, "bench_merge_x86.c",
               "sse_com but Mula pshufb bytewise popcount", 0,
               PV_FN_SSE(prim_merge_vv_sse_com_pshufb));
    PV_VARIANT(ST_MERGE_VEC_VEC, "sse_com128",     PV_ISA_SSE4, "bench_merge_x86.c",
               "128 codes/iter, full-width pshufb popcount", 0,
               PV_FN_SSE(prim_merge_vv_sse_com128));
    PV_VARIANT(ST_MERGE_VEC_VEC, "avx2_com",       PV_ISA_AVX2, "bench_merge_x86.c",
               "16 codes/_mm256_shuffle_epi8 (2 lanes), COM cursors", 0,
               PV_FN_SSE_AVX2(prim_merge_vv_avx2_com));
    PV_VARIANT(ST_MERGE_VEC_VEC, "prepop",         PV_ISA_AVX2, "bench_merge_avx2.c",
               "16 B/iter, on-the-fly prefix-popcount pshufb controls (no tab)", 0,
               PV_FN_SSE_AVX2(prim_merge_vv_prepop));
    PV_VARIANT(ST_MERGE_VEC_VEC, "unroll16",       PV_ISA_SSE4, "asof-d24c0eb (prior production)",
               "2x-unrolled stride-16 expand_tab merge; production before the two-table merge", 0,
               PV_FN_SSE(prim_merge_vv_unroll16));
}

#endif /* PIVCO_PRIM_VARIANTS_MERGE_H */
