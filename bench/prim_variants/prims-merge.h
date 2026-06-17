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
 * merge_vec_vec : cursor16 — pre-COM64 shipped stride-16 merge
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
 * Registry — merge family (no-op where the ISA is unavailable)
 * ========================================================================== */
static void pv_register_merge(void) {
#if defined(USE_NEON_KERNELS)
    pv_build_cc_tables();
    /* merge_vec_vec */
    PV_VARIANT(ST_MERGE_VEC_VEC, "cursor16",  PV_ISA_NEON, "historical (pre-5cccccc)",
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
}

#endif /* PIVCO_PRIM_VARIANTS_MERGE_H */
