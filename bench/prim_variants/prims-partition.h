/* bench/prim_variants/prims-partition.h — partition-family variant graveyard.
 *
 * Logical primitives: enc_partition_full (ST_PART), enc_partition_none
 * (ST_BMBUILD), enc_partition_right (ST_FUSEDHALF).  See prims.h for the
 * contract + naming (PV_ = constants/macros, pv_ = plumbing, prim_ = kernels).
 * Registry + per-entry provenance are at the bottom.
 */
#ifndef PIVCO_PRIM_VARIANTS_PARTITION_H
#define PIVCO_PRIM_VARIANTS_PARTITION_H

/* ============================================================================
 * prefix64 — 64-codes/iter wide-mask + SWAR byte-prefix-sum partition
 *   Build 8 group-mask bytes at once, turn the per-group popcounts into a
 *   byte-wise prefix sum via *0x0101010101010101 so the 8 groups compact
 *   independently (no loop-carried cursor).  Depends on production symbols in
 *   scope here: compress_tab[], compress_popcnt[], enc_mask8_codes_la_neon().
 * ========================================================================== */
#if defined(USE_NEON_KERNELS)

static const uint16_t PV_WLO[8] = {1,2,4,8,16,32,64,128};

/* 64 codes -> 8 group mask bytes (wire order): vtst the partition bit, AND by
   {1..128} for one bit/lane, then a 3-level vpaddq tree -> uint16x8 -> vmovn. */
static inline uint8x8_t prim_enc_mask64_neon(const uint16_t *cw,
                                             uint16x8_t testbit, uint16x8_t wlo) {
    uint16x8_t W0 = vandq_u16(vtstq_u16(vld1q_u16(cw     ), testbit), wlo);
    uint16x8_t W1 = vandq_u16(vtstq_u16(vld1q_u16(cw +  8), testbit), wlo);
    uint16x8_t W2 = vandq_u16(vtstq_u16(vld1q_u16(cw + 16), testbit), wlo);
    uint16x8_t W3 = vandq_u16(vtstq_u16(vld1q_u16(cw + 24), testbit), wlo);
    uint16x8_t W4 = vandq_u16(vtstq_u16(vld1q_u16(cw + 32), testbit), wlo);
    uint16x8_t W5 = vandq_u16(vtstq_u16(vld1q_u16(cw + 40), testbit), wlo);
    uint16x8_t W6 = vandq_u16(vtstq_u16(vld1q_u16(cw + 48), testbit), wlo);
    uint16x8_t W7 = vandq_u16(vtstq_u16(vld1q_u16(cw + 56), testbit), wlo);
    uint16x8_t a = vpaddq_u16(W0, W1), b = vpaddq_u16(W2, W3);
    uint16x8_t c = vpaddq_u16(W4, W5), d = vpaddq_u16(W6, W7);
    uint16x8_t e = vpaddq_u16(a, b), f = vpaddq_u16(c, d);
    return vmovn_u16(vpaddq_u16(e, f));
}

/* full: both halves compacted, in place. */
static inline int prim_part_full_prefix64_neon(uint16_t *codes_la, int n, int depth,
                                               uint8_t *bm, uint16_t *right_out) {
    int n_left = 0, n_right = 0, j = 0;
    uint16x8_t testbit = vdupq_n_u16((uint16_t)(1u << (15 - depth)));
    uint16x8_t wlo = vld1q_u16(PV_WLO);
    for (; j + 64 <= n; j += 64) {
        uint8x8_t mask8 = prim_enc_mask64_neon(codes_la + j, testbit, wlo);
        vst1_u8(bm + (j >> 3), mask8);
        uint64_t mk   = vget_lane_u64(vreinterpret_u64_u8(mask8), 0);
        uint64_t pc   = vget_lane_u64(vreinterpret_u64_u8(vcnt_u8(mask8)), 0);
        uint64_t pref = pc * 0x0101010101010101ULL;
        #define PV_PART_GRP(GI, REXCL)                                          \
        do {                                                                    \
            uint8_t mask = (uint8_t)(mk >> (8 * (GI)));                         \
            uint32_t r_excl = (REXCL);                                          \
            uint32_t l_excl = (uint32_t)(8 * (GI)) - r_excl;                    \
            uint8x16_t data = vreinterpretq_u8_u16(                             \
                vld1q_u16(codes_la + j + 8 * (GI)));                            \
            const uint8_t *tab = compress_tab[mask];                           \
            vst1q_u8((uint8_t *)(right_out + n_right + r_excl),                \
                     vqtbl1q_u8(data, vld1q_u8(tab)));                         \
            vst1q_u8((uint8_t *)(codes_la  + n_left  + l_excl),               \
                     vqtbl1q_u8(data, vld1q_u8(tab + 16)));                    \
        } while (0)
        PV_PART_GRP(0, 0);
        PV_PART_GRP(1, (uint32_t)((pref)       & 0xFF));
        PV_PART_GRP(2, (uint32_t)((pref >> 8)  & 0xFF));
        PV_PART_GRP(3, (uint32_t)((pref >> 16) & 0xFF));
        PV_PART_GRP(4, (uint32_t)((pref >> 24) & 0xFF));
        PV_PART_GRP(5, (uint32_t)((pref >> 32) & 0xFF));
        PV_PART_GRP(6, (uint32_t)((pref >> 40) & 0xFF));
        PV_PART_GRP(7, (uint32_t)((pref >> 48) & 0xFF));
        #undef PV_PART_GRP
        uint32_t tot_r = (uint32_t)(pref >> 56);
        n_right += tot_r; n_left += 64 - tot_r;
    }
    for (; j + 8 <= n; j += 8) {            /* production stride-8 tail */
        uint16x8_t code_vec = vld1q_u16(codes_la + j);
        uint8_t mask = enc_mask8_codes_la_neon(code_vec, -(15 - depth));
        bm[j >> 3] = mask;
        const uint8_t *tab = compress_tab[mask];
        uint8x16_t data = vreinterpretq_u8_u16(code_vec);
        int nr = compress_popcnt[mask];
        vst1q_u8((uint8_t *)(right_out + n_right), vqtbl1q_u8(data, vld1q_u8(tab)));
        vst1q_u8((uint8_t *)(codes_la  + n_left ), vqtbl1q_u8(data, vld1q_u8(tab + 16)));
        n_right += nr; n_left += (8 - nr);
    }
    if (j < n) {
        int tail = n - j, shift_d = 15 - depth;
        uint16_t tb[8]; for (int k = 0; k < tail; k++) tb[k] = codes_la[j + k];
        uint8_t mask = 0;
        for (int k = 0; k < tail; k++) mask |= (uint8_t)(((tb[k] >> shift_d) & 1) << k);
        bm[j >> 3] = mask;
        for (int k = 0; k < tail; k++)
            if (mask & (1 << k)) right_out[n_right++] = tb[k];
            else                 codes_la[n_left++]   = tb[k];
    }
    return n_right;
}

/* none: wide mask build only (no compaction / no cursor). */
static inline int prim_part_none_prefix64_neon(uint16_t *codes_la, int n, int depth, uint8_t *bm) {
    int n_right = 0, j = 0;
    uint16x8_t testbit = vdupq_n_u16((uint16_t)(1u << (15 - depth)));
    uint16x8_t wlo = vld1q_u16(PV_WLO);
    for (; j + 64 <= n; j += 64) {
        uint8x8_t mask8 = prim_enc_mask64_neon(codes_la + j, testbit, wlo);
        vst1_u8(bm + (j >> 3), mask8);
        n_right += vaddv_u8(vcnt_u8(mask8));
    }
    for (; j + 8 <= n; j += 8) {
        uint8_t mask = enc_mask8_codes_la_neon(vld1q_u16(codes_la + j), -(15 - depth));
        bm[j >> 3] = mask; n_right += compress_popcnt[mask];
    }
    if (j < n) {
        int tail = n - j, shift_d = 15 - depth; uint8_t mask = 0;
        for (int k = 0; k < tail; k++) mask |= (uint8_t)(((codes_la[j + k] >> shift_d) & 1) << k);
        bm[j >> 3] = mask; n_right += __builtin_popcount(mask);
    }
    return n_right;
}

/* right: compact the RIGHT half only (half the store volume of full). */
static inline int prim_part_right_prefix64_neon(uint16_t *codes_la, int n, int depth,
                                                uint8_t *bm, uint16_t *right_out) {
    int n_right = 0, j = 0;
    uint16x8_t testbit = vdupq_n_u16((uint16_t)(1u << (15 - depth)));
    uint16x8_t wlo = vld1q_u16(PV_WLO);
    for (; j + 64 <= n; j += 64) {
        uint8x8_t mask8 = prim_enc_mask64_neon(codes_la + j, testbit, wlo);
        vst1_u8(bm + (j >> 3), mask8);
        uint64_t mk   = vget_lane_u64(vreinterpret_u64_u8(mask8), 0);
        uint64_t pc   = vget_lane_u64(vreinterpret_u64_u8(vcnt_u8(mask8)), 0);
        uint64_t pref = pc * 0x0101010101010101ULL;
        #define PV_PART_GRP_R(GI, REXCL)                                        \
        do {                                                                    \
            uint8_t mask = (uint8_t)(mk >> (8 * (GI)));                         \
            uint8x16_t data = vreinterpretq_u8_u16(                            \
                vld1q_u16(codes_la + j + 8 * (GI)));                           \
            vst1q_u8((uint8_t *)(right_out + n_right + (REXCL)),              \
                     vqtbl1q_u8(data, vld1q_u8(compress_tab[mask])));         \
        } while (0)
        PV_PART_GRP_R(0, 0);
        PV_PART_GRP_R(1, (uint32_t)((pref)       & 0xFF));
        PV_PART_GRP_R(2, (uint32_t)((pref >> 8)  & 0xFF));
        PV_PART_GRP_R(3, (uint32_t)((pref >> 16) & 0xFF));
        PV_PART_GRP_R(4, (uint32_t)((pref >> 24) & 0xFF));
        PV_PART_GRP_R(5, (uint32_t)((pref >> 32) & 0xFF));
        PV_PART_GRP_R(6, (uint32_t)((pref >> 40) & 0xFF));
        PV_PART_GRP_R(7, (uint32_t)((pref >> 48) & 0xFF));
        #undef PV_PART_GRP_R
        n_right += (uint32_t)(pref >> 56);
    }
    for (; j + 8 <= n; j += 8) {
        uint16x8_t code_vec = vld1q_u16(codes_la + j);
        uint8_t mask = enc_mask8_codes_la_neon(code_vec, -(15 - depth));
        bm[j >> 3] = mask;
        vst1q_u8((uint8_t *)(right_out + n_right),
                 vqtbl1q_u8(vreinterpretq_u8_u16(code_vec), vld1q_u8(compress_tab[mask])));
        n_right += compress_popcnt[mask];
    }
    if (j < n) {
        int tail = n - j, shift_d = 15 - depth;
        uint16_t tb[8]; for (int k = 0; k < tail; k++) tb[k] = codes_la[j + k];
        uint8_t mask = 0;
        for (int k = 0; k < tail; k++) mask |= (uint8_t)(((tb[k] >> shift_d) & 1) << k);
        bm[j >> 3] = mask;
        for (int k = 0; k < tail; k++) if (mask & (1 << k)) right_out[n_right++] = tb[k];
    }
    return n_right;
}

/* ctx_t adapters (registered below). */
static void prim_part_full_prefix64 (const ctx_t *c){ prim_part_full_prefix64_neon (c->la_work, c->n, c->depth, c->bm, c->tmp16); }
static void prim_part_none_prefix64 (const ctx_t *c){ prim_part_none_prefix64_neon (c->la_work, c->n, c->depth, c->bm); }
static void prim_part_right_prefix64(const ctx_t *c){ prim_part_right_prefix64_neon(c->la_work, c->n, c->depth, c->bm, c->tmp16); }

#endif /* USE_NEON_KERNELS */

/* ============================================================================
 * Registry — partition family (no-op where the ISA is unavailable)
 * ========================================================================== */
static void pv_register_partition(void) {
#if defined(USE_NEON_KERNELS)
    PV_VARIANT(ST_PART,      "prefix64", PV_ISA_NEON,
               "Jeff Plaisance / 6d61760",
               "M4 FULL +15-18% vs shipped COM; needs Graviton check", 1,
               prim_part_full_prefix64);
    PV_VARIANT(ST_BMBUILD,   "prefix64", PV_ISA_NEON,
               "Jeff Plaisance / 6d61760",
               "M4 NONE ~3-4% faster than shipped COM", 0,
               prim_part_none_prefix64);
    PV_VARIANT(ST_FUSEDHALF, "prefix64", PV_ISA_NEON,
               "Jeff Plaisance / 6d61760",
               "M4 RIGHT ~2-3% slower than shipped COM", 0,
               prim_part_right_prefix64);
#endif
}

#endif /* PIVCO_PRIM_VARIANTS_PARTITION_H */
