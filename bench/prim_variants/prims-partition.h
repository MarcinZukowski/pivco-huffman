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

/* coalesce-vext switch macros (from bench_coalesce.c); used by
 * prim_part_coal_vext below.  zero_v must be in scope. */
#define COALESCE_CASE_0(V_v, cnt_var, accum_var, so_far_var, out_p, n_var) \
    case 0: {                                                              \
        uint8x16_t _merged = vorrq_u8((accum_var), (V_v));                 \
        if ((cnt_var) < 8) { (accum_var) = _merged; (so_far_var) = (cnt_var); } \
        else { vst1q_u8((out_p) + (n_var), _merged); (n_var) += 16;        \
            (accum_var) = zero_v; (so_far_var) = (cnt_var) - 8; }          \
    } break;
#define COALESCE_CASE_K(K, V_v, cnt_var, accum_var, so_far_var, out_p, n_var) \
    case K: {                                                                 \
        uint8x16_t _shifted = vextq_u8(zero_v, (V_v), 16 - (K) * 2);          \
        uint8x16_t _merged  = vorrq_u8((accum_var), _shifted);                \
        if ((K) + (cnt_var) < 8) { (accum_var) = _merged; (so_far_var) = (K) + (cnt_var); } \
        else { vst1q_u8((out_p) + (n_var), _merged); (n_var) += 16;           \
            (accum_var) = vextq_u8((V_v), zero_v, (8 - (K)) * 2); (so_far_var) = (K) + (cnt_var) - 8; } \
    } break;
#define COALESCE_SWITCH(V_v, cnt_var, accum_var, so_far_var, out_p, n_var)    \
    switch (so_far_var) {                                                     \
        COALESCE_CASE_0(   V_v, cnt_var, accum_var, so_far_var, out_p, n_var) \
        COALESCE_CASE_K(1, V_v, cnt_var, accum_var, so_far_var, out_p, n_var) \
        COALESCE_CASE_K(2, V_v, cnt_var, accum_var, so_far_var, out_p, n_var) \
        COALESCE_CASE_K(3, V_v, cnt_var, accum_var, so_far_var, out_p, n_var) \
        COALESCE_CASE_K(4, V_v, cnt_var, accum_var, so_far_var, out_p, n_var) \
        COALESCE_CASE_K(5, V_v, cnt_var, accum_var, so_far_var, out_p, n_var) \
        COALESCE_CASE_K(6, V_v, cnt_var, accum_var, so_far_var, out_p, n_var) \
        COALESCE_CASE_K(7, V_v, cnt_var, accum_var, so_far_var, out_p, n_var) \
    }

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

/* ============================================================================
 * asof-5f3222e — pre-COM stride-8 serial-cursor FULL partition (2026-05-26)
 *   Production encode partition before COM (6ddd75d/f151ce7): one movmask + one
 *   compress_tab[mask] load -> two vqtbl1q + two stores per 8 codes, serial
 *   cursor, 1-byte bm store.
 * ========================================================================== */
static inline int prim_part_full_asof_5f3222e_neon(uint16_t *codes_la, int n, int depth,
                                                   uint8_t *bm, uint16_t *right_out) {
    int n_left = 0, n_right = 0, j = 0;
    int neg_shift_d = -(15 - depth);
    for (; j + 8 <= n; j += 8) {
        uint16x8_t code_vec = vld1q_u16(codes_la + j);
        uint8_t mask = enc_mask8_codes_la_neon(code_vec, neg_shift_d);
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
static void prim_part_full_asof_5f3222e(const ctx_t *c){ prim_part_full_asof_5f3222e_neon(c->la_work, c->n, c->depth, c->bm, c->tmp16); }

/* ctx_t adapters (registered below). */
static void prim_part_full_prefix64 (const ctx_t *c){ prim_part_full_prefix64_neon (c->la_work, c->n, c->depth, c->bm, c->tmp16); }
static void prim_part_none_prefix64 (const ctx_t *c){ prim_part_none_prefix64_neon (c->la_work, c->n, c->depth, c->bm); }
static void prim_part_right_prefix64(const ctx_t *c){ prim_part_right_prefix64_neon(c->la_work, c->n, c->depth, c->bm, c->tmp16); }

/* ============================================================================
 * com / com_v2_transpose — 64 codes/iter, 8 chunks, *0x0101.. prefix-sum
 *   cursors (no per-chunk serial n_left/n_right add, no compress_popcnt[]
 *   load).  The two differ only in how the 8 partition-mask bytes are built:
 *     com           : 8 independent enc_mask8 (per-chunk vaddvq movemask)
 *     com_v2_transpose : an 8x8 vsli transpose of the per-lane depth bits.
 *   com_v2 is a documented LOSER (the transpose costs more than the 8 vaddvq
 *   it removes); the shipped production partition (com_v3, vtst+vpaddq tree)
 *   replaced both.  From bench_partition_neon.c.
 * ========================================================================== */
static inline uint8_t pv_enc_mask8_neon(uint16x8_t code_vec, int neg_shift_d) {
    int16x8_t shr_vec = vdupq_n_s16((int16_t)neg_shift_d);
    uint16x8_t bit_lsb = vandq_u16(vshlq_u16(code_vec, shr_vec), vdupq_n_u16(1));
    static const int16_t weights[8] = {0,1,2,3,4,5,6,7};
    uint16x8_t weighted = vshlq_u16(bit_lsb, vld1q_s16(weights));
    return (uint8_t)vaddvq_u16(weighted);
}

/* shared compact+scatter of one 64-code window given the 8 mask bytes already
 * packed into mask_word and the cv0..cv7 code vectors. */
#define PV_PART_COM_BODY(MASK_WORD)                                          \
    memcpy(bm + (j >> 3), &MASK_WORD, 8);                                    \
    uint8x8_t pc_v = vcnt_u8(vcreate_u8(MASK_WORD));                         \
    uint64_t pc_word = vget_lane_u64(vreinterpret_u64_u8(pc_v), 0);          \
    uint64_t pfx = pc_word * 0x0101010101010101ULL;                         \
    PV_PCHUNK(0,cv0); PV_PCHUNK(1,cv1); PV_PCHUNK(2,cv2); PV_PCHUNK(3,cv3);  \
    PV_PCHUNK(4,cv4); PV_PCHUNK(5,cv5); PV_PCHUNK(6,cv6); PV_PCHUNK(7,cv7);  \
    uint32_t total_r = (uint32_t)(pfx >> 56);                               \
    n_right += total_r; n_left += 64 - total_r;

#define PV_PCHUNK(K_, CV) do {                                              \
        uint8_t M = (uint8_t)(mask_word >> (8*(K_)));                        \
        uint32_t cr = (K_)==0 ? 0u : (uint32_t)((pfx >> (8*((K_)-1))) & 0xFF);\
        uint32_t cl = 8u*(K_) - cr;                                         \
        const uint8_t *tab = compress_tab[M];                              \
        uint8x16_t data  = vreinterpretq_u8_u16(CV);                       \
        vst1q_u8((uint8_t *)(right_out + n_right + cr), vqtbl1q_u8(data, vld1q_u8(tab)));      \
        vst1q_u8((uint8_t *)(codes_la  + n_left  + cl), vqtbl1q_u8(data, vld1q_u8(tab + 16))); \
    } while (0)

static inline int prim_part_com_neon(uint16_t *codes_la, int n, int depth,
                                     uint8_t *bm, uint16_t *right_out) {
    int n_left = 0, n_right = 0, j = 0;
    int neg_shift_d = -(15 - depth);
    for (; j + 64 <= n; j += 64) {
        uint16x8_t cv0=vld1q_u16(codes_la+j),    cv1=vld1q_u16(codes_la+j+8),
                   cv2=vld1q_u16(codes_la+j+16), cv3=vld1q_u16(codes_la+j+24),
                   cv4=vld1q_u16(codes_la+j+32), cv5=vld1q_u16(codes_la+j+40),
                   cv6=vld1q_u16(codes_la+j+48), cv7=vld1q_u16(codes_la+j+56);
        uint8_t m0=pv_enc_mask8_neon(cv0,neg_shift_d), m1=pv_enc_mask8_neon(cv1,neg_shift_d),
                m2=pv_enc_mask8_neon(cv2,neg_shift_d), m3=pv_enc_mask8_neon(cv3,neg_shift_d),
                m4=pv_enc_mask8_neon(cv4,neg_shift_d), m5=pv_enc_mask8_neon(cv5,neg_shift_d),
                m6=pv_enc_mask8_neon(cv6,neg_shift_d), m7=pv_enc_mask8_neon(cv7,neg_shift_d);
        uint64_t mask_word = (uint64_t)m0 | ((uint64_t)m1<<8) | ((uint64_t)m2<<16)
                           | ((uint64_t)m3<<24) | ((uint64_t)m4<<32) | ((uint64_t)m5<<40)
                           | ((uint64_t)m6<<48) | ((uint64_t)m7<<56);
        PV_PART_COM_BODY(mask_word);
    }
    int shift_d = 15 - depth;
    for (; j < n; j++) {
        uint16_t c = codes_la[j];
        if ((c >> shift_d) & 1) right_out[n_right++] = c;
        else                    codes_la[n_left++]   = c;
    }
    return n_right;
}

/* build all 8 partition-mask bytes at once via an 8x8 vsli transpose. */
static inline uint64_t pv_build_8_masks_transpose(
        uint16x8_t cv0, uint16x8_t cv1, uint16x8_t cv2, uint16x8_t cv3,
        uint16x8_t cv4, uint16x8_t cv5, uint16x8_t cv6, uint16x8_t cv7,
        int neg_shift_d) {
    int16x8_t s = vdupq_n_s16((int16_t)neg_shift_d);
    uint16x8_t one = vdupq_n_u16(1);
    uint16x8_t b0=vandq_u16(vshlq_u16(cv0,s),one), b1=vandq_u16(vshlq_u16(cv1,s),one),
               b2=vandq_u16(vshlq_u16(cv2,s),one), b3=vandq_u16(vshlq_u16(cv3,s),one),
               b4=vandq_u16(vshlq_u16(cv4,s),one), b5=vandq_u16(vshlq_u16(cv5,s),one),
               b6=vandq_u16(vshlq_u16(cv6,s),one), b7=vandq_u16(vshlq_u16(cv7,s),one);
    uint16x8_t a0=vtrn1q_u16(b0,b1), a1=vtrn2q_u16(b0,b1),
               a2=vtrn1q_u16(b2,b3), a3=vtrn2q_u16(b2,b3),
               a4=vtrn1q_u16(b4,b5), a5=vtrn2q_u16(b4,b5),
               a6=vtrn1q_u16(b6,b7), a7=vtrn2q_u16(b6,b7);
    uint32x4_t c0=vtrn1q_u32(vreinterpretq_u32_u16(a0),vreinterpretq_u32_u16(a2)),
               c2=vtrn2q_u32(vreinterpretq_u32_u16(a0),vreinterpretq_u32_u16(a2)),
               c1=vtrn1q_u32(vreinterpretq_u32_u16(a1),vreinterpretq_u32_u16(a3)),
               c3=vtrn2q_u32(vreinterpretq_u32_u16(a1),vreinterpretq_u32_u16(a3)),
               c4=vtrn1q_u32(vreinterpretq_u32_u16(a4),vreinterpretq_u32_u16(a6)),
               c6=vtrn2q_u32(vreinterpretq_u32_u16(a4),vreinterpretq_u32_u16(a6)),
               c5=vtrn1q_u32(vreinterpretq_u32_u16(a5),vreinterpretq_u32_u16(a7)),
               c7=vtrn2q_u32(vreinterpretq_u32_u16(a5),vreinterpretq_u32_u16(a7));
    uint64x2_t t0=vtrn1q_u64(vreinterpretq_u64_u32(c0),vreinterpretq_u64_u32(c4)),
               t4=vtrn2q_u64(vreinterpretq_u64_u32(c0),vreinterpretq_u64_u32(c4)),
               t1=vtrn1q_u64(vreinterpretq_u64_u32(c1),vreinterpretq_u64_u32(c5)),
               t5=vtrn2q_u64(vreinterpretq_u64_u32(c1),vreinterpretq_u64_u32(c5)),
               t2=vtrn1q_u64(vreinterpretq_u64_u32(c2),vreinterpretq_u64_u32(c6)),
               t6=vtrn2q_u64(vreinterpretq_u64_u32(c2),vreinterpretq_u64_u32(c6)),
               t3=vtrn1q_u64(vreinterpretq_u64_u32(c3),vreinterpretq_u64_u32(c7)),
               t7=vtrn2q_u64(vreinterpretq_u64_u32(c3),vreinterpretq_u64_u32(c7));
    uint8x8_t acc = vmovn_u16(vreinterpretq_u16_u64(t0));
    acc = vsli_n_u8(acc, vmovn_u16(vreinterpretq_u16_u64(t1)), 1);
    acc = vsli_n_u8(acc, vmovn_u16(vreinterpretq_u16_u64(t2)), 2);
    acc = vsli_n_u8(acc, vmovn_u16(vreinterpretq_u16_u64(t3)), 3);
    acc = vsli_n_u8(acc, vmovn_u16(vreinterpretq_u16_u64(t4)), 4);
    acc = vsli_n_u8(acc, vmovn_u16(vreinterpretq_u16_u64(t5)), 5);
    acc = vsli_n_u8(acc, vmovn_u16(vreinterpretq_u16_u64(t6)), 6);
    acc = vsli_n_u8(acc, vmovn_u16(vreinterpretq_u16_u64(t7)), 7);
    return vget_lane_u64(vreinterpret_u64_u8(acc), 0);
}

static inline int prim_part_com_v2_neon(uint16_t *codes_la, int n, int depth,
                                        uint8_t *bm, uint16_t *right_out) {
    int n_left = 0, n_right = 0, j = 0;
    int neg_shift_d = -(15 - depth);
    for (; j + 64 <= n; j += 64) {
        uint16x8_t cv0=vld1q_u16(codes_la+j),    cv1=vld1q_u16(codes_la+j+8),
                   cv2=vld1q_u16(codes_la+j+16), cv3=vld1q_u16(codes_la+j+24),
                   cv4=vld1q_u16(codes_la+j+32), cv5=vld1q_u16(codes_la+j+40),
                   cv6=vld1q_u16(codes_la+j+48), cv7=vld1q_u16(codes_la+j+56);
        uint64_t mask_word = pv_build_8_masks_transpose(cv0,cv1,cv2,cv3,cv4,cv5,cv6,cv7,neg_shift_d);
        PV_PART_COM_BODY(mask_word);
    }
    int shift_d = 15 - depth;
    for (; j < n; j++) {
        uint16_t c = codes_la[j];
        if ((c >> shift_d) & 1) right_out[n_right++] = c;
        else                    codes_la[n_left++]   = c;
    }
    return n_right;
}
#undef PV_PCHUNK
#undef PV_PART_COM_BODY

static void prim_part_com         (const ctx_t *c){ prim_part_com_neon   (c->la_work, c->n, c->depth, c->bm, c->tmp16); }
static void prim_part_com_v2_trans(const ctx_t *c){ prim_part_com_v2_neon(c->la_work, c->n, c->depth, c->bm, c->tmp16); }

/* ============================================================================
 * split16 / split16_unroll — dense-codes SIMD-mask partition (stride 8 / 16).
 *   The encode-side tree-walk split on left-aligned uint16 codes: SIMD
 *   movemask build (vshlq + vaddvq) then production-shape partition_8 via
 *   compress_tab.  split16 is stride-8; split16_unroll is stride-16 (two
 *   partition_8 per iter for ILP).  From bench_encode_split.c (CODES16 /
 *   CODES16U).  These ARE the modern production partition shape minus the
 *   *0x0101 prefix-sum cursor-decouple; left compacts in place.
 * ========================================================================== */
static inline uint8_t pv_simd_mask8_neon(uint16x8_t code_vec, int neg_shift_d) {
    int16x8_t shr_vec = vdupq_n_s16((int16_t)neg_shift_d);
    uint16x8_t bit_lsb = vandq_u16(vshlq_u16(code_vec, shr_vec), vdupq_n_u16(1));
    static const int16_t weights_shift[8] = {0, 1, 2, 3, 4, 5, 6, 7};
    uint16x8_t weighted = vshlq_u16(bit_lsb, vld1q_s16(weights_shift));
    return (uint8_t)vaddvq_u16(weighted);
}
static inline int pv_partition_8_reg(uint8x16_t data, uint8_t mask,
                                     uint16_t *left_out, uint16_t *right_out) {
    const uint8_t *tab = compress_tab[mask];
    vst1q_u8((uint8_t *)right_out, vqtbl1q_u8(data, vld1q_u8(tab)));
    vst1q_u8((uint8_t *)left_out,  vqtbl1q_u8(data, vld1q_u8(tab + 16)));
    return compress_popcnt[mask];
}
static inline int prim_part_split16_neon(uint16_t *codes_la, int n, int depth,
                                         uint8_t *bm, uint16_t *left_out, uint16_t *right_out) {
    int neg_shift_d = -(15 - depth);
    int n_left = 0, n_right = 0, j = 0;
    for (; j + 8 <= n; j += 8) {
        uint16x8_t code_vec = vld1q_u16(codes_la + j);
        uint8_t mask = pv_simd_mask8_neon(code_vec, neg_shift_d);
        bm[j >> 3] = mask;
        uint8x16_t data = vreinterpretq_u8_u16(code_vec);
        int nr = pv_partition_8_reg(data, mask, left_out + n_left, right_out + n_right);
        n_right += nr; n_left += (8 - nr);
    }
    int shift_d = 15 - depth;
    if (j < n) {
        int tail = n - j; uint16_t tb[8]; for (int k=0;k<tail;k++) tb[k]=codes_la[j+k];
        uint8_t mask = 0; for (int k=0;k<tail;k++) mask |= (uint8_t)(((tb[k]>>shift_d)&1)<<k);
        bm[j>>3] = mask;
        for (int k=0;k<tail;k++) if (mask&(1<<k)) right_out[n_right++]=tb[k]; else left_out[n_left++]=tb[k];
    }
    return n_right;
}
static inline int prim_part_split16_unroll_neon(uint16_t *codes_la, int n, int depth,
                                                uint8_t *bm, uint16_t *left_out, uint16_t *right_out) {
    int neg_shift_d = -(15 - depth);
    int n_left = 0, n_right = 0, j = 0;
    for (; j + 16 <= n; j += 16) {
        uint16x8_t cv0 = vld1q_u16(codes_la + j), cv1 = vld1q_u16(codes_la + j + 8);
        uint8_t m0 = pv_simd_mask8_neon(cv0, neg_shift_d), m1 = pv_simd_mask8_neon(cv1, neg_shift_d);
        bm[j >> 3] = m0; bm[(j >> 3) + 1] = m1;
        int nr0 = pv_partition_8_reg(vreinterpretq_u8_u16(cv0), m0, left_out + n_left, right_out + n_right);
        n_right += nr0; n_left += (8 - nr0);
        int nr1 = pv_partition_8_reg(vreinterpretq_u8_u16(cv1), m1, left_out + n_left, right_out + n_right);
        n_right += nr1; n_left += (8 - nr1);
    }
    int shift_d = 15 - depth;
    for (; j + 8 <= n; j += 8) {
        uint16x8_t cv = vld1q_u16(codes_la + j);
        uint8_t m = pv_simd_mask8_neon(cv, neg_shift_d); bm[j >> 3] = m;
        int nr = pv_partition_8_reg(vreinterpretq_u8_u16(cv), m, left_out + n_left, right_out + n_right);
        n_right += nr; n_left += (8 - nr);
    }
    if (j < n) {
        int tail = n - j; uint16_t tb[8]; for (int k=0;k<tail;k++) tb[k]=codes_la[j+k];
        uint8_t mask = 0; for (int k=0;k<tail;k++) mask |= (uint8_t)(((tb[k]>>shift_d)&1)<<k);
        bm[j>>3] = mask;
        for (int k=0;k<tail;k++) if (mask&(1<<k)) right_out[n_right++]=tb[k]; else left_out[n_left++]=tb[k];
    }
    return n_right;
}
/* left_out aliases la_work for the in-place ST_PART contract. */
static void prim_part_split16       (const ctx_t *c){ prim_part_split16_neon       (c->la_work, c->n, c->depth, c->bm, c->la_work, c->tmp16); }
static void prim_part_split16_unroll(const ctx_t *c){ prim_part_split16_unroll_neon(c->la_work, c->n, c->depth, c->bm, c->la_work, c->tmp16); }

/* ============================================================================
 * coal_* — partition store-coalescing experiments (ALL documented LOSERS;
 *   see docs/COALESCE.md).  They compact a uint16 source into densely-packed
 *   bytes from a PREBUILT bitmap (no bm write, no mask compute), so they map
 *   to the unfused ST_PARTBM (full: left+right) / ST_PARTHALF (right only)
 *   stages — exactly the prebuilt-bitmap re-read shape the unfusing-cost
 *   decomposition uses.  From bench_coalesce.c.  vext = switch-on-so_far;
 *   tbl = runtime vqtbl1q shuffle; macro = 4-iter lookahead; macro1s = macro
 *   but only the left side coalesced; half_tree = single-side OR-tree macro.
 * ========================================================================== */
static const uint8_t PV_COAL_IOTA[16] = {0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15};

static void prim_part_coal_vext(const ctx_t *c) {
    const uint16_t *src = c->la_work; const uint8_t *bitmap = c->bm;
    uint8_t *left_bytes = (uint8_t *)c->la_work, *right_bytes = (uint8_t *)c->tmp16;
    int n = c->n;
    const uint8x16_t zero_v = vdupq_n_u8(0);
    uint8x16_t accum_l = zero_v, accum_r = zero_v;
    int so_far_l = 0, so_far_r = 0, n_left_bytes = 0, n_right_bytes = 0;
    for (int j = 0; j + 8 <= n; j += 8) {
        uint8_t mask = bitmap[j >> 3];
        uint8x16_t data = vld1q_u8((const uint8_t *)(src + j));
        const uint8_t *tab = compress_tab[mask];
        uint8x16_t r_v = vqtbl1q_u8(data, vld1q_u8(tab));
        uint8x16_t l_v = vqtbl1q_u8(data, vld1q_u8(tab + 16));
        int cnt_r = compress_popcnt[mask]; int cnt_l = 8 - cnt_r;
        COALESCE_SWITCH(r_v, cnt_r, accum_r, so_far_r, right_bytes, n_right_bytes)
        COALESCE_SWITCH(l_v, cnt_l, accum_l, so_far_l, left_bytes,  n_left_bytes)
    }
    if (so_far_l > 0) vst1q_u8(left_bytes  + n_left_bytes,  accum_l);
    if (so_far_r > 0) vst1q_u8(right_bytes + n_right_bytes, accum_r);
}

static void prim_part_coal_tbl(const ctx_t *c) {
    const uint16_t *src = c->la_work; const uint8_t *bitmap = c->bm;
    uint8_t *left_bytes = (uint8_t *)c->la_work, *right_bytes = (uint8_t *)c->tmp16;
    int n = c->n;
    const uint8x16_t zero_v = vdupq_n_u8(0);
    const uint8x16_t iota = vld1q_u8(PV_COAL_IOTA);
    uint8x16_t accum_l = zero_v, accum_r = zero_v;
    int so_far_l = 0, so_far_r = 0, n_left_bytes = 0, n_right_bytes = 0;
    for (int j = 0; j + 8 <= n; j += 8) {
        uint8_t mask = bitmap[j >> 3];
        uint8x16_t data = vld1q_u8((const uint8_t *)(src + j));
        const uint8_t *tab = compress_tab[mask];
        uint8x16_t r_v = vqtbl1q_u8(data, vld1q_u8(tab));
        uint8x16_t l_v = vqtbl1q_u8(data, vld1q_u8(tab + 16));
        int cnt_r = compress_popcnt[mask]; int cnt_l = 8 - cnt_r;
        { uint8x16_t shuf_left = vsubq_u8(iota, vdupq_n_u8((uint8_t)(so_far_r * 2)));
          uint8x16_t merged = vorrq_u8(accum_r, vqtbl1q_u8(r_v, shuf_left));
          int new_sf = so_far_r + cnt_r;
          if (new_sf >= 8) { vst1q_u8(right_bytes + n_right_bytes, merged); n_right_bytes += 16;
            accum_r = vqtbl1q_u8(r_v, vaddq_u8(iota, vdupq_n_u8((uint8_t)((8 - so_far_r) * 2)))); so_far_r = new_sf - 8;
          } else { accum_r = merged; so_far_r = new_sf; } }
        { uint8x16_t shuf_left = vsubq_u8(iota, vdupq_n_u8((uint8_t)(so_far_l * 2)));
          uint8x16_t merged = vorrq_u8(accum_l, vqtbl1q_u8(l_v, shuf_left));
          int new_sf = so_far_l + cnt_l;
          if (new_sf >= 8) { vst1q_u8(left_bytes + n_left_bytes, merged); n_left_bytes += 16;
            accum_l = vqtbl1q_u8(l_v, vaddq_u8(iota, vdupq_n_u8((uint8_t)((8 - so_far_l) * 2)))); so_far_l = new_sf - 8;
          } else { accum_l = merged; so_far_l = new_sf; } }
    }
    if (so_far_l > 0) vst1q_u8(left_bytes  + n_left_bytes,  accum_l);
    if (so_far_r > 0) vst1q_u8(right_bytes + n_right_bytes, accum_r);
}

/* NOTE: bench_coalesce's 4-iter macro variants (coalesce_macro,
 * coalesce_macro_one_sided, half_coalesce_macro_tree) are NOT registered.
 * Their (lo,hi) 32-byte accumulator only holds 16 codes, but a 4-mask
 * (32-code) block can produce up to 32 codes on one side, so they cannot
 * partition a general bitmap byte-exactly — in the source they were pure
 * throughput probes on identity data with NO correctness check.  Verified
 * here to mismatch the scalar reference, so they're omitted (they would
 * report FAIL).  The kernels are preserved in bench_coalesce.c. */

/* p16 — 16-byte-at-a-time rank partition via two-table compaction (16b-enc.txt,
 * green-lit).  Per 16 lanes: idx = tab1[m0] | tab2[pc0][m1] (disjoint supports,
 * OR is exact), one vqtbl1q over the 16 input ranks, store 16 / advance by
 * popcount (merge-style) -- half the store count of the per-8 ctab8 scatter.
 * Left side reuses the same tables under ~mask.  ~40 KB tables (tab1 4 KB +
 * tab2 9x4 KB): NEON/big-L1 only (would bust Zen3 32 KB L1).  Reuses the
 * production masks64_neon for the mask + bitmap. */
static uint8_t pv_p16_tab1[256][16]      __attribute__((aligned(16)));
static uint8_t pv_p16_tab2[9][256][16]   __attribute__((aligned(16)));
static int pv_p16_built = 0;
static void pv_build_p16(void) {
    if (pv_p16_built) return;
    for (int m = 0; m < 256; m++) {
        memset(pv_p16_tab1[m], 0, 16);
        int p = 0;
        for (int k = 0; k < 8; k++) if ((m >> k) & 1) pv_p16_tab1[m][p++] = (uint8_t)k;
    }
    for (int pc0 = 0; pc0 <= 8; pc0++)
        for (int m1 = 0; m1 < 256; m1++) {
            memset(pv_p16_tab2[pc0][m1], 0, 16);
            int p = pc0;
            for (int k = 0; k < 8; k++)
                if ((m1 >> k) & 1) { if (p < 16) pv_p16_tab2[pc0][m1][p] = (uint8_t)(8 + k); p++; }
        }
    pv_p16_built = 1;
}
static int prim_part_p16_neon(uint8_t *ranks, int n, uint8_t thr, uint8_t *bm, uint8_t *tmp) {
    pv_build_p16();
    int n_left = 0, n_right = 0, j = 0;
    uint8x16_t vt = vdupq_n_u8(thr);
    static const uint8_t bw_a[16] = {1,2,4,8,16,32,64,128, 1,2,4,8,16,32,64,128};
    uint8x16_t bw = vld1q_u8(bw_a);
    for (; j + 64 <= n; j += 64) {
        uint8x16_t v0 = vld1q_u8(ranks + j),      v1 = vld1q_u8(ranks + j + 16);
        uint8x16_t v2 = vld1q_u8(ranks + j + 32), v3 = vld1q_u8(ranks + j + 48);
        uint64_t mask_word = masks64_neon(v0, v1, v2, v3, vt, bw);
        memcpy(bm + (j >> 3), &mask_word, 8);
        uint64_t pcw = vget_lane_u64(vreinterpret_u64_u8(vcnt_u8(vcreate_u8(mask_word))), 0);
        uint8x16_t vg[4] = { v0, v1, v2, v3 };
#define _P16(g) do {                                                            \
        uint8_t  m0 = (uint8_t)(mask_word >> (16*(g)));                          \
        uint8_t  m1 = (uint8_t)(mask_word >> (16*(g) + 8));                      \
        uint32_t pc0 = (uint32_t)((pcw >> (16*(g)))     & 0xFF);                 \
        uint32_t pc1 = (uint32_t)((pcw >> (16*(g) + 8)) & 0xFF);                 \
        uint8x16_t ri = vorrq_u8(vld1q_u8(pv_p16_tab1[m0]),                      \
                                 vld1q_u8(pv_p16_tab2[pc0][m1]));                \
        vst1q_u8(tmp + n_right, vqtbl1q_u8(vg[g], ri));                          \
        uint8x16_t li = vorrq_u8(vld1q_u8(pv_p16_tab1[(uint8_t)~m0]),            \
                                 vld1q_u8(pv_p16_tab2[8 - pc0][(uint8_t)~m1]));  \
        vst1q_u8(ranks + n_left, vqtbl1q_u8(vg[g], li));                         \
        n_right += pc0 + pc1; n_left += 16 - (pc0 + pc1);                        \
    } while (0)
        _P16(0); _P16(1); _P16(2); _P16(3);
#undef _P16
    }
    for (; j < n; j++) {
        if ((j & 7) == 0) bm[j >> 3] = 0;
        uint8_t r = ranks[j];
        if (r > thr) { bm[j >> 3] |= (uint8_t)(1u << (j & 7)); tmp[n_right++] = r; }
        else         { ranks[n_left++] = r; }
    }
    return n_right;
}
static void prim_part_p16(const ctx_t *c){ prim_part_p16_neon(c->ranks_work, c->n, c->rank_thr, c->bm, c->tmp8); }

/* asof-a3a3d19 — the per-8-chunk ctab8 COM64 partition that was production NEON
 * before p16rev was promoted (a3a3d19 = last commit with it as production; p16rev
 * superseded it on every ARM uarch).  Frozen here verbatim so the prior baseline
 * stays benchable.  Reuses the production ctab8/build_tabs/masks64_neon. */
static int prim_part_asof_a3a3d19_neon(uint8_t *ranks, int n, uint8_t thr,
                                       uint8_t *bm, uint8_t *tmp) {
    build_tabs();
    int n_left = 0, n_right = 0;
    int j = 0;
    uint8x16_t vt = vdupq_n_u8(thr);
    static const uint8_t bw_a[16] = {1,2,4,8,16,32,64,128, 1,2,4,8,16,32,64,128};
    uint8x16_t bw = vld1q_u8(bw_a);
    for (; j + 64 <= n; j += 64) {
        uint8x16_t v0 = vld1q_u8(ranks + j);
        uint8x16_t v1 = vld1q_u8(ranks + j + 16);
        uint8x16_t v2 = vld1q_u8(ranks + j + 32);
        uint8x16_t v3 = vld1q_u8(ranks + j + 48);
        uint64_t mask_word = masks64_neon(v0, v1, v2, v3, vt, bw);
        memcpy(bm + (j >> 3), &mask_word, 8);
        uint8x8_t pc_v = vcnt_u8(vcreate_u8(mask_word));
        uint64_t pc_word = vget_lane_u64(vreinterpret_u64_u8(pc_v), 0);
        uint64_t pfx = pc_word * 0x0101010101010101ULL;
        uint8x8_t cv[8] = {
            vget_low_u8(v0), vget_high_u8(v0),
            vget_low_u8(v1), vget_high_u8(v1),
            vget_low_u8(v2), vget_high_u8(v2),
            vget_low_u8(v3), vget_high_u8(v3),
        };
#define _PART_C64(K_) do {                                                      \
        uint32_t cr = (K_)==0 ? 0u : (uint32_t)((pfx >> (8*((K_)-1))) & 0xFF);   \
        uint32_t cl = 8u*(K_) - cr;                                              \
        const uint8_t *tab = ctab8[(uint8_t)(mask_word >> (8*(K_)))];            \
        uint8x8_t right = vtbl1_u8(cv[K_], vld1_u8(tab));                        \
        uint8x8_t left  = vtbl1_u8(cv[K_], vld1_u8(tab + 8));                    \
        vst1_u8(tmp   + n_right + cr, right);                                    \
        vst1_u8(ranks + n_left + cl, left);                                     \
    } while (0)
        _PART_C64(0); _PART_C64(1); _PART_C64(2); _PART_C64(3);
        _PART_C64(4); _PART_C64(5); _PART_C64(6); _PART_C64(7);
#undef _PART_C64
        uint32_t total_r = (uint32_t)(pfx >> 56);
        n_right += total_r;
        n_left += 64 - total_r;
    }
    for (; j < n; j++) {
        if ((j & 7) == 0) bm[j >> 3] = 0;
        uint8_t r = ranks[j];
        if (r > thr) { bm[j >> 3] |= (uint8_t)(1u << (j & 7)); tmp[n_right++] = r; }
        else         { ranks[n_left++] = r; }
    }
    return n_right;
}
static void prim_part_asof_a3a3d19(const ctx_t *c){ prim_part_asof_a3a3d19_neon(c->ranks_work, c->n, c->rank_thr, c->bm, c->tmp8); }

/* asof-f9974f5 — the p16rev FULL partition that was production NEON before the
 * PR #22 software pipelining (f9974f5 = last commit with it as production).
 * Frozen here verbatim: the mask crosses to a GPR via the old masks64_neon
 * (copied below, also reshaped by #22) and is vcnt'd after a vcreate
 * round-trip; nothing is started for the next group-set.  Reuses the
 * production build_tabs/p16rev_tabA/p16rev_tabB0, unchanged by #22. */
static inline uint64_t pv_masks64_asof_f9974f5_neon(uint8x16_t v0, uint8x16_t v1,
                                       uint8x16_t v2, uint8x16_t v3,
                                       uint8x16_t vt, uint8x16_t bw)
{
    uint8x16_t w0 = vandq_u8(vcgtq_u8(v0, vt), bw);   /* chunks 0,1 */
    uint8x16_t w1 = vandq_u8(vcgtq_u8(v1, vt), bw);   /* chunks 2,3 */
    uint8x16_t w2 = vandq_u8(vcgtq_u8(v2, vt), bw);   /* chunks 4,5 */
    uint8x16_t w3 = vandq_u8(vcgtq_u8(v3, vt), bw);   /* chunks 6,7 */
    uint8x16_t t0 = vpaddq_u8(w0, w1);
    uint8x16_t t1 = vpaddq_u8(w2, w3);
    uint8x16_t u0 = vpaddq_u8(t0, t1);
    uint8x16_t r  = vpaddq_u8(u0, u0);                /* low 8 bytes = mask_0..7 */
    return vget_lane_u64(vreinterpret_u64_u8(vget_low_u8(r)), 0);
}
static inline int prim_part_asof_f9974f5_neon(uint8_t *ranks, int n, uint8_t thr,
                                    uint8_t *bm, uint8_t *tmp)
{
    build_tabs();
    int n_left = 0, n_right = 0;
    int j = 0;
    uint8x16_t vt = vdupq_n_u8(thr);
    static const uint8_t bw_a[16] = {1,2,4,8,16,32,64,128, 1,2,4,8,16,32,64,128};
    uint8x16_t bw = vld1q_u8(bw_a);
    /* Right recovery: reversing the WHOLE comb register lands the top-pc reversed
     * right lanes at output [0,pc) (the [pc,16) tail is left-reversed garbage the
     * next group overwrites). */
    static const uint8_t rev16_a[16] = {15,14,13,12,11,10,9,8,7,6,5,4,3,2,1,0};
    uint8x16_t rev16 = vld1q_u8(rev16_a);
    for (; j + 64 <= n; j += 64) {
        uint8x16_t v0 = vld1q_u8(ranks + j);
        uint8x16_t v1 = vld1q_u8(ranks + j + 16);
        uint8x16_t v2 = vld1q_u8(ranks + j + 32);
        uint8x16_t v3 = vld1q_u8(ranks + j + 48);
        uint64_t mask_word = pv_masks64_asof_f9974f5_neon(v0, v1, v2, v3, vt, bw);
        memcpy(bm + (j >> 3), &mask_word, 8);
        uint64_t pcw = vget_lane_u64(vreinterpret_u64_u8(
                           vcnt_u8(vcreate_u8(mask_word))), 0);
        /* Prefix-sum the per-chunk popcounts (byte k of pfx = sum pc[0..k]) so
         * each group's store offsets (cr rights / 16g-cr lefts before it) are
         * known up front -- no serial n_left/n_right chain across the 4 groups;
         * the cursors advance once per 64.  Overlap safety is unchanged: group
         * g+1's store still starts exactly at group g's valid end, and all 4
         * input loads precede every store in program order.  (issue #5) */
        uint64_t pfx = pcw * 0x0101010101010101ULL;
        uint8x16_t vg[4] = { v0, v1, v2, v3 };
#define _PART(g) do {                                                       \
        uint8_t  m0 = (uint8_t)(mask_word >> (16*(g)));                     \
        uint8_t  m1 = (uint8_t)(mask_word >> (16*(g) + 8));                 \
        uint32_t pc0 = (uint32_t)((pcw >> (16*(g)))     & 0xFF);            \
        uint32_t cr  = (g) == 0 ? 0u                                        \
                     : (uint32_t)((pfx >> (8*(2*(g) - 1))) & 0xFF);         \
        uint8x16_t ri = vorrq_u8(vld1q_u8(p16rev_tabA[m0]),                   \
                                 vld1q_u8(&p16rev_tabB0[m1][pc0]));           \
        uint8x16_t comb = vqtbl1q_u8(vg[g], ri);                           \
        vst1q_u8(ranks + n_left + (16*(g) - cr), comb);                     \
        vst1q_u8(tmp + n_right + cr, vqtbl1q_u8(comb, rev16));              \
    } while (0)
        _PART(0); _PART(1); _PART(2); _PART(3);
#undef _PART
        uint32_t total_r = (uint32_t)(pfx >> 56);
        n_right += (int)total_r;
        n_left  += 64 - (int)total_r;
    }
    for (; j < n; j++) {
        if ((j & 7) == 0) bm[j >> 3] = 0;
        uint8_t r = ranks[j];
        if (r > thr) { bm[j >> 3] |= (uint8_t)(1u << (j & 7)); tmp[n_right++] = r; }
        else         { ranks[n_left++] = r; }
    }
    return n_right;
}
static void prim_part_asof_f9974f5(const ctx_t *c){ prim_part_asof_f9974f5_neon(c->ranks_work, c->n, c->rank_thr, c->bm, c->tmp8); }

/* p16revback — p16rev without the per-group rev shuffle.  The combined index (one OR
 * of the production p16rev_tabA/tabB) still yields {left fwd | right reversed} in
 * one vqtbl1q; the LEFT store is unchanged (comb -> ranks, forward).  But the
 * RIGHT side stores the WHOLE comb register *backwards* into a scratch buffer
 * (write at rb+w-16, then w -= pc, keeping the top pc lanes = this group's rights
 * reversed).  Processing groups in increasing position with a decreasing cursor
 * lays the rights out globally reversed; one 16-wide reverse pass after the
 * stride loop turns rb[w..] into the forward right output in tmp.
 * Hot loop: 2 vld + 1 orr + 1 vqtbl1q + 2 st per group (vs p16rev's extra rev vld +
 * vqtbl1q on the serial path) — at the cost of the extra streaming reverse pass.
 * Reuses the production p16rev_tabA/tabB (in scope) + a realloc'd scratch buffer. */
static uint8_t *pv_p16revback_buf = NULL;
static int      pv_p16revback_cap = 0;
static int prim_part_p16revback_neon(uint8_t *ranks, int n, uint8_t thr, uint8_t *bm, uint8_t *tmp) {
    build_tabs();                         /* ensures p16rev_tabA/tabB are built */
    if (n + 64 > pv_p16revback_cap) {
        pv_p16revback_buf = (uint8_t *)realloc(pv_p16revback_buf, (size_t)n + 64);
        pv_p16revback_cap = n + 64;
    }
    uint8_t *rb = pv_p16revback_buf;
    int n_left = 0, j = 0, w = n + 16;    /* backward right cursor (high end, exclusive) */
    uint8x16_t vt = vdupq_n_u8(thr);
    static const uint8_t bw_a[16] = {1,2,4,8,16,32,64,128, 1,2,4,8,16,32,64,128};
    uint8x16_t bw = vld1q_u8(bw_a);
    for (; j + 64 <= n; j += 64) {
        uint8x16_t v0 = vld1q_u8(ranks + j),      v1 = vld1q_u8(ranks + j + 16);
        uint8x16_t v2 = vld1q_u8(ranks + j + 32), v3 = vld1q_u8(ranks + j + 48);
        uint64_t mask_word = masks64_neon(v0, v1, v2, v3, vt, bw);
        memcpy(bm + (j >> 3), &mask_word, 8);
        uint64_t pcw = vget_lane_u64(vreinterpret_u64_u8(vcnt_u8(vcreate_u8(mask_word))), 0);
        uint8x16_t vg[4] = { v0, v1, v2, v3 };
#define _P16REVBACK(g) do {                                                        \
        uint8_t  m0 = (uint8_t)(mask_word >> (16*(g)));                         \
        uint8_t  m1 = (uint8_t)(mask_word >> (16*(g) + 8));                     \
        uint32_t pc0 = (uint32_t)((pcw >> (16*(g)))     & 0xFF);                \
        uint32_t pc1 = (uint32_t)((pcw >> (16*(g) + 8)) & 0xFF);                \
        uint32_t pc  = pc0 + pc1;                                               \
        uint8x16_t ri = vorrq_u8(vld1q_u8(p16rev_tabA[m0]),                       \
                                 vld1q_u8(&p16rev_tabB0[m1][pc0]));               \
        uint8x16_t comb = vqtbl1q_u8(vg[g], ri);                               \
        vst1q_u8(ranks + n_left, comb);                                        \
        vst1q_u8(rb + w - 16, comb);                                           \
        w -= pc; n_left += 16 - pc;                                             \
    } while (0)
        _P16REVBACK(0); _P16REVBACK(1); _P16REVBACK(2); _P16REVBACK(3);
#undef _P16REVBACK
    }
    int n_right = (n + 16) - w;           /* rights produced by the stride loop */
    for (; j < n; j++) {                  /* scalar tail appends rights forward */
        if ((j & 7) == 0) bm[j >> 3] = 0;
        uint8_t r = ranks[j];
        if (r > thr) { bm[j >> 3] |= (uint8_t)(1u << (j & 7)); tmp[n_right++] = r; }
        else         { ranks[n_left++] = r; }
    }
    /* reverse rb[w .. w+T) into tmp[0 .. T), 16 bytes/iter via a fixed shuffle. */
    static const uint8_t REV16_a[16] = {15,14,13,12,11,10,9,8,7,6,5,4,3,2,1,0};
    uint8x16_t rev16 = vld1q_u8(REV16_a);
    int T = (n + 16) - w, i = 0;
    const uint8_t *src = rb + w;
    for (; i + 64 <= T; i += 64) {
        vst1q_u8(tmp + (T - i - 16), vqtbl1q_u8(vld1q_u8(src + i),      rev16));
        vst1q_u8(tmp + (T - i - 32), vqtbl1q_u8(vld1q_u8(src + i + 16), rev16));
        vst1q_u8(tmp + (T - i - 48), vqtbl1q_u8(vld1q_u8(src + i + 32), rev16));
        vst1q_u8(tmp + (T - i - 64), vqtbl1q_u8(vld1q_u8(src + i + 48), rev16));
    }
    for (; i + 16 <= T; i += 16)
        vst1q_u8(tmp + (T - i - 16), vqtbl1q_u8(vld1q_u8(src + i), rev16));
    for (; i < T; i++) tmp[T - 1 - i] = src[i];
    return n_right;
}
static void prim_part_p16revback(const ctx_t *c){ prim_part_p16revback_neon(c->ranks_work, c->n, c->rank_thr, c->bm, c->tmp8); }

/* p16rev — the classic p16rev cursor scheme: serial per-group n_left/n_right
 * chain (each group's store addresses wait on the previous group's popcount),
 * kept benchable as the baseline for the issue-#5 pfx-sum cursors that replaced
 * it in production.  Serial cursors were production from the p16rev promotion
 * (3a138a6) through 4d93965.  Uses the current 8 KB tabB0 offset load (that
 * span's production paired these cursors with the 36 KB tabB), so a same-binary
 * A/B vs production isolates the CURSOR scheme alone. */
static int prim_part_p16rev_neon(uint8_t *ranks, int n, uint8_t thr,
                                 uint8_t *bm, uint8_t *tmp)
{
    build_tabs();
    int n_left = 0, n_right = 0;
    int j = 0;
    uint8x16_t vt = vdupq_n_u8(thr);
    static const uint8_t bw_a[16] = {1,2,4,8,16,32,64,128, 1,2,4,8,16,32,64,128};
    uint8x16_t bw = vld1q_u8(bw_a);
    static const uint8_t rev16_a[16] = {15,14,13,12,11,10,9,8,7,6,5,4,3,2,1,0};
    uint8x16_t rev16 = vld1q_u8(rev16_a);
    for (; j + 64 <= n; j += 64) {
        uint8x16_t v0 = vld1q_u8(ranks + j);
        uint8x16_t v1 = vld1q_u8(ranks + j + 16);
        uint8x16_t v2 = vld1q_u8(ranks + j + 32);
        uint8x16_t v3 = vld1q_u8(ranks + j + 48);
        uint64_t mask_word = masks64_neon(v0, v1, v2, v3, vt, bw);
        memcpy(bm + (j >> 3), &mask_word, 8);
        uint64_t pcw = vget_lane_u64(vreinterpret_u64_u8(
                           vcnt_u8(vcreate_u8(mask_word))), 0);
        uint8x16_t vg[4] = { v0, v1, v2, v3 };
#define _PARTSER(g) do {                                                     \
        uint8_t  m0 = (uint8_t)(mask_word >> (16*(g)));                     \
        uint8_t  m1 = (uint8_t)(mask_word >> (16*(g) + 8));                 \
        uint32_t pc0 = (uint32_t)((pcw >> (16*(g)))     & 0xFF);            \
        uint32_t pc1 = (uint32_t)((pcw >> (16*(g) + 8)) & 0xFF);            \
        uint32_t pc  = pc0 + pc1;                                           \
        uint8x16_t ri = vorrq_u8(vld1q_u8(p16rev_tabA[m0]),                   \
                                 vld1q_u8(&p16rev_tabB0[m1][pc0]));           \
        uint8x16_t comb = vqtbl1q_u8(vg[g], ri);                           \
        vst1q_u8(ranks + n_left, comb);                                    \
        vst1q_u8(tmp + n_right, vqtbl1q_u8(comb, rev16));                   \
        n_right += pc; n_left += 16 - pc;                                   \
    } while (0)
        _PARTSER(0); _PARTSER(1); _PARTSER(2); _PARTSER(3);
#undef _PARTSER
    }
    for (; j < n; j++) {
        if ((j & 7) == 0) bm[j >> 3] = 0;
        uint8_t r = ranks[j];
        if (r > thr) { bm[j >> 3] |= (uint8_t)(1u << (j & 7)); tmp[n_right++] = r; }
        else         { ranks[n_left++] = r; }
    }
    return n_right;
}
static void prim_part_p16rev(const ctx_t *c){ prim_part_p16rev_neon(c->ranks_work, c->n, c->rank_thr, c->bm, c->tmp8); }

/* right16 — 16-wide one-sided (right) rank compaction for part_core.  Production
 * part_core_neon compacts per-8-chunk: 2 vtbl1_u8 + 2 eight-byte stores per 16
 * lanes (one side).  This does ONE vqtbl1q + ONE 16-byte store per 16 lanes via
 * the p16 right-pack two-table index (ri = tab1[m0] | tab2[pc0][m1], reused from
 * the p16 variant), halving both the shuffle and the store-instruction count on
 * the store-bound loop.  Cost: the 36 KB tab2 latency (the same that sank the
 * both-sided p16).  Right-only (the common HALF-node case); reads ranks, writes
 * bm + tmp.  16-byte tmp overstore absorbed by the buffer's tail slack. */
static int prim_part_right16_neon(uint8_t *ranks, int n, uint8_t thr, uint8_t *bm, uint8_t *tmp) {
    pv_build_p16();
    int n_right = 0, j = 0;
    uint8x16_t vt = vdupq_n_u8(thr);
    static const uint8_t bw_a[16] = {1,2,4,8,16,32,64,128, 1,2,4,8,16,32,64,128};
    uint8x16_t bw = vld1q_u8(bw_a);
    for (; j + 64 <= n; j += 64) {
        uint8x16_t v0 = vld1q_u8(ranks + j),      v1 = vld1q_u8(ranks + j + 16);
        uint8x16_t v2 = vld1q_u8(ranks + j + 32), v3 = vld1q_u8(ranks + j + 48);
        uint64_t mask_word = masks64_neon(v0, v1, v2, v3, vt, bw);
        memcpy(bm + (j >> 3), &mask_word, 8);
        uint64_t pcw = vget_lane_u64(vreinterpret_u64_u8(vcnt_u8(vcreate_u8(mask_word))), 0);
        uint8x16_t vg[4] = { v0, v1, v2, v3 };
#define _R16(g) do {                                                            \
        uint8_t  m0 = (uint8_t)(mask_word >> (16*(g)));                         \
        uint8_t  m1 = (uint8_t)(mask_word >> (16*(g) + 8));                     \
        uint32_t pc0 = (uint32_t)((pcw >> (16*(g)))     & 0xFF);                \
        uint32_t pc1 = (uint32_t)((pcw >> (16*(g) + 8)) & 0xFF);                \
        uint8x16_t ri = vorrq_u8(vld1q_u8(pv_p16_tab1[m0]),                     \
                                 vld1q_u8(pv_p16_tab2[pc0][m1]));               \
        vst1q_u8(tmp + n_right, vqtbl1q_u8(vg[g], ri));                         \
        n_right += pc0 + pc1;                                                   \
    } while (0)
        _R16(0); _R16(1); _R16(2); _R16(3);
#undef _R16
    }
    for (; j < n; j++) {
        if ((j & 7) == 0) bm[j >> 3] = 0;
        uint8_t r = ranks[j];
        if (r > thr) { bm[j >> 3] |= (uint8_t)(1u << (j & 7)); tmp[n_right++] = r; }
    }
    return n_right;
}
static void prim_part_right16(const ctx_t *c){ prim_part_right16_neon(c->ranks_work, c->n, c->rank_thr, c->bm, c->tmp8); }


/* ============================================================================
 * enc_partition_full : ryg-cshuf — ryg's "second way" half-combine (2026-07)
 *   OUR INTERPRETATION of the ryg email thread (partition ideas, licensed),
 *   not his verbatim code.
 *   Production p16rev combines the two per-8 halves with one unaligned 16B
 *   load from the 32B-padded p16rev_tabB0 row at +pc0 (never line-crossing
 *   by construction).  The B half's payload is only row bytes [8,16) (lefts
 *   at [8,8+lc1), reversed rights at (15-pc1,15]), so this variant keeps a
 *   256x8 table (2KB vs 8KB), loads it ALIGNED into a low half, and shifts
 *   it into place with a computed control {i + pc0 - 8}: out-of-range
 *   controls TBL-zero natively on NEON.  Trades the unaligned load for
 *   dup + add + one extra tbl per group.  M4 unaligned loads measured free,
 *   so expected ~wash here; the x86 twin is the real question.
 * ========================================================================== */
static uint8_t pv_rygb8n[256][8] __attribute__((aligned(64)));
static int pv_rygb8n_built = 0;
static void pv_build_rygb8n(void) {
    if (pv_rygb8n_built) return;
    build_tabs();
    for (int m = 0; m < 256; m++) memcpy(pv_rygb8n[m], &p16rev_tabB0[m][8], 8);
    pv_rygb8n_built = 1;
}
static inline int pv_part_rygcshuf_neon(uint8_t *ranks, int n, uint8_t thr,
                                        uint8_t *bm, uint8_t *tmp)
{
    pv_build_rygb8n();
    int n_left = 0, n_right = 0;
    int j = 0;
    uint8x16_t vt = vdupq_n_u8(thr);
    static const uint8_t bw_a[16] = {1,2,4,8,16,32,64,128, 1,2,4,8,16,32,64,128};
    uint8x16_t bw = vld1q_u8(bw_a);
    static const uint8_t rev16_a[16] = {15,14,13,12,11,10,9,8,7,6,5,4,3,2,1,0};
    uint8x16_t rev16 = vld1q_u8(rev16_a);
    static const uint8_t iota_a[16] = {0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15};
    uint8x16_t iota = vld1q_u8(iota_a);
    for (; j + 64 <= n; j += 64) {
        uint8x16_t v0 = vld1q_u8(ranks + j);
        uint8x16_t v1 = vld1q_u8(ranks + j + 16);
        uint8x16_t v2 = vld1q_u8(ranks + j + 32);
        uint8x16_t v3 = vld1q_u8(ranks + j + 48);
        uint64_t mask_word = masks64_neon(v0, v1, v2, v3, vt, bw);
        memcpy(bm + (j >> 3), &mask_word, 8);
        uint64_t pcw = vget_lane_u64(vreinterpret_u64_u8(
                           vcnt_u8(vcreate_u8(mask_word))), 0);
        uint64_t pfx = pcw * 0x0101010101010101ULL;
        uint8x16_t vg[4] = { v0, v1, v2, v3 };
#define _PRYGN(g) do {                                                      \
        uint8_t  m0 = (uint8_t)(mask_word >> (16*(g)));                     \
        uint8_t  m1 = (uint8_t)(mask_word >> (16*(g) + 8));                 \
        uint32_t pc0 = (uint32_t)((pcw >> (16*(g)))     & 0xFF);            \
        uint32_t cr  = (g) == 0 ? 0u                                        \
                     : (uint32_t)((pfx >> (8*(2*(g) - 1))) & 0xFF);         \
        uint8x16_t b8 = vcombine_u8(vld1_u8(pv_rygb8n[m1]), vdup_n_u8(0));  \
        uint8x16_t sh = vaddq_u8(iota, vdupq_n_u8((uint8_t)(pc0 - 8)));     \
        uint8x16_t ri = vorrq_u8(vld1q_u8(p16rev_tabA[m0]),                 \
                                 vqtbl1q_u8(b8, sh));                       \
        uint8x16_t comb = vqtbl1q_u8(vg[g], ri);                            \
        vst1q_u8(ranks + n_left + (16*(g) - cr), comb);                     \
        vst1q_u8(tmp + n_right + cr, vqtbl1q_u8(comb, rev16));              \
    } while (0)
        _PRYGN(0); _PRYGN(1); _PRYGN(2); _PRYGN(3);
#undef _PRYGN
        uint32_t total_r = (uint32_t)(pfx >> 56);
        n_right += (int)total_r;
        n_left  += 64 - (int)total_r;
    }
    for (; j < n; j++) {
        if ((j & 7) == 0) bm[j >> 3] = 0;
        uint8_t r = ranks[j];
        if (r > thr) { bm[j >> 3] |= (uint8_t)(1u << (j & 7)); tmp[n_right++] = r; }
        else         { ranks[n_left++] = r; }
    }
    return n_right;
}
static void prim_part_rygcshuf(const ctx_t *c){ pv_part_rygcshuf_neon(c->ranks_work, c->n, c->rank_thr, c->bm, c->tmp8); }


/* ============================================================================
 * enc_partition_full : ryg-gaptab — ryg's shared-gap B table (2026-07)
 *   OUR INTERPRETATION of the ryg email thread (partition ideas, licensed),
 *   not his verbatim code.
 *   Production's combine unchanged (unaligned 16B window at +pc0, OR, one
 *   tbl); only the table shrinks: the 8-byte payloads (row bytes [8,16) of
 *   p16rev_tabB0) are packed at STRIDE 16 with the leading 8-zero gap of
 *   entry m+1 doubling as the trailing zeros of entry m (window reads at
 *   most byte m*16+23 < next payload at m*16+24).  256*16+8 bytes: 4KB vs
 *   8KB, total partition LUT 12KB -> 8KB.  Cost: rows lose the 32B
 *   alignment guarantee, so a fraction of B-loads cross a cache line
 *   (~1/8 of rows on M4's 128B lines). */
static uint8_t pv_ryggapn[256*16 + 16] __attribute__((aligned(64)));
static int pv_ryggapn_built = 0;
static void pv_build_ryggapn(void) {
    if (pv_ryggapn_built) return;
    build_tabs();
    memset(pv_ryggapn, 0, sizeof pv_ryggapn);
    for (int m = 0; m < 256; m++)
        memcpy(pv_ryggapn + m*16 + 8, &p16rev_tabB0[m][8], 8);
    pv_ryggapn_built = 1;
}
static inline int pv_part_ryggap_neon(uint8_t *ranks, int n, uint8_t thr,
                                      uint8_t *bm, uint8_t *tmp)
{
    pv_build_ryggapn();
    int n_left = 0, n_right = 0;
    int j = 0;
    uint8x16_t vt = vdupq_n_u8(thr);
    static const uint8_t bw_a[16] = {1,2,4,8,16,32,64,128, 1,2,4,8,16,32,64,128};
    uint8x16_t bw = vld1q_u8(bw_a);
    static const uint8_t rev16_a[16] = {15,14,13,12,11,10,9,8,7,6,5,4,3,2,1,0};
    uint8x16_t rev16 = vld1q_u8(rev16_a);
    for (; j + 64 <= n; j += 64) {
        uint8x16_t v0 = vld1q_u8(ranks + j);
        uint8x16_t v1 = vld1q_u8(ranks + j + 16);
        uint8x16_t v2 = vld1q_u8(ranks + j + 32);
        uint8x16_t v3 = vld1q_u8(ranks + j + 48);
        uint64_t mask_word = masks64_neon(v0, v1, v2, v3, vt, bw);
        memcpy(bm + (j >> 3), &mask_word, 8);
        uint64_t pcw = vget_lane_u64(vreinterpret_u64_u8(
                           vcnt_u8(vcreate_u8(mask_word))), 0);
        uint64_t pfx = pcw * 0x0101010101010101ULL;
        uint8x16_t vg[4] = { v0, v1, v2, v3 };
#define _PGAPN(g) do {                                                      \
        uint8_t  m0 = (uint8_t)(mask_word >> (16*(g)));                     \
        uint8_t  m1 = (uint8_t)(mask_word >> (16*(g) + 8));                 \
        uint32_t pc0 = (uint32_t)((pcw >> (16*(g)))     & 0xFF);            \
        uint32_t cr  = (g) == 0 ? 0u                                        \
                     : (uint32_t)((pfx >> (8*(2*(g) - 1))) & 0xFF);         \
        uint8x16_t ri = vorrq_u8(vld1q_u8(p16rev_tabA[m0]),                 \
                                 vld1q_u8(pv_ryggapn + (uint32_t)m1*16 + pc0)); \
        uint8x16_t comb = vqtbl1q_u8(vg[g], ri);                            \
        vst1q_u8(ranks + n_left + (16*(g) - cr), comb);                     \
        vst1q_u8(tmp + n_right + cr, vqtbl1q_u8(comb, rev16));              \
    } while (0)
        _PGAPN(0); _PGAPN(1); _PGAPN(2); _PGAPN(3);
#undef _PGAPN
        uint32_t total_r = (uint32_t)(pfx >> 56);
        n_right += (int)total_r;
        n_left  += 64 - (int)total_r;
    }
    for (; j < n; j++) {
        if ((j & 7) == 0) bm[j >> 3] = 0;
        uint8_t r = ranks[j];
        if (r > thr) { bm[j >> 3] |= (uint8_t)(1u << (j & 7)); tmp[n_right++] = r; }
        else         { ranks[n_left++] = r; }
    }
    return n_right;
}
static void prim_part_ryggap(const ctx_t *c){ pv_part_ryggap_neon(c->ranks_work, c->n, c->rank_thr, c->bm, c->tmp8); }

#endif /* USE_NEON_KERNELS */

/* ============================================================================
 * x86 (SSE4.1 / AVX2) partition variants — extracted verbatim from
 * extras/bench/bench_partition_{x86,unroll,avx2}.c.  They reuse the production
 * compress_tab / compress_popcnt / pv_enc_mask8_x86 (in scope here, built
 * by prim_codec_init).  Same (codes_la,n,depth,bm,right_out) -> n_right contract
 * as the production partition; the ctx_t adapter maps that onto la_work/tmp16.
 * ========================================================================== */
#if defined(__SSE4_1__) && !defined(__AVX512VBMI2__)

/* Shared SWAR per-byte popcount helper (each output byte = popcount of the
 * corresponding input byte).  Defined once here; the merge x86 section reuses
 * it via the same #ifndef guard. */
#ifndef PV_X86_POPCNT_BYTES_U64
#define PV_X86_POPCNT_BYTES_U64 1
static inline uint64_t pv_popcnt_bytes_u64(uint64_t x) {
    x = x - ((x >> 1) & 0x5555555555555555ULL);
    x = (x & 0x3333333333333333ULL) + ((x >> 2) & 0x3333333333333333ULL);
    x = (x + (x >> 4)) & 0x0f0f0f0f0f0f0f0fULL;
    return x;
}
#endif

/* Self-contained 8-code partition-mask: left-shift each u16 by `depth` so the
 * partition bit (15-depth) lands in the MSB, signed-saturate-pack to bytes,
 * movemask -> 8-bit mask.  Local copy so the graveyard doesn't depend on the
 * production x86-backend pv_enc_mask8_x86 (absent on the AVX-512 build). */
static inline uint8_t pv_enc_mask8_x86(__m128i code_vec, __m128i shift_count) {
    __m128i shifted = _mm_sll_epi16(code_vec, shift_count);
    __m128i bytes   = _mm_packs_epi16(shifted, _mm_setzero_si128());
    return (uint8_t)_mm_movemask_epi8(bytes);
}

/* sse_com: 64 codes/iter, 8 chunks, prefix-sum cursors (no compress_popcnt
 * load, no serial cursor).  bench_partition_x86.c::com_partition. */
static inline int prim_part_full_com_x86(uint16_t *codes_la, int n, int depth,
                                         uint8_t *bm, uint16_t *right_out) {
    int n_left = 0, n_right = 0, j = 0;
    __m128i shift_count = _mm_cvtsi32_si128(depth);
    for (; j + 64 <= n; j += 64) {
        __m128i c0=_mm_loadu_si128((const __m128i*)(codes_la+j)),    c1=_mm_loadu_si128((const __m128i*)(codes_la+j+8)),
                c2=_mm_loadu_si128((const __m128i*)(codes_la+j+16)), c3=_mm_loadu_si128((const __m128i*)(codes_la+j+24)),
                c4=_mm_loadu_si128((const __m128i*)(codes_la+j+32)), c5=_mm_loadu_si128((const __m128i*)(codes_la+j+40)),
                c6=_mm_loadu_si128((const __m128i*)(codes_la+j+48)), c7=_mm_loadu_si128((const __m128i*)(codes_la+j+56));
        uint64_t mask_word = (uint64_t)pv_enc_mask8_x86(c0,shift_count)
            | ((uint64_t)pv_enc_mask8_x86(c1,shift_count)<<8)  | ((uint64_t)pv_enc_mask8_x86(c2,shift_count)<<16)
            | ((uint64_t)pv_enc_mask8_x86(c3,shift_count)<<24) | ((uint64_t)pv_enc_mask8_x86(c4,shift_count)<<32)
            | ((uint64_t)pv_enc_mask8_x86(c5,shift_count)<<40) | ((uint64_t)pv_enc_mask8_x86(c6,shift_count)<<48)
            | ((uint64_t)pv_enc_mask8_x86(c7,shift_count)<<56);
        memcpy(bm + (j>>3), &mask_word, 8);
        uint64_t pfx = pv_popcnt_bytes_u64(mask_word) * 0x0101010101010101ULL;
        #define PV_PC(K_, CV) do {                                              \
            uint8_t M = (uint8_t)(mask_word >> (8*(K_)));                        \
            uint32_t cr = (K_)==0 ? 0u : (uint32_t)((pfx >> (8*((K_)-1))) & 0xFF);\
            uint32_t cl = 8u*(K_) - cr;                                         \
            const uint8_t *t = compress_tab[M];                                \
            _mm_storeu_si128((__m128i*)(right_out + n_right + cr),             \
                             _mm_shuffle_epi8(CV,_mm_load_si128((const __m128i*)t)));\
            _mm_storeu_si128((__m128i*)(codes_la + n_left + cl),               \
                             _mm_shuffle_epi8(CV,_mm_load_si128((const __m128i*)(t+16))));\
        } while (0)
        PV_PC(0,c0); PV_PC(1,c1); PV_PC(2,c2); PV_PC(3,c3); PV_PC(4,c4); PV_PC(5,c5); PV_PC(6,c6); PV_PC(7,c7);
        #undef PV_PC
        uint32_t total_r = (uint32_t)(pfx >> 56);
        n_right += total_r; n_left += 64 - total_r;
    }
    int shift_d = 15 - depth;
    for (; j < n; j++){ uint16_t c=codes_la[j]; if((c>>shift_d)&1) right_out[n_right++]=c; else codes_la[n_left++]=c; }
    return n_right;
}
static void prim_part_full_com(const ctx_t *c){
    prim_part_full_com_x86(c->la_work, c->n, c->depth, c->bm, c->tmp16);
}

/* u16-like: the per-8-chunk u8 rank partition (mirrors the u16 partition
 * shape) — one movemask + one storel per 8-rank chunk, 2x-unrolled — to A/B
 * against the shipped 16-wide + single-movemask part_full_x86.  Uses the
 * production x86 rank tables (x86_ctab_r/l, x86_pc8, x86_mask8); 8-byte storel
 * keeps the in-place left write off the next iter. */
static void prim_part_u16like(const ctx_t *c) {
    uint8_t *ranks = c->ranks_work, *bm = c->bm, *tmp = c->tmp8;
    int n = c->n; uint8_t thr = c->rank_thr;
    x86_build_tabs();
    int n_left = 0, n_right = 0, j = 0;
    __m128i thr1 = _mm_set1_epi8((char)(thr + 1));
    for (; j + 16 <= n; j += 16) {
        __m128i v  = _mm_loadu_si128((const __m128i *)(ranks + j));
        __m128i v1 = _mm_unpackhi_epi64(v, v);
        uint8_t m0 = x86_mask8(v,  thr1);
        uint8_t m1 = x86_mask8(v1, thr1);
        bm[j >> 3] = m0; bm[(j >> 3) + 1] = m1;
        int nr0 = x86_pc8[m0], nr1 = x86_pc8[m1];
        _mm_storel_epi64((__m128i *)(tmp + n_right),
            _mm_shuffle_epi8(v,  _mm_load_si128((const __m128i *)x86_ctab_r[m0])));
        _mm_storel_epi64((__m128i *)(tmp + n_right + nr0),
            _mm_shuffle_epi8(v1, _mm_load_si128((const __m128i *)x86_ctab_r[m1])));
        _mm_storel_epi64((__m128i *)(ranks + n_left),
            _mm_shuffle_epi8(v,  _mm_load_si128((const __m128i *)x86_ctab_l[m0])));
        _mm_storel_epi64((__m128i *)(ranks + n_left + (8 - nr0)),
            _mm_shuffle_epi8(v1, _mm_load_si128((const __m128i *)x86_ctab_l[m1])));
        n_right += nr0 + nr1; n_left += (8 - nr0) + (8 - nr1);
    }
    for (; j < n; j++) {
        if ((j & 7) == 0) bm[j >> 3] = 0;
        uint8_t r = ranks[j];
        if (r > thr) { bm[j >> 3] |= (uint8_t)(1u << (j & 7)); tmp[n_right++] = r; }
        else         { ranks[n_left++] = r; }
    }
}

/* asof-3a138a6: the 32 ranks/iter dense X86_COMPACT16 partition that was
 * production x86 (part_full_x86) before p16rev was promoted — two SSE movemasks
 * OR'd into a 32-bit mask, one 4-byte bitmap write, two X86_COMPACT16 (16-wide,
 * two min-merged indices) per iter, POPCNT cursor advance.  Frozen here verbatim
 * as the baseline; reuses the production x86_ctab_r/l + x86_pre_r/l (80 KB). */
static void prim_part_asof_3a138a6_x86(const ctx_t *c) {
    uint8_t *ranks = c->ranks_work, *bm = c->bm, *tmp = c->tmp8;
    int n = c->n; uint8_t thr = c->rank_thr;
    x86_build_tabs();
    int n_left = 0, n_right = 0, j = 0;
    __m128i thr1 = _mm_set1_epi8((char)(thr + 1));
    for (; j + 32 <= n; j += 32) {
        __m128i v0 = _mm_loadu_si128((const __m128i *)(ranks + j));
        __m128i v1 = _mm_loadu_si128((const __m128i *)(ranks + j + 16));
        uint32_t mlo = (uint16_t)_mm_movemask_epi8(
            _mm_cmpeq_epi8(_mm_min_epu8(v0, thr1), thr1));
        uint32_t mhi = (uint16_t)_mm_movemask_epi8(
            _mm_cmpeq_epi8(_mm_min_epu8(v1, thr1), thr1));
        uint32_t mm = mlo | (mhi << 16);
        memcpy(bm + (j >> 3), &mm, 4);
        int r0 = x86_pc8[(uint8_t)mlo];
        X86_COMPACT16(v0, (uint8_t)mlo, (uint8_t)(mlo >> 8), r0, n_left, n_right);
        int nr01 = __builtin_popcount(mlo);
        n_right += nr01; n_left += 16 - nr01;
        int r2 = x86_pc8[(uint8_t)mhi];
        X86_COMPACT16(v1, (uint8_t)mhi, (uint8_t)(mhi >> 8), r2, n_left, n_right);
        int nr23 = __builtin_popcount(mhi);
        n_right += nr23; n_left += 16 - nr23;
    }
    for (; j + 16 <= n; j += 16) {
        __m128i v = _mm_loadu_si128((const __m128i *)(ranks + j));
        uint16_t mm = (uint16_t)_mm_movemask_epi8(
            _mm_cmpeq_epi8(_mm_min_epu8(v, thr1), thr1));
        memcpy(bm + (j >> 3), &mm, 2);
        int rlo = x86_pc8[(uint8_t)mm], rhi = x86_pc8[(uint8_t)(mm >> 8)];
        X86_COMPACT16(v, (uint8_t)mm, (uint8_t)(mm >> 8), rlo, n_left, n_right);
        n_right += rlo + rhi; n_left += 16 - rlo - rhi;
    }
    for (; j < n; j++) {
        if ((j & 7) == 0) bm[j >> 3] = 0;
        uint8_t r = ranks[j];
        if (r > thr) { bm[j >> 3] |= (uint8_t)(1u << (j & 7)); tmp[n_right++] = r; }
        else         { ranks[n_left++] = r; }
    }
}

/* halftab: sse32 but drop ctab_l/pre_l — left = right under ~mask
 * (ctab_l[m]==ctab_r[~m], pre_l[llo][m]==pre_r[llo][~m]).  40 KB tables vs 80 KB,
 * same SIMD op count (+2 scalar NOT), better L1 reuse (both sides share one set). */
#define X86_COMPACT16_HALF(v, mlo, mhi, rlo, ldst, rdst)                        \
    do {                                                                        \
        __m128i ridx_ = _mm_min_epu8(                                           \
            _mm_load_si128((const __m128i *)x86_ctab_r[mlo]),                   \
            _mm_load_si128((const __m128i *)x86_pre_r[rlo][mhi]));              \
        _mm_storeu_si128((__m128i *)(tmp + (rdst)), _mm_shuffle_epi8((v), ridx_)); \
        unsigned nmlo_ = (uint8_t)~(unsigned)(mlo);                             \
        unsigned nmhi_ = (uint8_t)~(unsigned)(mhi);                             \
        int      llo_  = 8 - (rlo);                                             \
        __m128i lidx_ = _mm_min_epu8(                                           \
            _mm_load_si128((const __m128i *)x86_ctab_r[nmlo_]),                 \
            _mm_load_si128((const __m128i *)x86_pre_r[llo_][nmhi_]));           \
        _mm_storeu_si128((__m128i *)(ranks + (ldst)), _mm_shuffle_epi8((v), lidx_)); \
    } while (0)
static void prim_part_halftab(const ctx_t *c) {
    uint8_t *ranks = c->ranks_work, *bm = c->bm, *tmp = c->tmp8;
    int n = c->n; uint8_t thr = c->rank_thr;
    x86_build_tabs();
    int n_left = 0, n_right = 0, j = 0;
    __m128i thr1 = _mm_set1_epi8((char)(thr + 1));
    for (; j + 32 <= n; j += 32) {
        __m128i v0 = _mm_loadu_si128((const __m128i *)(ranks + j));
        __m128i v1 = _mm_loadu_si128((const __m128i *)(ranks + j + 16));
        uint32_t mlo = (uint16_t)_mm_movemask_epi8(_mm_cmpeq_epi8(_mm_min_epu8(v0, thr1), thr1));
        uint32_t mhi = (uint16_t)_mm_movemask_epi8(_mm_cmpeq_epi8(_mm_min_epu8(v1, thr1), thr1));
        uint32_t mm = mlo | (mhi << 16);
        memcpy(bm + (j >> 3), &mm, 4);
        int r0 = x86_pc8[(uint8_t)mlo];
        X86_COMPACT16_HALF(v0, (uint8_t)mlo, (uint8_t)(mlo >> 8), r0, n_left, n_right);
        int nr01 = __builtin_popcount(mlo);
        n_right += nr01; n_left += 16 - nr01;
        int r2 = x86_pc8[(uint8_t)mhi];
        X86_COMPACT16_HALF(v1, (uint8_t)mhi, (uint8_t)(mhi >> 8), r2, n_left, n_right);
        int nr23 = __builtin_popcount(mhi);
        n_right += nr23; n_left += 16 - nr23;
    }
    for (; j + 16 <= n; j += 16) {
        __m128i v = _mm_loadu_si128((const __m128i *)(ranks + j));
        uint16_t mm = (uint16_t)_mm_movemask_epi8(_mm_cmpeq_epi8(_mm_min_epu8(v, thr1), thr1));
        memcpy(bm + (j >> 3), &mm, 2);
        int rlo = x86_pc8[(uint8_t)mm], rhi = x86_pc8[(uint8_t)(mm >> 8)];
        X86_COMPACT16_HALF(v, (uint8_t)mm, (uint8_t)(mm >> 8), rlo, n_left, n_right);
        n_right += rlo + rhi; n_left += 16 - rlo - rhi;
    }
    for (; j < n; j++) {
        if ((j & 7) == 0) bm[j >> 3] = 0;
        uint8_t r = ranks[j];
        if (r > thr) { bm[j >> 3] |= (uint8_t)(1u << (j & 7)); tmp[n_right++] = r; }
        else         { ranks[n_left++] = r; }
    }
}

/* p16rev combined-index LUTs (SSE), used by the p16revback variant below.
 * tabA[m0] | tabB[pc0][m1] packs {left, forward, front} | {right, reversed,
 * back} (the two sides tile the 16 lanes, so the OR of two disjoint tables is
 * exact).  The p16rev kernel itself is now production (part_full_x86 in
 * pivco_huffman_primitives_x86.h, with its own copy of these tables), so only
 * the backward-store p16revback variant carries them here. */
static uint8_t pv_x86p16rev_tabA[256][16]    __attribute__((aligned(16)));
static uint8_t pv_x86p16rev_tabB[9][256][16] __attribute__((aligned(16)));
static uint8_t pv_x86p16rev_rev[17][16]      __attribute__((aligned(16)));
static int pv_x86p16rev_built = 0;
static void pv_build_x86p16rev(void) {
    if (pv_x86p16rev_built) return;
    for (int m0 = 0; m0 < 256; m0++) {
        memset(pv_x86p16rev_tabA[m0], 0, 16);
        int lp = 0, rp = 15;
        for (int k = 0; k < 8; k++) {
            if ((m0 >> k) & 1) pv_x86p16rev_tabA[m0][rp--] = (uint8_t)k;
            else               pv_x86p16rev_tabA[m0][lp++] = (uint8_t)k;
        }
    }
    for (int pc0 = 0; pc0 <= 8; pc0++)
        for (int m1 = 0; m1 < 256; m1++) {
            memset(pv_x86p16rev_tabB[pc0][m1], 0, 16);
            int lp = 8 - pc0, rp = 15 - pc0;
            for (int k = 0; k < 8; k++) {
                if ((m1 >> k) & 1) pv_x86p16rev_tabB[pc0][m1][rp--] = (uint8_t)(8 + k);
                else               pv_x86p16rev_tabB[pc0][m1][lp++] = (uint8_t)(8 + k);
            }
        }
    for (int pc = 0; pc <= 16; pc++) {
        memset(pv_x86p16rev_rev[pc], 0x80, 16);   /* tail lanes -> pshufb 0 */
        for (int i = 0; i < pc; i++) pv_x86p16rev_rev[pc][i] = (uint8_t)(15 - i);
    }
    pv_x86p16rev_built = 1;
}
/* p16revback (x86) — p16rev without the per-group right pshufb.  Store the whole
 * comb register backward into a scratch buffer (keeping the top pc lanes = this
 * group's rights reversed), then one 16-wide pshufb reverse pass (4x unrolled)
 * turns the globally-reversed scratch into the forward right output.  Reuses the
 * p16rev combined-index tables; hot loop drops the right pshufb at the cost of
 * the streaming reverse pass.  (NEON p16revback won only on Graviton2 — this
 * tests whether the store-bound SSE port behaves differently.) */
static uint8_t *pv_x86p16revback_buf = NULL;
static int      pv_x86p16revback_cap = 0;
static void prim_part_p16revback_x86(const ctx_t *c) {
    uint8_t *ranks = c->ranks_work, *bm = c->bm, *tmp = c->tmp8;
    int n = c->n; uint8_t thr = c->rank_thr;
    pv_build_x86p16rev();
    if (n + 64 > pv_x86p16revback_cap) {
        pv_x86p16revback_buf = (uint8_t *)realloc(pv_x86p16revback_buf, (size_t)n + 64);
        pv_x86p16revback_cap = n + 64;
    }
    uint8_t *rb = pv_x86p16revback_buf;
    int n_left = 0, j = 0, w = n + 16;
    __m128i thr1 = _mm_set1_epi8((char)(thr + 1));
#define _P16REVBACK_X86(v, mlo_, mhi_) do {                                    \
        uint32_t pc0_ = (uint32_t)__builtin_popcount((unsigned)(mlo_));         \
        uint32_t pc_  = pc0_ + (uint32_t)__builtin_popcount((unsigned)(mhi_));  \
        __m128i cidx_ = _mm_or_si128(                                           \
            _mm_load_si128((const __m128i *)pv_x86p16rev_tabA[(mlo_)]),         \
            _mm_load_si128((const __m128i *)pv_x86p16rev_tabB[pc0_][(mhi_)]));  \
        __m128i comb_ = _mm_shuffle_epi8((v), cidx_);                          \
        _mm_storeu_si128((__m128i *)(ranks + n_left), comb_);                  \
        _mm_storeu_si128((__m128i *)(rb + w - 16), comb_);                     \
        w -= pc_; n_left += 16 - pc_;                                           \
    } while (0)
    for (; j + 32 <= n; j += 32) {
        __m128i v0 = _mm_loadu_si128((const __m128i *)(ranks + j));
        __m128i v1 = _mm_loadu_si128((const __m128i *)(ranks + j + 16));
        uint32_t mlo = (uint16_t)_mm_movemask_epi8(_mm_cmpeq_epi8(_mm_min_epu8(v0, thr1), thr1));
        uint32_t mhi = (uint16_t)_mm_movemask_epi8(_mm_cmpeq_epi8(_mm_min_epu8(v1, thr1), thr1));
        uint32_t mm = mlo | (mhi << 16);
        memcpy(bm + (j >> 3), &mm, 4);
        _P16REVBACK_X86(v0, (uint8_t)mlo, (uint8_t)(mlo >> 8));
        _P16REVBACK_X86(v1, (uint8_t)mhi, (uint8_t)(mhi >> 8));
    }
    for (; j + 16 <= n; j += 16) {
        __m128i v = _mm_loadu_si128((const __m128i *)(ranks + j));
        uint16_t mm = (uint16_t)_mm_movemask_epi8(_mm_cmpeq_epi8(_mm_min_epu8(v, thr1), thr1));
        memcpy(bm + (j >> 3), &mm, 2);
        _P16REVBACK_X86(v, (uint8_t)mm, (uint8_t)(mm >> 8));
    }
#undef _P16REVBACK_X86
    int n_right = (n + 16) - w;
    for (; j < n; j++) {
        if ((j & 7) == 0) bm[j >> 3] = 0;
        uint8_t r = ranks[j];
        if (r > thr) { bm[j >> 3] |= (uint8_t)(1u << (j & 7)); tmp[n_right++] = r; }
        else         { ranks[n_left++] = r; }
    }
    /* reverse rb[w .. w+T) into tmp[0 .. T), 64 bytes/iter (pshufb REV16). */
    static const uint8_t REV16_a[16] = {15,14,13,12,11,10,9,8,7,6,5,4,3,2,1,0};
    __m128i rev16 = _mm_load_si128((const __m128i *)REV16_a);
    int T = (n + 16) - w, i = 0;
    const uint8_t *src = rb + w;
#define _REVST(off) _mm_storeu_si128((__m128i *)(tmp + (T - i - (off) - 16)),   \
            _mm_shuffle_epi8(_mm_loadu_si128((const __m128i *)(src + i + (off))), rev16))
    for (; i + 64 <= T; i += 64) { _REVST(0); _REVST(16); _REVST(32); _REVST(48); }
    for (; i + 16 <= T; i += 16) _REVST(0);
#undef _REVST
    for (; i < T; i++) tmp[T - 1 - i] = src[i];
}

/* p16rev (x86) — the classic p16rev cursor scheme: serial per-group
 * n_left/n_right chain (group 1's store addresses wait on group 0's popcount),
 * kept benchable as the baseline for the issue-#5 pfx-style per-group offsets
 * that replaced it in production.  Serial cursors were production from the
 * x86 p16rev promotion through 45147c8.  Uses the current 8 KB
 * x86_p16rev_tabB0 offset load (that span's production paired these cursors
 * with the 36 KB tabB), so a same-binary A/B vs production isolates the
 * CURSOR scheme alone. */
static void prim_part_p16rev_x86(const ctx_t *c) {
    uint8_t *ranks = c->ranks_work, *bm = c->bm, *tmp = c->tmp8;
    int n = c->n; uint8_t thr = c->rank_thr;
    x86_build_tabs();
    int n_left = 0, n_right = 0, j = 0;
    __m128i thr1 = _mm_set1_epi8((char)(thr + 1));
    static const uint8_t rev16_a[16] = {15,14,13,12,11,10,9,8,7,6,5,4,3,2,1,0};
    const __m128i rev16 = _mm_loadu_si128((const __m128i *)rev16_a);
#define _P16REVSER(v, mlo_, mhi_) do {                                          \
        uint32_t pc0_ = (uint32_t)__builtin_popcount((unsigned)(mlo_));         \
        uint32_t pc_  = pc0_ + (uint32_t)__builtin_popcount((unsigned)(mhi_));  \
        __m128i cidx_ = _mm_or_si128(                                           \
            _mm_load_si128((const __m128i *)x86_p16rev_tabA[(mlo_)]),           \
            _mm_loadu_si128((const __m128i *)&x86_p16rev_tabB0[(mhi_)][pc0_])); \
        __m128i comb_ = _mm_shuffle_epi8((v), cidx_);                          \
        _mm_storeu_si128((__m128i *)(ranks + n_left), comb_);                  \
        _mm_storeu_si128((__m128i *)(tmp + n_right),                            \
            _mm_shuffle_epi8(comb_, rev16));                                    \
        n_right += pc_; n_left += 16 - pc_;                                     \
    } while (0)
    for (; j + 32 <= n; j += 32) {
        __m128i v0 = _mm_loadu_si128((const __m128i *)(ranks + j));
        __m128i v1 = _mm_loadu_si128((const __m128i *)(ranks + j + 16));
        uint32_t mlo = (uint16_t)_mm_movemask_epi8(_mm_cmpeq_epi8(_mm_min_epu8(v0, thr1), thr1));
        uint32_t mhi = (uint16_t)_mm_movemask_epi8(_mm_cmpeq_epi8(_mm_min_epu8(v1, thr1), thr1));
        uint32_t mm = mlo | (mhi << 16);
        memcpy(bm + (j >> 3), &mm, 4);
        _P16REVSER(v0, (uint8_t)mlo, (uint8_t)(mlo >> 8));
        _P16REVSER(v1, (uint8_t)mhi, (uint8_t)(mhi >> 8));
    }
    for (; j + 16 <= n; j += 16) {
        __m128i v = _mm_loadu_si128((const __m128i *)(ranks + j));
        uint16_t mm = (uint16_t)_mm_movemask_epi8(_mm_cmpeq_epi8(_mm_min_epu8(v, thr1), thr1));
        memcpy(bm + (j >> 3), &mm, 2);
        _P16REVSER(v, (uint8_t)mm, (uint8_t)(mm >> 8));
    }
#undef _P16REVSER
    for (; j < n; j++) {
        if ((j & 7) == 0) bm[j >> 3] = 0;
        uint8_t r = ranks[j];
        if (r > thr) { bm[j >> 3] |= (uint8_t)(1u << (j & 7)); tmp[n_right++] = r; }
        else         { ranks[n_left++] = r; }
    }
    (void)n_right;
}

/* asof-ae49fe1 — the stride-8 ctab8 one-sided (right) compaction that was
 * production part_core_x86 before the 16-wide right16 path was promoted
 * (ae49fe1 = last commit with it as production).  Frozen as the baseline;
 * reuses the production x86_ctab_r/x86_pc8/x86_mask8. */
static void prim_part_right_asof_ae49fe1_x86(const ctx_t *c) {
    uint8_t *ranks = c->ranks_work, *bm = c->bm, *tmp = c->tmp8;
    int n = c->n; uint8_t thr = c->rank_thr;
    x86_build_tabs();
    int n_right = 0, j = 0;
    __m128i thr1 = _mm_set1_epi8((char)(thr + 1));
    for (; j + 8 <= n; j += 8) {
        __m128i v = _mm_loadl_epi64((const __m128i *)(ranks + j));
        uint8_t m = x86_mask8(v, thr1);
        bm[j >> 3] = m;
        _mm_storel_epi64((__m128i *)(tmp + n_right),
            _mm_shuffle_epi8(v, _mm_load_si128((const __m128i *)x86_ctab_r[m])));
        n_right += x86_pc8[m];
    }
    for (; j < n; j++) {
        if ((j & 7) == 0) bm[j >> 3] = 0;
        uint8_t r = ranks[j];
        if (r > thr) { bm[j >> 3] |= (uint8_t)(1u << (j & 7)); tmp[n_right++] = r; }
    }
}

/* ============================================================================
 * enc_partition_full : ryg-cshuf — ryg's "second way" half-combine (2026-07)
 *   OUR INTERPRETATION of the ryg email thread (partition ideas, licensed),
 *   not his verbatim code.
 *   Same idea as the NEON twin (see there): 256x8 aligned B table (2KB vs
 *   8KB) + computed shuffle instead of the padded-row unaligned load.  The
 *   {i - lc0} control's -lc0 broadcast comes GPR-free from psadbw on the
 *   INVERTED compare (sum of 0xff left lanes, low byte = -lc0 mod 256) +
 *   pshufb-0 broadcast (ryg).  pshufb zeroes the negative lanes; controls in
 *   [8,16) read the movq-zeroed high half.  Costs andnot+psadbw+paddb+2
 *   pshufb per group vs one never-splitting unaligned load: expected loss on
 *   the port-5-bound narrow Intels (c3/IVB..c5/SKX), the real question is
 *   c5a/Zen 2 + c6a/Zen 3 with dual shuffle ports.
 * ========================================================================== */
static uint8_t pv_rygb8x[256][8] __attribute__((aligned(64)));
static int pv_rygb8x_built = 0;
static void pv_build_rygb8x(void) {
    if (pv_rygb8x_built) return;
    x86_build_tabs();
    for (int m = 0; m < 256; m++) memcpy(pv_rygb8x[m], &x86_p16rev_tabB0[m][8], 8);
    pv_rygb8x_built = 1;
}
static inline int pv_part_rygcshuf_x86(uint8_t *ranks, int n, uint8_t thr,
                                       uint8_t *bm, uint8_t *tmp)
{
    pv_build_rygb8x();
    int n_left = 0, n_right = 0;
    int j = 0;
    __m128i thr1 = _mm_set1_epi8((char)(thr + 1));
    static const uint8_t rev16_a[16] = {15,14,13,12,11,10,9,8,7,6,5,4,3,2,1,0};
    static const uint8_t iota_a[16]  = {0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15};
    const __m128i rev16 = _mm_loadu_si128((const __m128i *)rev16_a);
    const __m128i iota  = _mm_loadu_si128((const __m128i *)iota_a);
    const __m128i ffs   = _mm_set1_epi8((char)0xff);
    const __m128i zero  = _mm_setzero_si128();
#define _PRYG(v, cmp_, mlo_, mhi_, cl_, cr_) do {                              \
        __m128i sad_ = _mm_sad_epu8(_mm_andnot_si128((cmp_), ffs), zero);      \
        __m128i sh_  = _mm_add_epi8(iota, _mm_shuffle_epi8(sad_, zero));       \
        __m128i cb_  = _mm_shuffle_epi8(                                       \
            _mm_loadl_epi64((const __m128i *)pv_rygb8x[(mhi_)]), sh_);         \
        __m128i cidx_ = _mm_or_si128(                                          \
            _mm_load_si128((const __m128i *)x86_p16rev_tabA[(mlo_)]), cb_);    \
        __m128i comb_ = _mm_shuffle_epi8((v), cidx_);                          \
        _mm_storeu_si128((__m128i *)(ranks + n_left + (cl_)), comb_);          \
        _mm_storeu_si128((__m128i *)(tmp + n_right + (cr_)),                   \
            _mm_shuffle_epi8(comb_, rev16));                                   \
    } while (0)
    for (; j + 32 <= n; j += 32) {
        __m128i v0 = _mm_loadu_si128((const __m128i *)(ranks + j));
        __m128i v1 = _mm_loadu_si128((const __m128i *)(ranks + j + 16));
        __m128i c0 = _mm_cmpeq_epi8(_mm_min_epu8(v0, thr1), thr1);
        __m128i c1 = _mm_cmpeq_epi8(_mm_min_epu8(v1, thr1), thr1);
        uint32_t mlo = (uint16_t)_mm_movemask_epi8(c0);
        uint32_t mhi = (uint16_t)_mm_movemask_epi8(c1);
        uint32_t mm = mlo | (mhi << 16);
        memcpy(bm + (j >> 3), &mm, 4);
        uint32_t cr1   = (uint32_t)__builtin_popcount(mlo);
        uint32_t total = (uint32_t)__builtin_popcount(mm);
        _PRYG(v0, c0, (uint8_t)mlo, (uint8_t)(mlo >> 8), 0, 0);
        _PRYG(v1, c1, (uint8_t)mhi, (uint8_t)(mhi >> 8), 16 - cr1, cr1);
        n_right += (int)total; n_left += 32 - (int)total;
    }
    for (; j + 16 <= n; j += 16) {
        __m128i v = _mm_loadu_si128((const __m128i *)(ranks + j));
        __m128i cm = _mm_cmpeq_epi8(_mm_min_epu8(v, thr1), thr1);
        uint16_t mm16 = (uint16_t)_mm_movemask_epi8(cm);
        memcpy(bm + (j >> 3), &mm16, 2);
        uint32_t pc16 = (uint32_t)__builtin_popcount((unsigned)mm16);
        _PRYG(v, cm, (uint8_t)mm16, (uint8_t)(mm16 >> 8), 0, 0);
        n_right += (int)pc16; n_left += 16 - (int)pc16;
    }
#undef _PRYG
    for (; j < n; j++) {
        if ((j & 7) == 0) bm[j >> 3] = 0;
        uint8_t r = ranks[j];
        if (r > thr) { bm[j >> 3] |= (uint8_t)(1u << (j & 7)); tmp[n_right++] = r; }
        else         { ranks[n_left++] = r; }
    }
    return n_right;
}
static void prim_part_rygcshuf_x86w(const ctx_t *c){ pv_part_rygcshuf_x86(c->ranks_work, c->n, c->rank_thr, c->bm, c->tmp8); }

/* ============================================================================
 * enc_partition_full : ryg-gaptab — ryg's shared-gap B table (2026-07)
 *   OUR INTERPRETATION of the ryg email thread (partition ideas, licensed),
 *   not his verbatim code.
 *   x86 twin of the NEON version above: production's combine unchanged,
 *   256*16+8 shared-gap table (4KB vs 8KB, total LUT 12KB -> 8KB); the
 *   cost is that ~1/4 of rows sit at line offset 48, so their +pc0 windows
 *   (pc0>0) cross a 64B line — the split-load tax production's 32B-padded
 *   rows avoid.  The bet is L1 residency > split loads on the 32KB-L1
 *   hosts (c3/IVB, c4/HSW). */
static uint8_t pv_ryggapx[256*16 + 16] __attribute__((aligned(64)));
static int pv_ryggapx_built = 0;
static void pv_build_ryggapx(void) {
    if (pv_ryggapx_built) return;
    x86_build_tabs();
    memset(pv_ryggapx, 0, sizeof pv_ryggapx);
    for (int m = 0; m < 256; m++)
        memcpy(pv_ryggapx + m*16 + 8, &x86_p16rev_tabB0[m][8], 8);
    pv_ryggapx_built = 1;
}
static inline int pv_part_ryggap_x86(uint8_t *ranks, int n, uint8_t thr,
                                     uint8_t *bm, uint8_t *tmp)
{
    pv_build_ryggapx();
    int n_left = 0, n_right = 0;
    int j = 0;
    __m128i thr1 = _mm_set1_epi8((char)(thr + 1));
    static const uint8_t rev16_a[16] = {15,14,13,12,11,10,9,8,7,6,5,4,3,2,1,0};
    const __m128i rev16 = _mm_loadu_si128((const __m128i *)rev16_a);
#define _PGAP(v, mlo_, mhi_, cl_, cr_) do {                                    \
        uint32_t pc0_ = (uint32_t)__builtin_popcount((unsigned)(mlo_));        \
        __m128i cidx_ = _mm_or_si128(                                          \
            _mm_load_si128((const __m128i *)x86_p16rev_tabA[(mlo_)]),          \
            _mm_loadu_si128((const __m128i *)(pv_ryggapx + (uint32_t)(mhi_)*16 + pc0_))); \
        __m128i comb_ = _mm_shuffle_epi8((v), cidx_);                          \
        _mm_storeu_si128((__m128i *)(ranks + n_left + (cl_)), comb_);          \
        _mm_storeu_si128((__m128i *)(tmp + n_right + (cr_)),                   \
            _mm_shuffle_epi8(comb_, rev16));                                   \
    } while (0)
    for (; j + 32 <= n; j += 32) {
        __m128i v0 = _mm_loadu_si128((const __m128i *)(ranks + j));
        __m128i v1 = _mm_loadu_si128((const __m128i *)(ranks + j + 16));
        uint32_t mlo = (uint16_t)_mm_movemask_epi8(
            _mm_cmpeq_epi8(_mm_min_epu8(v0, thr1), thr1));
        uint32_t mhi = (uint16_t)_mm_movemask_epi8(
            _mm_cmpeq_epi8(_mm_min_epu8(v1, thr1), thr1));
        uint32_t mm = mlo | (mhi << 16);
        memcpy(bm + (j >> 3), &mm, 4);
        uint32_t cr1   = (uint32_t)__builtin_popcount(mlo);
        uint32_t total = (uint32_t)__builtin_popcount(mm);
        _PGAP(v0, (uint8_t)mlo, (uint8_t)(mlo >> 8), 0, 0);
        _PGAP(v1, (uint8_t)mhi, (uint8_t)(mhi >> 8), 16 - cr1, cr1);
        n_right += (int)total; n_left += 32 - (int)total;
    }
    for (; j + 16 <= n; j += 16) {
        __m128i v = _mm_loadu_si128((const __m128i *)(ranks + j));
        uint16_t mm16 = (uint16_t)_mm_movemask_epi8(
            _mm_cmpeq_epi8(_mm_min_epu8(v, thr1), thr1));
        memcpy(bm + (j >> 3), &mm16, 2);
        uint32_t pc16 = (uint32_t)__builtin_popcount((unsigned)mm16);
        _PGAP(v, (uint8_t)mm16, (uint8_t)(mm16 >> 8), 0, 0);
        n_right += (int)pc16; n_left += 16 - (int)pc16;
    }
#undef _PGAP
    for (; j < n; j++) {
        if ((j & 7) == 0) bm[j >> 3] = 0;
        uint8_t r = ranks[j];
        if (r > thr) { bm[j >> 3] |= (uint8_t)(1u << (j & 7)); tmp[n_right++] = r; }
        else         { ranks[n_left++] = r; }
    }
    return n_right;
}
static void prim_part_ryggap_x86w(const ctx_t *c){ pv_part_ryggap_x86(c->ranks_work, c->n, c->rank_thr, c->bm, c->tmp8); }

#if defined(__AVX2__)
/* avx32: like sse32 but the 32-byte routing mask comes from a single ymm
 * vpcmpeqb(vpminub) + vpmovmskb instead of two SSE movemasks. */
static void prim_part_avx32_u8(const ctx_t *c) {
    uint8_t *ranks = c->ranks_work, *bm = c->bm, *tmp = c->tmp8;
    int n = c->n; uint8_t thr = c->rank_thr;
    x86_build_tabs();
    int n_left = 0, n_right = 0, j = 0;
    __m256i thr1y = _mm256_set1_epi8((char)(thr + 1));
    __m128i thr1  = _mm256_castsi256_si128(thr1y);
    for (; j + 32 <= n; j += 32) {
        __m256i v = _mm256_loadu_si256((const __m256i *)(ranks + j));
        uint32_t mm = (uint32_t)_mm256_movemask_epi8(
            _mm256_cmpeq_epi8(_mm256_min_epu8(v, thr1y), thr1y));
        memcpy(bm + (j >> 3), &mm, 4);
        __m128i v0 = _mm256_castsi256_si128(v);
        __m128i v1 = _mm256_extracti128_si256(v, 1);
        int r0 = x86_pc8[(uint8_t)mm];
        X86_COMPACT16(v0, (uint8_t)mm, (uint8_t)(mm >> 8), r0, n_left, n_right);
        int nr01 = __builtin_popcount(mm & 0xFFFF);
        n_right += nr01; n_left += 16 - nr01;
        int r2 = x86_pc8[(uint8_t)(mm >> 16)];
        X86_COMPACT16(v1, (uint8_t)(mm >> 16), (uint8_t)(mm >> 24), r2, n_left, n_right);
        int nr23 = __builtin_popcount(mm >> 16);
        n_right += nr23; n_left += 16 - nr23;
    }
    for (; j + 16 <= n; j += 16) {
        __m128i v = _mm_loadu_si128((const __m128i *)(ranks + j));
        uint16_t mm = (uint16_t)_mm_movemask_epi8(
            _mm_cmpeq_epi8(_mm_min_epu8(v, thr1), thr1));
        memcpy(bm + (j >> 3), &mm, 2);
        int rlo = x86_pc8[(uint8_t)mm], rhi = x86_pc8[(uint8_t)(mm >> 8)];
        X86_COMPACT16(v, (uint8_t)mm, (uint8_t)(mm >> 8), rlo, n_left, n_right);
        n_right += rlo + rhi; n_left += 16 - rlo - rhi;
    }
    for (; j < n; j++) {
        if ((j & 7) == 0) bm[j >> 3] = 0;
        uint8_t r = ranks[j];
        if (r > thr) { bm[j >> 3] |= (uint8_t)(1u << (j & 7)); tmp[n_right++] = r; }
        else         { ranks[n_left++] = r; }
    }
}
#endif  /* __AVX2__ */

#if defined(__AVX2__)
/* unroll2y: 2x-unrolled compress_tab partition with the mask gen + load fused
 * into one ymm chain.  bench_partition_unroll.c::partition_2y. */
static inline int prim_part_full_unroll2y_x86(uint16_t *codes_la, int n, int depth,
                                              uint8_t *bm, uint16_t *right_out) {
    int n_left = 0, n_right = 0, j = 0;
    __m128i shift_count = _mm_cvtsi32_si128(depth);
    for (; j + 16 <= n; j += 16) {
        __m256i code01 = _mm256_loadu_si256((const __m256i *)(codes_la + j));
        __m256i shifted = _mm256_sll_epi16(code01, shift_count);
        __m256i packed  = _mm256_packs_epi16(shifted, _mm256_setzero_si256());
        uint32_t mfull = (uint32_t)_mm256_movemask_epi8(packed);
        uint8_t m0 = (uint8_t)(mfull        & 0xFF);
        uint8_t m1 = (uint8_t)((mfull >> 16) & 0xFF);
        bm[j >> 3]       = m0;
        bm[(j >> 3) + 1] = m1;

        __m128i code0 = _mm256_castsi256_si128(code01);
        __m128i code1 = _mm256_extracti128_si256(code01, 1);

        const uint8_t *tab0 = compress_tab[m0];
        const uint8_t *tab1 = compress_tab[m1];
        __m128i sr0 = _mm_load_si128((const __m128i *)tab0);
        __m128i sl0 = _mm_load_si128((const __m128i *)(tab0 + 16));
        __m128i sr1 = _mm_load_si128((const __m128i *)tab1);
        __m128i sl1 = _mm_load_si128((const __m128i *)(tab1 + 16));

        __m128i r0 = _mm_shuffle_epi8(code0, sr0);
        __m128i l0 = _mm_shuffle_epi8(code0, sl0);
        __m128i r1 = _mm_shuffle_epi8(code1, sr1);
        __m128i l1 = _mm_shuffle_epi8(code1, sl1);

        int nr0 = compress_popcnt[m0];
        int nr1 = compress_popcnt[m1];

        _mm_storeu_si128((__m128i *)(right_out + n_right),       r0);
        _mm_storeu_si128((__m128i *)(right_out + n_right + nr0), r1);
        _mm_storeu_si128((__m128i *)(codes_la  + n_left),        l0);
        _mm_storeu_si128((__m128i *)(codes_la  + n_left + (8 - nr0)), l1);

        n_right += nr0 + nr1;
        n_left  += (8 - nr0) + (8 - nr1);
    }
    if (j + 8 <= n) {
        __m128i code = _mm_loadu_si128((const __m128i *)(codes_la + j));
        __m128i shifted = _mm_sll_epi16(code, shift_count);
        __m128i packed  = _mm_packs_epi16(shifted, _mm_setzero_si128());
        uint8_t mask    = (uint8_t)_mm_movemask_epi8(packed);
        bm[j >> 3] = mask;
        const uint8_t *tab = compress_tab[mask];
        __m128i sr = _mm_load_si128((const __m128i *)tab);
        __m128i sl = _mm_load_si128((const __m128i *)(tab + 16));
        __m128i r  = _mm_shuffle_epi8(code, sr);
        __m128i l  = _mm_shuffle_epi8(code, sl);
        int nr = compress_popcnt[mask];
        _mm_storeu_si128((__m128i *)(right_out + n_right), r);
        _mm_storeu_si128((__m128i *)(codes_la  + n_left ), l);
        n_right += nr;
        n_left  += (8 - nr);
        j += 8;
    }
    int shift_d = 15 - depth;
    for (; j < n; j++){ uint16_t c=codes_la[j]; if((c>>shift_d)&1) right_out[n_right++]=c; else codes_la[n_left++]=c; }
    return n_right;
}
static void prim_part_full_unroll2y(const ctx_t *c){
    prim_part_full_unroll2y_x86(c->la_work, c->n, c->depth, c->bm, c->tmp16);
}

#if defined(__BMI2__)
/* pext: no compress_tab; derive the pshufb control on the fly via BMI2
 * pdep/pext on a packed-indices vector.  bench_partition_avx2.c::partition_pext.
 * (BMI2 pext is Intel/Zen4-fast, AMD-pre-Zen4-slow -- gated on __BMI2__.) */
static inline int prim_part_full_pext_x86(uint16_t *codes_la, int n, int depth,
                                          uint8_t *bm, uint16_t *right_out) {
    int n_left = 0, n_right = 0, j = 0;
    __m128i shift_count = _mm_cvtsi32_si128(depth);
    const __m128i dup_shuf = _mm_setr_epi8(0,0,1,1,2,2,3,3,4,4,5,5,6,6,7,7);
    const __m128i odd_offset = _mm_setr_epi8(0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1);
    for (; j + 8 <= n; j += 8) {
        __m128i code_vec = _mm_loadu_si128((const __m128i *)(codes_la + j));
        __m128i shifted = _mm_sll_epi16(code_vec, shift_count);
        __m128i packed  = _mm_packs_epi16(shifted, _mm_setzero_si128());
        uint8_t mask    = (uint8_t)_mm_movemask_epi8(packed);
        bm[j >> 3] = mask;

        uint32_t indices = 0x76543210u;
        uint32_t mask_ex_r = _pdep_u32((uint32_t)mask,           0x11111111u) * 0x0Fu;
        uint32_t mask_ex_l = _pdep_u32((uint32_t)(uint8_t)~mask, 0x11111111u) * 0x0Fu;
        uint32_t comp_r    = _pext_u32(indices, mask_ex_r);
        uint32_t comp_l    = _pext_u32(indices, mask_ex_l);

        uint64_t spread_r = _pdep_u64((uint64_t)comp_r, 0x0F0F0F0F0F0F0F0Full);
        uint64_t spread_l = _pdep_u64((uint64_t)comp_l, 0x0F0F0F0F0F0F0F0Full);
        __m128i r_bytes  = _mm_cvtsi64_si128((int64_t)spread_r);
        __m128i l_bytes  = _mm_cvtsi64_si128((int64_t)spread_l);
        __m128i r_dup    = _mm_shuffle_epi8(r_bytes, dup_shuf);
        __m128i l_dup    = _mm_shuffle_epi8(l_bytes, dup_shuf);
        __m128i shuf_r   = _mm_add_epi8(_mm_add_epi8(r_dup, r_dup), odd_offset);
        __m128i shuf_l   = _mm_add_epi8(_mm_add_epi8(l_dup, l_dup), odd_offset);

        __m128i right = _mm_shuffle_epi8(code_vec, shuf_r);
        __m128i left  = _mm_shuffle_epi8(code_vec, shuf_l);
        int nr = __builtin_popcount(mask);
        _mm_storeu_si128((__m128i *)(right_out + n_right), right);
        _mm_storeu_si128((__m128i *)(codes_la  + n_left ), left);
        n_right += nr;
        n_left  += (8 - nr);
    }
    int shift_d = 15 - depth;
    for (; j < n; j++){ uint16_t c=codes_la[j]; if((c>>shift_d)&1) right_out[n_right++]=c; else codes_la[n_left++]=c; }
    return n_right;
}
static void prim_part_full_pext(const ctx_t *c){
    prim_part_full_pext_x86(c->la_work, c->n, c->depth, c->bm, c->tmp16);
}
#endif /* __BMI2__ */
#endif /* __AVX2__ */
#endif /* __SSE4_1__ */

/* ============================================================================
 * coalesce LOSERS — AVX-512 store-coalescing partition experiments
 *   From extras/bench/bench_coalesce_avx512.c (bench_compressstoreu,
 *   bench_macro), reshaped to the ST_PART contract: in-place u16 partition,
 *   mask from bit (15-depth) of each code, full bm written, left compacted
 *   back into codes_la (= la_work), right into right_out (= tmp16) — matching
 *   p_part_scalar / the production AVX-512 partition.  Both are documented
 *   LOSERS vs the production vpcompressw + full-store partition.
 * ========================================================================== */
#if defined(__AVX512VBMI2__) && defined(__AVX512VBMI__)
#include <immintrin.h>

/* Per-32-code mask: shift the partition bit (15-depth) to bit 15, read the
 * sign bits.  Identical to enc_mask32_codes_la_avx512 in the production
 * backend (the bench reference scalar_partition uses bit 15-depth). */
static inline uint32_t pv_part_mask32(__m512i code_vec, int depth) {
    return (uint32_t)_mm512_movepi16_mask(_mm512_slli_epi16(code_vec, depth));
}

/* coal_compressstoreu: single-instruction _mm512_mask_compressstoreu_epi16
 * per side (writes only popcount*2 bytes, advancing by the same). */
static int pv_part_coal_compressstoreu_avx512(uint16_t *codes_la, int n, int depth,
                                              uint8_t *bm, uint16_t *right_out) {
    int n_left = 0, n_right = 0, j = 0;
    for (; j + 32 <= n; j += 32) {
        __m512i data = _mm512_loadu_si512((const __m512i *)(codes_la + j));
        uint32_t mask = pv_part_mask32(data, depth);
        memcpy(bm + (j >> 3), &mask, 4);
        _mm512_mask_compressstoreu_epi16(right_out + n_right, (__mmask32)mask,  data);
        _mm512_mask_compressstoreu_epi16(codes_la  + n_left,  (__mmask32)~mask, data);
        int nr = __builtin_popcount(mask);
        n_right += nr; n_left += 32 - nr;
    }
    for (; j + 8 <= n; j += 8) {
        __m128i data = _mm_loadu_si128((const __m128i *)(codes_la + j));
        __m128i sh   = _mm_slli_epi16(data, depth);
        uint32_t mask = (uint32_t)_mm_movepi16_mask(sh) & 0xFF;
        bm[j >> 3] = (uint8_t)mask;
        _mm_mask_compressstoreu_epi16(right_out + n_right, (__mmask8)mask,  data);
        _mm_mask_compressstoreu_epi16(codes_la  + n_left,  (__mmask8)~mask, data);
        int nr = __builtin_popcount(mask);
        n_right += nr; n_left += 8 - nr;
    }
    if (j < n) {
        int tail = n - j, shd = 15 - depth; uint8_t mask = 0;
        uint16_t tb[8]; for (int k = 0; k < tail; k++) tb[k] = codes_la[j + k];
        for (int k = 0; k < tail; k++) mask |= (uint8_t)(((tb[k] >> shd) & 1) << k);
        bm[j >> 3] = mask;
        for (int k = 0; k < tail; k++)
            if (mask & (1 << k)) right_out[n_right++] = tb[k];
            else                 codes_la[n_left++]   = tb[k];
    }
    return n_right;
}
static void prim_part_coal_compressstoreu(const ctx_t *c) {
    pv_part_coal_compressstoreu_avx512(c->la_work, c->n, c->depth, c->bm, c->tmp16);
}

/* coal_vpermb_macro: 2-iter macro-block coalesce.  Accumulate two consecutive
 * iters' compressed data into one 64-byte zmm via runtime byte-shift (vpermb),
 * one store per side per macro = 0.5 stores/iter. */
static const uint8_t pv_coal_iota64[64] __attribute__((aligned(64))) = {
     0, 1, 2, 3, 4, 5, 6, 7, 8, 9,10,11,12,13,14,15,
    16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,
    32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,
    48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63};
static int pv_part_coal_macro_avx512(uint16_t *codes_la, int n, int depth,
                                     uint8_t *bm, uint16_t *right_out) {
    const __m512i iota = _mm512_load_si512((const __m512i *)pv_coal_iota64);
    int n_left = 0, n_right = 0, j = 0;
    for (; j + 64 <= n; j += 64) {
        __m512i d0 = _mm512_loadu_si512((const __m512i *)(codes_la + j));
        __m512i d1 = _mm512_loadu_si512((const __m512i *)(codes_la + j + 32));
        uint32_t m0 = pv_part_mask32(d0, depth);
        uint32_t m1 = pv_part_mask32(d1, depth);
        memcpy(bm + (j >> 3),     &m0, 4);
        memcpy(bm + (j >> 3) + 4, &m1, 4);
        int pr0 = __builtin_popcount(m0), pr1 = __builtin_popcount(m1);
        int pl0 = 32 - pr0, pl1 = 32 - pr1;
        __m512i r0 = _mm512_maskz_compress_epi16((__mmask32)m0, d0);
        __m512i r1 = _mm512_maskz_compress_epi16((__mmask32)m1, d1);
        __m512i l0 = _mm512_maskz_compress_epi16((__mmask32)~m0, d0);
        __m512i l1 = _mm512_maskz_compress_epi16((__mmask32)~m1, d1);
        /* Right side.  Place r1 at byte offset pr0*2 via vpermb on (iota -
         * pr0*2); vpermb uses only the low 6 bits, so positions below the
         * offset wrap to garbage — zero them with a maskz keyed on
         * iota >= pr0*2 so the OR with r0 keeps r0's low bytes intact.  When
         * pr0+pr1 > 32 the combined data exceeds one 64-byte zmm, so fall
         * back to two stores (the coalesce can't represent it).  (The
         * original throughput-only bench skipped both the mask and this
         * spill check — it never verified.) */
        if (pr0 + pr1 <= 32) {
            __m512i offr = _mm512_set1_epi8((char)(pr0 * 2));
            __mmask64 keepr = _mm512_cmpge_epu8_mask(iota, offr);
            __m512i r1p = _mm512_maskz_permutexvar_epi8(keepr, _mm512_sub_epi8(iota, offr), r1);
            _mm512_storeu_si512((__m512i *)(right_out + n_right), _mm512_or_si512(r0, r1p));
        } else {
            _mm512_storeu_si512((__m512i *)(right_out + n_right), r0);
            _mm512_storeu_si512((__m512i *)(right_out + n_right + pr0), r1);
        }
        n_right += pr0 + pr1;
        /* Left side (in-place into codes_la). */
        if (pl0 + pl1 <= 32) {
            __m512i offl = _mm512_set1_epi8((char)(pl0 * 2));
            __mmask64 keepl = _mm512_cmpge_epu8_mask(iota, offl);
            __m512i l1p = _mm512_maskz_permutexvar_epi8(keepl, _mm512_sub_epi8(iota, offl), l1);
            _mm512_storeu_si512((__m512i *)(codes_la + n_left), _mm512_or_si512(l0, l1p));
        } else {
            _mm512_storeu_si512((__m512i *)(codes_la + n_left), l0);
            _mm512_storeu_si512((__m512i *)(codes_la + n_left + pl0), l1);
        }
        n_left += pl0 + pl1;
    }
    /* stride-8 tail */
    for (; j + 8 <= n; j += 8) {
        __m128i data = _mm_loadu_si128((const __m128i *)(codes_la + j));
        uint32_t mask = (uint32_t)_mm_movepi16_mask(_mm_slli_epi16(data, depth)) & 0xFF;
        bm[j >> 3] = (uint8_t)mask;
        _mm_mask_compressstoreu_epi16(right_out + n_right, (__mmask8)mask,  data);
        _mm_mask_compressstoreu_epi16(codes_la  + n_left,  (__mmask8)~mask, data);
        int nr = __builtin_popcount(mask);
        n_right += nr; n_left += 8 - nr;
    }
    if (j < n) {
        int tail = n - j, shd = 15 - depth; uint8_t mask = 0;
        uint16_t tb[8]; for (int k = 0; k < tail; k++) tb[k] = codes_la[j + k];
        for (int k = 0; k < tail; k++) mask |= (uint8_t)(((tb[k] >> shd) & 1) << k);
        bm[j >> 3] = mask;
        for (int k = 0; k < tail; k++)
            if (mask & (1 << k)) right_out[n_right++] = tb[k];
            else                 codes_la[n_left++]   = tb[k];
    }
    return n_right;
}
static void prim_part_coal_macro(const ctx_t *c) {
    pv_part_coal_macro_avx512(c->la_work, c->n, c->depth, c->bm, c->tmp16);
}

/* ============================================================================
 * enc_partition_full / enc_partition_right : asof-10e19a1 -- the zero-masked
 * (maskz) compress forms, production until the issue-#11 merge-masking fix
 * (part_left_avx512 had the same one-line pattern; it has no bench stage).
 * Zen 4/5 false-dep on the maskz destination; c8a micro: full 0.0344 vs
 * 0.0159, right 0.0175 vs 0.0095.  NB: partition micro is bimodal on
 * c7a/c8i from per-run buffer-address (4K-aliasing) luck -- compare like
 * modes or min-of-several.
 * ========================================================================== */
static int pv_part_full_avx512_maskz(uint8_t *ranks, int n, uint8_t thr,
                                     uint8_t *bm, uint8_t *tmp)
{
    int n_left = 0, n_right = 0;
    int j = 0;
    __m512i vt = _mm512_set1_epi8((char)thr);
    for (; j + 64 <= n; j += 64) {
        __m512i v = _mm512_loadu_si512((const void *)(ranks + j));
        __mmask64 k = _mm512_cmpgt_epu8_mask(v, vt);
        int p = __builtin_popcountll(k);
        memcpy(bm + (j >> 3), &k, 8);
        _mm512_storeu_si512((void *)(tmp + n_right),   _mm512_maskz_compress_epi8(k, v));
        _mm512_storeu_si512((void *)(ranks + n_left), _mm512_maskz_compress_epi8(~k, v));
        n_right += p;
        n_left += 64 - p;
    }
    for (; j < n; j++) {
        if ((j & 7) == 0) bm[j >> 3] = 0;
        uint8_t r = ranks[j];
        if (r > thr) { bm[j >> 3] |= (uint8_t)(1u << (j & 7)); tmp[n_right++] = r; }
        else         { ranks[n_left++] = r; }
    }
    return n_right;
}
static int pv_part_right_avx512_maskz(uint8_t *ranks, int n, uint8_t thr,
                                      uint8_t *bm, uint8_t *tmp)
{
    int n_right = 0, j = 0;
    __m512i vt = _mm512_set1_epi8((char)thr);
    for (; j + 64 <= n; j += 64) {
        __m512i v = _mm512_loadu_si512((const void *)(ranks + j));
        __mmask64 k = _mm512_cmpgt_epu8_mask(v, vt);
        memcpy(bm + (j >> 3), &k, 8);
        _mm512_storeu_si512((void *)(tmp + n_right), _mm512_maskz_compress_epi8(k, v));
        n_right += __builtin_popcountll(k);
    }
    for (; j < n; j++) {
        if ((j & 7) == 0) bm[j >> 3] = 0;
        uint8_t r = ranks[j];
        if (r > thr) { bm[j >> 3] |= (uint8_t)(1u << (j & 7)); tmp[n_right++] = r; }
    }
    return n_right;
}
static void prim_part_maskz(const ctx_t *c){ pv_part_full_avx512_maskz(c->ranks_work, c->n, c->rank_thr, c->bm, c->tmp8); }
static void prim_part_right_maskz(const ctx_t *c){ pv_part_right_avx512_maskz(c->ranks_work, c->n, c->rank_thr, c->bm, c->tmp8); }
#endif /* __AVX512VBMI2__ && __AVX512VBMI__ */

/* ============================================================================
 * Registry — partition family (no-op where the ISA is unavailable)
 * ========================================================================== */
static void pv_register_partition(void) {
    /* enc_partition_full — AVX-512 coalesce losers */
    PV_VARIANT(ST_U16_PART, "coal_compressstoreu", PV_ISA_AVX512,
               "bench_coalesce_avx512.c bench_compressstoreu",
               "1-instr compress+store; LOSER vs full-store production", 1,
               PV_FN_VBMI2(prim_part_coal_compressstoreu));
    PV_VARIANT(ST_U16_PART, "coal_vpermb_macro", PV_ISA_AVX512,
               "bench_coalesce_avx512.c bench_macro",
               "2-iter vpermb coalesce, 0.5 stores/iter; LOSER", 1,
               PV_FN_VBMI2(prim_part_coal_macro));
    /* NEON */
    PV_VARIANT(ST_U16_PART,      "asof-5f3222e", PV_ISA_NEON,
               "5f3222e (2026-05-26)",
               "pre-COM stride-8 serial-cursor FULL partition", 1,
               PV_FN_NEON(prim_part_full_asof_5f3222e));
    PV_VARIANT(ST_U16_PART,      "prefix64", PV_ISA_NEON,
               "Jeff Plaisance / 6d61760",
               "M4 FULL +15-18% vs shipped COM; needs Graviton check", 1,
               PV_FN_NEON(prim_part_full_prefix64));
    PV_VARIANT(ST_BMBUILD,   "prefix64", PV_ISA_NEON,
               "Jeff Plaisance / 6d61760",
               "M4 NONE ~3-4% faster than shipped COM", 0,
               PV_FN_NEON(prim_part_none_prefix64));
    PV_VARIANT(ST_FUSEDHALF, "prefix64", PV_ISA_NEON,
               "Jeff Plaisance / 6d61760",
               "M4 RIGHT ~2-3% slower than shipped COM", 0,
               PV_FN_NEON(prim_part_right_prefix64));
    PV_VARIANT(ST_U16_PART, "com",              PV_ISA_NEON, "bench_partition_neon.c",
               "64/iter 8-chunk prefix-sum cursors, masks via 8x vaddvq; pre-com_v3 step", 1, PV_FN_NEON(prim_part_com));
    PV_VARIANT(ST_U16_PART, "com_v2_transpose", PV_ISA_NEON, "bench_partition_neon.c",
               "com but masks via 8x8 vsli transpose; LOSER (transpose > the 8 vaddvq it removes)", 1, PV_FN_NEON(prim_part_com_v2_trans));
    PV_VARIANT(ST_U16_PART, "split16",          PV_ISA_NEON, "bench_encode_split.c (CODES16)",
               "dense-codes SIMD-movemask partition_8, stride-8; no prefix-sum cursor decouple", 1, PV_FN_NEON(prim_part_split16));
    PV_VARIANT(ST_U16_PART, "split16_unroll",   PV_ISA_NEON, "bench_encode_split.c (CODES16U)",
               "split16 at stride-16 (2 partition_8/iter for ILP)", 1, PV_FN_NEON(prim_part_split16_unroll));
    PV_VARIANT(ST_PARTBM,   "coal_vext",     PV_ISA_NEON, "bench_coalesce.c",
               "per-iter store-coalesce, switch on so_far -> vextq; LOSER", 1, PV_FN_NEON(prim_part_coal_vext));
    PV_VARIANT(ST_PARTBM,   "coal_tbl",      PV_ISA_NEON, "bench_coalesce.c",
               "per-iter coalesce, runtime vqtbl1q shuffle; LOSER", 1, PV_FN_NEON(prim_part_coal_tbl));
    /* x86 enc_partition_full (u8 rank): the per-8-chunk u16-like port vs the
       shipped 16-wide + single-movemask part_full_x86. */
    PV_VARIANT(ST_PART, "asof-10e19a1", PV_ISA_AVX512, "10e19a1 (prior production)",
               "maskz compress x2; merge-masked form (issue #11) wins -54% c8a / ~-20% c7a (Zen false dep), ~0 c7i/c8i", 0, PV_FN_VBMI2(prim_part_maskz));
    PV_VARIANT(ST_PART_RIGHT, "asof-10e19a1", PV_ISA_AVX512, "10e19a1 (prior production)",
               "maskz compress; merge-masked form (issue #11) wins -46% c8a / ~-9% c7a, ~0 c7i/c8i", 0, PV_FN_VBMI2(prim_part_right_maskz));
    PV_VARIANT(ST_PART, "u16-like", PV_ISA_SSE4,
               "30f42a5 per-8-chunk port",
               "per-8-chunk u8 port (mirrors u16 shape); vs shipped 16-wide", 1,
               PV_FN_SSE(prim_part_u16like));
    PV_VARIANT(ST_PART, "asof-3a138a6", PV_ISA_SSE4,
               "32/iter dense X86_COMPACT16 (ex-production x86)",
               "the prior production x86 full partition, before p16rev was promoted; kept benchable as the baseline", 1,
               PV_FN_SSE(prim_part_asof_3a138a6_x86));
    PV_VARIANT(ST_PART, "halftab", PV_ISA_SSE4, "16b-enc.txt half-table",
               "sse32 but left = right under ~mask; drops ctab_l/pre_l (40KB vs 80KB); Zen3 +0.7% but AVX2 -3..-5% / IvyB -7.5% (complement ALU > L1 saving); not promoted", 1,
               PV_FN_SSE(prim_part_halftab));
    PV_VARIANT(ST_PART, "p16revback", PV_ISA_SSE4, "p16rev, right stored backward + one reverse pass",
               "drops p16rev's per-group right pshufb: store comb backward into scratch (keep top pc), one 4x-unrolled pshufb reverse pass after the stride loop. loses to p16rev on all x86 (Zen2/3 Skylake Haswell IvyB) -- the extra reverse-pass stores hurt the store-bound SSE loop; not promoted", 1,
               PV_FN_SSE(prim_part_p16revback_x86));
    PV_VARIANT(ST_PART, "ryg-cshuf", PV_ISA_SSE4, "ryg 'second way' half-combine (2026-07)",
               "256x8 aligned B table (2KB vs 8KB) + computed-shuffle combine; -lc0 broadcast GPR-free via psadbw on the inverted compare + pshufb-0 (ryg); vs production's never-line-crossing unaligned load.  Loses: +11% c5/SKX (port 5), +3% c4/HSW, wash c5a+c6a/Zen 2/3; not promoted", 1,
               PV_FN_SSE(prim_part_rygcshuf_x86w));
    PV_VARIANT(ST_PART, "ryg-gaptab", PV_ISA_SSE4, "ryg shared-gap B table (2026-07)",
               "production combine, 256*16+8 shared-gap tabB0 (4KB vs 8KB, LUT total 12->8KB); ~1/4 of rows at line offset 48 -> split B-loads at pc0>0.  LOST the bet: micro wash (+2% c3/IVB..-1% c6a/Zen 3), E2E fair enc_pb NEGATIVE everywhere (c4/HSW + c5/SKX -1.4% geomean, c3/IVB ~0) -- split loads beat the 4KB L1 saving even on 32KB-L1 hosts; not promoted", 1,
               PV_FN_SSE(prim_part_ryggap_x86w));
    PV_VARIANT(ST_PART, "p16rev", PV_ISA_SSE4, "classic p16rev (serial cursors)",
               "the cursor scheme production used before the issue-#5 per-group store offsets (production through 45147c8): serial n_left/n_right chain across the 2 groups; same shuffle + current 8KB tabB0 (that prod had the 36KB tabB), so the delta vs production isolates the cursors", 1,
               PV_FN_SSE(prim_part_p16rev_x86));
    PV_VARIANT(ST_PART, "avx32", PV_ISA_AVX2,
               "32 ranks/iter, 1x ymm movemask + 2x X86_COMPACT16",
               "32/iter; single 32-bit mask build vs 2x SSE", 1,
               PV_FN_SSE_AVX2(prim_part_avx32_u8));
    PV_VARIANT(ST_PART, "p16", PV_ISA_NEON, "16b-enc.txt (two-table partition)",
               "16 ranks/iter two-table compaction; only V1/Graviton3 +2.6%, loses M4 -12% N1 -14% V2 -4% V3 -2% (extra loads + 36KB tab2 latency); not promoted", 1,
               PV_FN_NEON(prim_part_p16));
    PV_VARIANT(ST_PART, "asof-a3a3d19", PV_ISA_NEON, "per-8-chunk ctab8 COM64 (ex-production)",
               "the prior production NEON full partition, before p16rev was promoted; kept benchable as the baseline", 1,
               PV_FN_NEON(prim_part_asof_a3a3d19));
    PV_VARIANT(ST_PART, "asof-f9974f5", PV_ISA_NEON, "f9974f5 (prior production)",
               "pre-PR#22 p16rev main loop (no software pipelining; mask popcount after a GPR round-trip)", 1,
               PV_FN_NEON(prim_part_asof_f9974f5));
    PV_VARIANT(ST_PART, "p16revback", PV_ISA_NEON, "p16rev, right stored backward + one final reverse pass",
               "drops p16rev's per-group rev shuffle: store comb backward into scratch (keep top pc), one 16-wide reverse pass after the stride loop. wins only N1/Graviton2 +7.5%; loses M4 -4% V1/G3 -5% V2/G4 -5% V3 -2% (extra reverse-pass store traffic > saved shuffle on wide cores); not promoted", 1,
               PV_FN_NEON(prim_part_p16revback));
    PV_VARIANT(ST_PART, "p16rev", PV_ISA_NEON, "classic p16rev (serial cursors)",
               "the cursor scheme production used before the issue-#5 pfx-sum cursors (production as of 3a138a6..4d93965): serial per-group n_left/n_right chain (each group's store addresses wait on the previous group's popcount); same shuffle + current 8KB tabB0 (that prod had the 36KB tabB), so the delta vs production isolates the cursors. pfx beats it ~1.5-1.7% c7g/c8g, 4.2% m9g, 4.3% M4 micro", 1,
               PV_FN_NEON(prim_part_p16rev));
    PV_VARIANT(ST_PART, "ryg-cshuf", PV_ISA_NEON, "ryg 'second way' half-combine (2026-07)",
               "256x8 aligned B table + computed-shuffle {i+pc0-8} combine instead of the padded-row unaligned load; dup+add+extra tbl per group.  +19% on M4 (unaligned loads are free there); not promoted", 1,
               PV_FN_NEON(prim_part_rygcshuf));
    PV_VARIANT(ST_PART, "ryg-gaptab", PV_ISA_NEON, "ryg shared-gap B table (2026-07)",
               "production combine, 256*16+8 shared-gap tabB0 (4KB vs 8KB, LUT total 12->8KB); some B-loads now cross cache lines.  M4 micro +1.6%; not promoted", 1,
               PV_FN_NEON(prim_part_ryggap));
    PV_VARIANT(ST_PART_RIGHT, "right16", PV_ISA_NEON, "16-wide one-sided right compaction",
               "1 vqtbl1q + 1 16-byte store per 16 lanes (p16 right-pack two-table index) vs the per-8-chunk production 2 vtbl1 + 2 8-byte stores; halves shuffle+store count, pays the 36 KB tab2 latency", 1,
               PV_FN_NEON(prim_part_right16));
    PV_VARIANT(ST_PART_RIGHT, "asof-ae49fe1", PV_ISA_SSE4, "stride-8 ctab8 one-sided (ex-production x86)",
               "the prior production x86 one-sided partition, before the 16-wide path was promoted; kept benchable as the baseline", 1,
               PV_FN_SSE(prim_part_right_asof_ae49fe1_x86));
    /* x86 enc_partition_full (u16 code_la graveyard) */
    PV_VARIANT(ST_U16_PART, "sse_com", PV_ISA_SSE4,
               "bench_partition_x86.c / IDEAS x86 COM partition",
               "64 codes/iter prefix-sum cursors; AMD win, Intel regress", 1,
               PV_FN_SSE(prim_part_full_com));
    PV_VARIANT(ST_U16_PART, "unroll2y", PV_ISA_AVX2,
               "bench_partition_unroll.c",
               "2x-unroll, ymm-fused mask gen + load", 1,
               PV_FN_SSE_AVX2(prim_part_full_unroll2y));
    PV_VARIANT(ST_U16_PART, "pext", PV_ISA_AVX2,
               "bench_partition_avx2.c",
               "BMI2 pdep/pext shuffle-control gen, no compress_tab", 1,
               PV_FN_SSE_AVX2_BMI2(prim_part_full_pext));
}

#endif /* PIVCO_PRIM_VARIANTS_PARTITION_H */
