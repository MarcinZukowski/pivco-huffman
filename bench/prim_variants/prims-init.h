/* bench/prim_variants/prims-init.h — enc_init (sym -> rank gather) variants. */
#ifndef PIVCO_PRIM_VARIANTS_INIT_H
#define PIVCO_PRIM_VARIANTS_INIT_H

#if defined(USE_NEON_KERNELS)

/* simd16 is a simplified form (no GPR interleave) of the current production
 * simd20 version, kept for posterity / per-uarch re-eval.
 * Based on #5 by dougallj. */

#define PV_INIT_LOADTAB(s2r)                                                 \
    uint8x16x4_t t0, t1, t2, t3;                                             \
    t0.val[0]=vld1q_u8((s2r)     ); t0.val[1]=vld1q_u8((s2r)+ 16);           \
    t0.val[2]=vld1q_u8((s2r)+ 32); t0.val[3]=vld1q_u8((s2r)+ 48);            \
    t1.val[0]=vld1q_u8((s2r)+ 64); t1.val[1]=vld1q_u8((s2r)+ 80);            \
    t1.val[2]=vld1q_u8((s2r)+ 96); t1.val[3]=vld1q_u8((s2r)+112);            \
    t2.val[0]=vld1q_u8((s2r)+128); t2.val[1]=vld1q_u8((s2r)+144);            \
    t2.val[2]=vld1q_u8((s2r)+160); t2.val[3]=vld1q_u8((s2r)+176);            \
    t3.val[0]=vld1q_u8((s2r)+192); t3.val[1]=vld1q_u8((s2r)+208);            \
    t3.val[2]=vld1q_u8((s2r)+224); t3.val[3]=vld1q_u8((s2r)+240);            \
    const uint8x16_t s64=vdupq_n_u8(64), s128=vdupq_n_u8(128), s192=vdupq_n_u8(192)

static void prim_init_simd16_neon(uint8_t *ranks, int n, const uint8_t *sym, const uint8_t *s2r) {
    int i = 0;
    if (n >= 16) {
        PV_INIT_LOADTAB(s2r);
        for (; i + 16 <= n; i += 16) {
            uint8x16_t c = vld1q_u8(sym + i);
            uint8x16_t r = vqtbl4q_u8(t0, c);
            r = vqtbx4q_u8(r, t1, vsubq_u8(c, s64));
            r = vqtbx4q_u8(r, t2, vsubq_u8(c, s128));
            r = vqtbx4q_u8(r, t3, vsubq_u8(c, s192));
            vst1q_u8(ranks + i, r);
        }
    }
    for (; i < n; i++) ranks[i] = s2r[sym[i]];
}
static void prim_init_simd16(const ctx_t *c){ prim_init_simd16_neon(c->ranks_work, c->n, c->symbuf, c->sym_to_rank); }

#undef PV_INIT_LOADTAB
#endif /* USE_NEON_KERNELS */

static void pv_register_init(void) {
    /* simd20 (the 20 sym/iter interleaved form) is production init_neon; this
     * keeps the pure-SIMD 16 sym/iter form for posterity / per-uarch re-eval. */
    PV_VARIANT(ST_ENC_INIT, "simd16", PV_ISA_NEON, "issue #5 (dougallj)",
               "256-entry s2r gather via vqtbl4 + 3x vqtbx4, 16 sym/iter (pure SIMD); production uses the 20 sym/iter interleaved form", 0,
               PV_FN_NEON(prim_init_simd16));
}

#endif /* PIVCO_PRIM_VARIANTS_INIT_H */
