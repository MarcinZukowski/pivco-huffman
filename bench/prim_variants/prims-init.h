/* bench/prim_variants/prims-init.h — enc_init (sym -> rank gather) variants. */
#ifndef PIVCO_PRIM_VARIANTS_INIT_H
#define PIVCO_PRIM_VARIANTS_INIT_H

/* scalar_batch (portable): the s2r gather is load-port bound (1 sequential input
 * load + 1 random table load + 1 store per byte).  Load 16 input bytes as 2x u64
 * so the load ports are free for the dependent table loads, and merge to 8x u16
 * stores.  ~2.3x the naive byte loop on M4.  The SSE/AVX2 backend has no usable
 * 256-entry SIMD gather (pshufb is a 16-byte table; vpermi2b is AVX-512 VBMI),
 * so this is the x86 win.  Based on #5 by dougallj. */
static void prim_init_scalar_batch_k(uint8_t *ranks, int n,
                                     const uint8_t *symbols, const uint8_t *s2r) {
#define PIVCO_S2R(x) ((unsigned)s2r[(uint8_t)(x)])
    int i = 0;
    for (; i + 16 <= n; i += 16) {
        uint64_t a, b;
        memcpy(&a, symbols + i,     8);
        memcpy(&b, symbols + i + 8, 8);
        uint16_t h0 = (uint16_t)(PIVCO_S2R(a)       | (PIVCO_S2R(a >> 8)  << 8));
        uint16_t h1 = (uint16_t)(PIVCO_S2R(a >> 16) | (PIVCO_S2R(a >> 24) << 8));
        uint16_t h2 = (uint16_t)(PIVCO_S2R(a >> 32) | (PIVCO_S2R(a >> 40) << 8));
        uint16_t h3 = (uint16_t)(PIVCO_S2R(a >> 48) | (PIVCO_S2R(a >> 56) << 8));
        uint16_t h4 = (uint16_t)(PIVCO_S2R(b)       | (PIVCO_S2R(b >> 8)  << 8));
        uint16_t h5 = (uint16_t)(PIVCO_S2R(b >> 16) | (PIVCO_S2R(b >> 24) << 8));
        uint16_t h6 = (uint16_t)(PIVCO_S2R(b >> 32) | (PIVCO_S2R(b >> 40) << 8));
        uint16_t h7 = (uint16_t)(PIVCO_S2R(b >> 48) | (PIVCO_S2R(b >> 56) << 8));
        memcpy(ranks + i,      &h0, 2); memcpy(ranks + i + 2,  &h1, 2);
        memcpy(ranks + i + 4,  &h2, 2); memcpy(ranks + i + 6,  &h3, 2);
        memcpy(ranks + i + 8,  &h4, 2); memcpy(ranks + i + 10, &h5, 2);
        memcpy(ranks + i + 12, &h6, 2); memcpy(ranks + i + 14, &h7, 2);
    }
    for (; i < n; i++) ranks[i] = s2r[symbols[i]];
#undef PIVCO_S2R
}
static void prim_init_scalar_batch(const ctx_t *c){ prim_init_scalar_batch_k(c->ranks_work, c->n, c->symbuf, c->sym_to_rank); }

/* scalar_batch32: same load-batched gather, but merge to 4x u32 stores (3 merges
 * each) instead of 8x u16 (1 merge each).  Halves the store count again at the
 * cost of more merge ALU; whether that wins depends on store-port pressure. */
static void prim_init_scalar_batch32_k(uint8_t *ranks, int n,
                                       const uint8_t *symbols, const uint8_t *s2r) {
#define PIVCO_S2R(x) ((unsigned)s2r[(uint8_t)(x)])
    int i = 0;
    for (; i + 16 <= n; i += 16) {
        uint64_t a, b;
        memcpy(&a, symbols + i,     8);
        memcpy(&b, symbols + i + 8, 8);
        uint32_t w0 = PIVCO_S2R(a)       | (PIVCO_S2R(a >> 8)  << 8) |
                      (PIVCO_S2R(a >> 16) << 16) | (PIVCO_S2R(a >> 24) << 24);
        uint32_t w1 = PIVCO_S2R(a >> 32) | (PIVCO_S2R(a >> 40) << 8) |
                      (PIVCO_S2R(a >> 48) << 16) | (PIVCO_S2R(a >> 56) << 24);
        uint32_t w2 = PIVCO_S2R(b)       | (PIVCO_S2R(b >> 8)  << 8) |
                      (PIVCO_S2R(b >> 16) << 16) | (PIVCO_S2R(b >> 24) << 24);
        uint32_t w3 = PIVCO_S2R(b >> 32) | (PIVCO_S2R(b >> 40) << 8) |
                      (PIVCO_S2R(b >> 48) << 16) | (PIVCO_S2R(b >> 56) << 24);
        memcpy(ranks + i,      &w0, 4); memcpy(ranks + i + 4,  &w1, 4);
        memcpy(ranks + i + 8,  &w2, 4); memcpy(ranks + i + 12, &w3, 4);
    }
    for (; i < n; i++) ranks[i] = s2r[symbols[i]];
#undef PIVCO_S2R
}
static void prim_init_scalar_batch32(const ctx_t *c){ prim_init_scalar_batch32_k(c->ranks_work, c->n, c->symbuf, c->sym_to_rank); }

/* scalar_2tab: no OR/shift in the merge.  Same u64 input batching as
 * scalar_batch, but the low half stays 8-bit (lo[s]=rank is exactly the existing
 * s2r byte table, so there's no separate lo table) and only a u16 hi[s]=rank<<8
 * table is extra (512 B).  A pair is (u16)s2r[s0] + hi[s1] -- disjoint byte lanes
 * so + == |, with no shift and the hi operand folding into the load+add on x86.
 * hi is built once at table-build time in production; cached on the s2r pointer
 * here so the per-call timing excludes the build. */
static uint16_t *pv_init_hi = 0;   /* heap, not file-static: forces a register-held base like production */
static const uint8_t *pv_init_hi_src = 0;
static void prim_init_2tab_k(uint8_t *ranks, int n,
                             const uint8_t *symbols, const uint8_t *s2r) {
    if (pv_init_hi_src != s2r) {
        if (!pv_init_hi) pv_init_hi = (uint16_t*)malloc(256 * sizeof(uint16_t));
        for (int s = 0; s < 256; s++) pv_init_hi[s] = (uint16_t)((unsigned)s2r[s] << 8);
        pv_init_hi_src = s2r;
    }
#define PV_LO(x) ((uint16_t)s2r[(uint8_t)(x)])
#define PV_HI(x) pv_init_hi[(uint8_t)(x)]
    int i = 0;
    for (; i + 16 <= n; i += 16) {
        uint64_t a, b;
        memcpy(&a, symbols + i,     8);
        memcpy(&b, symbols + i + 8, 8);
        uint16_t h0 = PV_LO(a)       + PV_HI(a >> 8);
        uint16_t h1 = PV_LO(a >> 16) + PV_HI(a >> 24);
        uint16_t h2 = PV_LO(a >> 32) + PV_HI(a >> 40);
        uint16_t h3 = PV_LO(a >> 48) + PV_HI(a >> 56);
        uint16_t h4 = PV_LO(b)       + PV_HI(b >> 8);
        uint16_t h5 = PV_LO(b >> 16) + PV_HI(b >> 24);
        uint16_t h6 = PV_LO(b >> 32) + PV_HI(b >> 40);
        uint16_t h7 = PV_LO(b >> 48) + PV_HI(b >> 56);
        memcpy(ranks + i,      &h0, 2); memcpy(ranks + i + 2,  &h1, 2);
        memcpy(ranks + i + 4,  &h2, 2); memcpy(ranks + i + 6,  &h3, 2);
        memcpy(ranks + i + 8,  &h4, 2); memcpy(ranks + i + 10, &h5, 2);
        memcpy(ranks + i + 12, &h6, 2); memcpy(ranks + i + 14, &h7, 2);
    }
    for (; i < n; i++) ranks[i] = s2r[symbols[i]];
#undef PV_LO
#undef PV_HI
}
static void prim_init_2tab(const ctx_t *c){ prim_init_2tab_k(c->ranks_work, c->n, c->symbuf, c->sym_to_rank); }

/* scalar_4tab: push the no-shift idea to u32 stores.  Four byte-position tables
 * (lo=s2r reused for <<0, plus t8/t16/t24 for <<8/<<16/<<24) let a u32 = 4 ranks
 * be accumulated as lo[s0]+t8[s1]+t16[s2]+t24[s3] -- all adds, the shifts baked
 * into the tables, the three hi operands folding into the adds on x86.  Halves
 * the store count vs 2tab (4x u32 vs 8x u16) at the cost of one more add per quad
 * and a bigger table footprint (t8/t16/t24 are u32 = 3 KB extra). */
static uint32_t *pv_init_t8 = 0, *pv_init_t16 = 0, *pv_init_t24 = 0;  /* heap, not file-static */
static const uint8_t *pv_init_t4_src = 0;
static void prim_init_4tab_k(uint8_t *ranks, int n,
                             const uint8_t *symbols, const uint8_t *s2r) {
    if (pv_init_t4_src != s2r) {
        if (!pv_init_t8) { pv_init_t8=(uint32_t*)malloc(256*4); pv_init_t16=(uint32_t*)malloc(256*4); pv_init_t24=(uint32_t*)malloc(256*4); }
        for (int s = 0; s < 256; s++) {
            uint32_t r = s2r[s];
            pv_init_t8[s]  = r << 8;
            pv_init_t16[s] = r << 16;
            pv_init_t24[s] = r << 24;
        }
        pv_init_t4_src = s2r;
    }
#define PV_W(x0,x8,x16,x24) ( (uint32_t)s2r[(uint8_t)(x0)]      \
                            + pv_init_t8 [(uint8_t)(x8)]        \
                            + pv_init_t16[(uint8_t)(x16)]       \
                            + pv_init_t24[(uint8_t)(x24)] )
    int i = 0;
    for (; i + 16 <= n; i += 16) {
        uint64_t a, b;
        memcpy(&a, symbols + i,     8);
        memcpy(&b, symbols + i + 8, 8);
        uint32_t w0 = PV_W(a,       a >> 8,  a >> 16, a >> 24);
        uint32_t w1 = PV_W(a >> 32, a >> 40, a >> 48, a >> 56);
        uint32_t w2 = PV_W(b,       b >> 8,  b >> 16, b >> 24);
        uint32_t w3 = PV_W(b >> 32, b >> 40, b >> 48, b >> 56);
        memcpy(ranks + i,      &w0, 4); memcpy(ranks + i + 4,  &w1, 4);
        memcpy(ranks + i + 8,  &w2, 4); memcpy(ranks + i + 12, &w3, 4);
    }
    for (; i < n; i++) ranks[i] = s2r[symbols[i]];
#undef PV_W
}
static void prim_init_4tab(const ctx_t *c){ prim_init_4tab_k(c->ranks_work, c->n, c->symbuf, c->sym_to_rank); }

/* bc2: symmetric 2-wide with ONE broadcast table.  bc16[s] = rank * 0x0101 puts
 * the rank in both bytes of a u16; each of the pair picks its byte by mask.
 * Symmetric analog of 2tab's asymmetric lo(u8 s2r) + hi(u16) — one table, at the
 * cost of 2 ANDs + 1 OR per pair (2tab does a single add). */
static uint16_t *pv_init_bc2 = 0;
static const uint8_t *pv_init_bc2_src = 0;
static void prim_init_bc2_k(uint8_t *ranks, int n,
                            const uint8_t *symbols, const uint8_t *s2r) {
    if (pv_init_bc2_src != s2r) {
        if (!pv_init_bc2) pv_init_bc2 = (uint16_t*)malloc(256 * 2);
        for (int s = 0; s < 256; s++) pv_init_bc2[s] = (uint16_t)((unsigned)s2r[s] * 0x0101u);
        pv_init_bc2_src = s2r;
    }
#define PV_B2(x0,x1) ( (uint16_t)( (pv_init_bc2[(uint8_t)(x0)] & 0x00FFu)  \
                                 | (pv_init_bc2[(uint8_t)(x1)] & 0xFF00u) ) )
    int i = 0;
    for (; i + 16 <= n; i += 16) {
        uint64_t a, b;
        memcpy(&a, symbols + i,     8);
        memcpy(&b, symbols + i + 8, 8);
        uint16_t h0 = PV_B2(a,       a >> 8);  uint16_t h1 = PV_B2(a >> 16, a >> 24);
        uint16_t h2 = PV_B2(a >> 32, a >> 40); uint16_t h3 = PV_B2(a >> 48, a >> 56);
        uint16_t h4 = PV_B2(b,       b >> 8);  uint16_t h5 = PV_B2(b >> 16, b >> 24);
        uint16_t h6 = PV_B2(b >> 32, b >> 40); uint16_t h7 = PV_B2(b >> 48, b >> 56);
        memcpy(ranks + i,      &h0, 2); memcpy(ranks + i + 2,  &h1, 2);
        memcpy(ranks + i + 4,  &h2, 2); memcpy(ranks + i + 6,  &h3, 2);
        memcpy(ranks + i + 8,  &h4, 2); memcpy(ranks + i + 10, &h5, 2);
        memcpy(ranks + i + 12, &h6, 2); memcpy(ranks + i + 14, &h7, 2);
    }
    for (; i < n; i++) ranks[i] = s2r[symbols[i]];
#undef PV_B2
}
static void prim_init_bc2(const ctx_t *c){ prim_init_bc2_k(c->ranks_work, c->n, c->symbuf, c->sym_to_rank); }

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
    const stage_t S = ST_ENC_INIT;
    /* Portable batched-scalar gather — runs on every backend (the SSE/AVX2
     * x86 init candidate; NEON ships the simd20 form). */
    PV_VARIANT(S, "scalar_batch", PV_ISA_SCALAR, "issue #5 (dougallj)",
               "16 input bytes as 2x u64 + 8x u16 merge stores; load/store-batched naive gather", 0,
               prim_init_scalar_batch);
    PV_VARIANT(S, "scalar_batch32", PV_ISA_SCALAR, "issue #5 (dougallj)",
               "16 input bytes as 2x u64 + 4x u32 merge stores (3 merges each)", 0,
               prim_init_scalar_batch32);
    PV_VARIANT(S, "scalar_2tab", PV_ISA_SCALAR, "issue #5 (dougallj)",
               "no-OR merge: lo = existing u8 s2r, only hi[s]=rank<<8 is a new u16 table; pair = (u16)s2r[s0]+hi[s1]", 0,
               prim_init_2tab);
    PV_VARIANT(S, "scalar_4tab", PV_ISA_SCALAR, "issue #5 (dougallj)",
               "no-shift u32 store: lo=s2r + t8/t16/t24 (rank<<8/16/24); quad = lo[s0]+t8[s1]+t16[s2]+t24[s3]", 0,
               prim_init_4tab);
    PV_VARIANT(S, "scalar_bc2", PV_ISA_SCALAR, "issue #5 (dougallj)",
               "1-table symmetric 2-wide: bc16[s]=rank*0x0101, u16 = mask+or per pair", 0,
               prim_init_bc2);
    /* simd20 (the 20 sym/iter interleaved form) is production init_neon; this
     * keeps the pure-SIMD 16 sym/iter form for posterity / per-uarch re-eval. */
    PV_VARIANT(S, "simd16", PV_ISA_NEON, "issue #5 (dougallj)",
               "256-entry s2r gather via vqtbl4 + 3x vqtbx4, 16 sym/iter (pure SIMD); production uses the 20 sym/iter interleaved form", 0,
               PV_FN_NEON(prim_init_simd16));
}

#endif /* PIVCO_PRIM_VARIANTS_INIT_H */
