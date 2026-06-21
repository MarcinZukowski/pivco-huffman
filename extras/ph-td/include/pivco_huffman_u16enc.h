/* extras/ph-td/include/pivco_huffman_u16enc.h — retired u16 (code_la) encode.
 *
 * The partbyrank u8 encode is the production path (src/ prim_enc_*).  These are
 * the old 16-bit left-aligned-code encode kernels, removed from the production
 * codec.  They have two remaining consumers, both of which include THIS header
 * (and the shared primitives header FIRST):
 *   - ph-td (extras/ph-td): the retired top-down codec, which round-trips via
 *     the u16 encoder; ph-td owns this header.
 *   - bench_prim: the u16enc_* rows (benched alongside the u8 path).
 *
 * Per-ISA #if dispatch (ph-td uses PIVCO_HAS_*, bench uses the router's
 * PIVCO_BACKEND_* — so gate on the architecture macro common to both).  Relies
 * on the shared primitives header providing compress_tab (neon/x86 tables), the
 * *_pack.h per-D helpers, bitmap_bytes(), the intrinsics, and
 * PIVCO_PRIM_ALWAYS_INLINE.  Moved verbatim from src/ (the src/ diff shows the
 * u16 -> u8 promotion). */
#ifndef PIVCO_HUFFMAN_U16ENC_H
#define PIVCO_HUFFMAN_U16ENC_H

#if defined(__aarch64__)

/* Retired u16 D5/6/7 ryg pack (moved out of pivco_huffman_neon_pack.h; uses the
 * shared pivco_pack_compact_d{5,6,7}_neon tables still defined there). */
/* Load 16 left-aligned u16 codes, right-shift, narrow to one u8x16
 * (1 code/byte), mask to D bits.  Lane k holds codes_la[base + k]. */
static inline uint8x16_t
pivco_u16pack_load_neon(const uint16_t *codes_la, int right_shift,
                                 uint8_t code_mask)
{
    int16x8_t rsh = vdupq_n_s16((int16_t)-right_shift);
    uint16x8_t a = vshlq_u16(vld1q_u16(codes_la    ), rsh);
    uint16x8_t b = vshlq_u16(vld1q_u16(codes_la + 8), rsh);
    /* vmovn_u16 = truncate (low 8 bits, no saturation).  D <= 7, so the
     * shifted code already fits in 8 bits — the mask below cleans any
     * stray high bits from the original codes_la lane. */
    uint8x16_t bytes = vcombine_u8(vmovn_u16(a), vmovn_u16(b));
    return vandq_u8(bytes, vdupq_n_u8(code_mask));
}

/* D as compile-time constant so vshrq_n_u64 / mask constants fold. */
#define PIVCO_U16PACK_NEON_DN(NAME, D_VAL, COMPACT_TAB)                            \
static inline int NAME(uint8_t *out, const uint16_t *codes_la,                 \
                       int n, int right_shift)                                  \
{                                                                                \
    const uint8x16_t c0 = vreinterpretq_u8_u16(                                  \
        vdupq_n_u16((uint16_t)(((1u << (D_VAL)) << 8) | 1u)));                   \
    const uint16x8_t c1 = vreinterpretq_u16_u32(                                 \
        vdupq_n_u32((uint32_t)(((1u << (2*(D_VAL))) << 16) | 1u)));              \
    const uint64x2_t c3 = vdupq_n_u64(((uint64_t)1 << (4*(D_VAL))) - 1);         \
    const uint8x16_t compact = vld1q_u8(COMPACT_TAB);                            \
    int i = 0;                                                                   \
    for (; i + 16 <= n; i += 16) {                                               \
        uint8x16_t cb = pivco_u16pack_load_neon(                         \
            codes_la + i, right_shift, (uint8_t)((1u << (D_VAL)) - 1u));         \
        /* Step 1: word[i] = cb[2i] + cb[2i+1] * 2^D  (8 u16 lanes)   */         \
        uint16x8_t prod_lo = vmull_u8(vget_low_u8(cb),  vget_low_u8(c0));        \
        uint16x8_t prod_hi = vmull_high_u8(cb, c0);                              \
        uint16x8_t w = vpaddq_u16(prod_lo, prod_hi);                             \
        /* Step 2: dword[i] = word[2i] + word[2i+1] * 2^(2D)  (4 u32 lanes) */   \
        uint32x4_t prod32_lo = vmull_u16(vget_low_u16(w),  vget_low_u16(c1));    \
        uint32x4_t prod32_hi = vmull_high_u16(w, c1);                            \
        uint32x4_t d  = vpaddq_u32(prod32_lo, prod32_hi);                        \
        /* Step 3: per-u64 lane, merge dword[2i+1] (right-shifted) with         \
         * dword[2i].  After srli by (32 - 4D): the high-32 dword sits at       \
         * bits [4D..4D+31].  Mask keeps low 4D bits of x, takes high 4D bits  \
         * from xs — together 8D bits per u64.                                  */ \
        uint64x2_t x  = vreinterpretq_u64_u32(d);                                \
        uint64x2_t xs = vshrq_n_u64(x, 32 - 4*(D_VAL));                          \
        uint64x2_t m  = vorrq_u64(vandq_u64(x, c3),                              \
                                   vbicq_u64(xs, c3));                           \
        /* Step 4: compact 2D consecutive bytes per 128-bit lane.   */           \
        uint8x16_t packed = vqtbl1q_u8(vreinterpretq_u8_u64(m), compact);        \
        vst1q_u8(out + ((i * (D_VAL)) >> 3), packed);                            \
    }                                                                            \
    return i;                                                                    \
}
PIVCO_U16PACK_NEON_DN(u16pack_d5_neon, 5, pivco_pack_compact_d5_neon)
PIVCO_U16PACK_NEON_DN(u16pack_d6_neon, 6, pivco_pack_compact_d6_neon)
PIVCO_U16PACK_NEON_DN(u16pack_d7_neon, 7, pivco_pack_compact_d7_neon)
#undef PIVCO_U16PACK_NEON_DN
/* ---------- Encode primitives (bitmap + partition) ----------
 *
 * The non-flat-internal-node hot path.  Builds the n-bit partition
 * bitmap from codes_la[0..n) (each codes_la[i] is the per-symbol left-
 * aligned Huffman code; bit (15 - depth) is the current depth's
 * partition decision) and partitions codes_la in place: left (bit==0)
 * stays in codes_la[0..n_left), right (bit==1) moves to right_out[0..n_right).
 * codes_la lanes are written through to next-level recursion unchanged
 * -- the codes_la representation is depth-threaded, NOT shifted across
 * levels.
 *
 * See pivco_huffman_primitives.h for the codec.c boundary convention.
 */

/* Dense movmask helper: given 8 left-aligned codes and a negative shift
 * amount = -(15 - depth), produce the 8-bit partition mask for this
 * batch.  Right-shifts each lane by (15-depth) so the partition bit
 * lands in the LSB, then horizontal-add weighted by 2^k.
 * Cost: 4 NEON ops (shl, and, shl, addv) per 8 codes. */
static inline uint8_t enc_mask8_codes_la_neon(uint16x8_t code_vec,
                                                int neg_shift_d)
{
    int16x8_t shr_vec = vdupq_n_s16((int16_t)neg_shift_d);
    uint16x8_t bit_lsb = vandq_u16(vshlq_u16(code_vec, shr_vec),
                                    vdupq_n_u16(1));
    static const int16_t weights[8] = {0, 1, 2, 3, 4, 5, 6, 7};
    uint16x8_t weighted = vshlq_u16(bit_lsb, vld1q_s16(weights));
    return (uint8_t)vaddvq_u16(weighted);
}

/* enc_masks8x8_codes_la_neon — build the partition masks for EIGHT 8-code
 * chunks at once, packed LE into a u64 (byte k = mask of chunk k).
 *
 * Per chunk: vtstq_u16(code, 1<<shift_d) -> 0x0000/0xFFFF per lane, AND with
 * the {1,2,4,..,128} bit-weights.  Then a vpaddq_u16 tree (4+2+1 = 7 ops)
 * reduces the 8 weighted vectors to one uint16x8 whose lane k is mask_k --
 * replacing 8 lane-crossing vaddvq reductions + 8 SIMD->GPR moves with 7
 * vpaddq + 1 vmovn + 1 fmov.  Big win on Graviton's narrow pipes (the
 * 8 addv were the bottleneck), neutral on Apple.  See the ARM "porting
 * x86 movemask to NEON" pairwise-reduction technique.  shift_d = 15-depth. */
static inline uint64_t enc_masks8x8_codes_la_neon(
        uint16x8_t c0, uint16x8_t c1, uint16x8_t c2, uint16x8_t c3,
        uint16x8_t c4, uint16x8_t c5, uint16x8_t c6, uint16x8_t c7,
        int shift_d)
{
    uint16x8_t bit = vdupq_n_u16((uint16_t)(1u << shift_d));
    static const uint16_t powers_arr[8] = {1,2,4,8,16,32,64,128};
    uint16x8_t pw = vld1q_u16(powers_arr);
    uint16x8_t w0=vandq_u16(vtstq_u16(c0,bit),pw), w1=vandq_u16(vtstq_u16(c1,bit),pw),
               w2=vandq_u16(vtstq_u16(c2,bit),pw), w3=vandq_u16(vtstq_u16(c3,bit),pw),
               w4=vandq_u16(vtstq_u16(c4,bit),pw), w5=vandq_u16(vtstq_u16(c5,bit),pw),
               w6=vandq_u16(vtstq_u16(c6,bit),pw), w7=vandq_u16(vtstq_u16(c7,bit),pw);
    uint16x8_t p01=vpaddq_u16(w0,w1), p23=vpaddq_u16(w2,w3),
               p45=vpaddq_u16(w4,w5), p67=vpaddq_u16(w6,w7);
    uint16x8_t q0=vpaddq_u16(p01,p23), q1=vpaddq_u16(p45,p67);
    uint16x8_t r=vpaddq_u16(q0,q1);   /* lane k = mask_k */
    return vget_lane_u64(vreinterpret_u64_u8(vmovn_u16(r)), 0);
}

/* u16part_core_neon — the single shared partition loop.  Parameterized by
 * compile-time-constant flags so each always-inline wrapper folds to a
 * specialized branch-free loop with no dead stores:
 *   BUILD=1  build the bitmap from codes_la's depth bit (fused, encode)
 *   BUILD=0  read the mask from bm_in (from-bitmap; future TD-decode share)
 *   EMIT_RIGHT/EMIT_LEFT  scatter that half (full=1,1 right=1,0 left=0,1 none=0,0)
 * The stride-8 scatter (the "partition8" step) is inlined here rather than
 * factored into a helper — a helper-call boundary cost the FULL path ~9% on
 * M4 because the compiler stopped folding the cursor math.  LEFT is written in
 * place over codes_la (n_left <= j keeps the 16-byte store safe); RIGHT goes to
 * right_out.  Returns n_right. */
/* u16part_full_neon — the FULL (both-sides) fused partition,
 * kept hand-written.  The generic u16part_core_neon below (with EMIT_RIGHT=
 * EMIT_LEFT=1) is logically identical but schedules ~8% slower on M4 for this
 * specific 1,1,1 case — and FULL is the hot common path, so it stays
 * specialized.  part_core handles every other variant (right/left/none and
 * the from-bitmap share) where the generic form matches hand-written speed. */
static inline int u16part_full_neon(uint16_t *codes_la, int n,
                                                    int depth, uint8_t *bm,
                                                    uint16_t *right_out)
{
    int n_left = 0, n_right = 0, j = 0;
    int neg_shift_d = -(15 - depth);

    /* V5 wide path: 64 codes/iter, 8 independent chunks.  Mirrors the
     * COM64 merge idea on the encode side (Jeff Plaisance).  Load all 8
     * code_vecs first so the in-place left compaction has no read-after-
     * write hazard, build all 8 bitmap bytes (one 8-byte store instead of
     * eight 1-byte stores), then a vcnt + 0x0101.. prefix sum precomputes
     * every chunk's left/right scatter cursor -- so the 8 compact+scatter
     * chunks are independent (no serial n_left/n_right add chain) and the
     * 8 compress_popcnt[mask] loads are replaced by one vcnt.
     * The 8-code loop below handles the n mod 64 residual. */
    for (; j + 64 <= n; j += 64) {
        uint16x8_t cv0=vld1q_u16(codes_la+j),    cv1=vld1q_u16(codes_la+j+8),
                   cv2=vld1q_u16(codes_la+j+16), cv3=vld1q_u16(codes_la+j+24),
                   cv4=vld1q_u16(codes_la+j+32), cv5=vld1q_u16(codes_la+j+40),
                   cv6=vld1q_u16(codes_la+j+48), cv7=vld1q_u16(codes_la+j+56);
        uint64_t mask_word = enc_masks8x8_codes_la_neon(
            cv0,cv1,cv2,cv3,cv4,cv5,cv6,cv7, 15 - depth);
        uint8_t m0=(uint8_t)mask_word,        m1=(uint8_t)(mask_word>>8),
                m2=(uint8_t)(mask_word>>16),  m3=(uint8_t)(mask_word>>24),
                m4=(uint8_t)(mask_word>>32),  m5=(uint8_t)(mask_word>>40),
                m6=(uint8_t)(mask_word>>48),  m7=(uint8_t)(mask_word>>56);
        memcpy(bm + (j >> 3), &mask_word, 8);
        uint8x8_t pc_v = vcnt_u8(vcreate_u8(mask_word));
        uint64_t pc_word = vget_lane_u64(vreinterpret_u64_u8(pc_v), 0);
        uint64_t pfx = pc_word * 0x0101010101010101ULL;
#define _V5_PART(K_, CV, M) do {                                              \
        uint32_t cr = (K_)==0 ? 0u : (uint32_t)((pfx >> (8*((K_)-1))) & 0xFF); \
        uint32_t cl = 8u*(K_) - cr;                                           \
        const uint8_t *tab = compress_tab[(M)];                               \
        uint8x16_t data  = vreinterpretq_u8_u16(CV);                          \
        uint8x16_t right = vqtbl1q_u8(data, vld1q_u8(tab));                    \
        uint8x16_t left  = vqtbl1q_u8(data, vld1q_u8(tab + 16));               \
        vst1q_u8((uint8_t *)(right_out + n_right + cr), right);               \
        vst1q_u8((uint8_t *)(codes_la  + n_left  + cl), left);                \
    } while (0)
        _V5_PART(0,cv0,m0); _V5_PART(1,cv1,m1); _V5_PART(2,cv2,m2); _V5_PART(3,cv3,m3);
        _V5_PART(4,cv4,m4); _V5_PART(5,cv5,m5); _V5_PART(6,cv6,m6); _V5_PART(7,cv7,m7);
#undef _V5_PART
        uint32_t total_r = (uint32_t)(pfx >> 56);
        n_right += total_r; n_left += 64 - total_r;
    }

    for (; j + 8 <= n; j += 8) {
        uint16x8_t code_vec = vld1q_u16(codes_la + j);
        uint8_t mask = enc_mask8_codes_la_neon(code_vec, neg_shift_d);
        bm[j >> 3] = mask;
        const uint8_t *tab = compress_tab[mask];
        uint8x16_t shuf_r = vld1q_u8(tab);
        uint8x16_t shuf_l = vld1q_u8(tab + 16);
        uint8x16_t data   = vreinterpretq_u8_u16(code_vec);
        uint8x16_t right  = vqtbl1q_u8(data, shuf_r);
        uint8x16_t left   = vqtbl1q_u8(data, shuf_l);
        int nr = compress_popcnt[mask];
        vst1q_u8((uint8_t *)(right_out + n_right), right);
        vst1q_u8((uint8_t *)(codes_la  + n_left ), left);
        n_right += nr;
        n_left  += (8 - nr);
    }
    if (j < n) {
        int tail = n - j, shift_d = 15 - depth;
        uint16_t tail_buf[8];
        for (int k = 0; k < tail; k++) tail_buf[k] = codes_la[j + k];
        uint8_t mask = 0;
        for (int k = 0; k < tail; k++)
            mask |= (uint8_t)(((tail_buf[k] >> shift_d) & 1) << k);
        bm[j >> 3] = mask;
        for (int k = 0; k < tail; k++) {
            if (mask & (1 << k)) right_out[n_right++] = tail_buf[k];
            else                 codes_la[n_left++]   = tail_buf[k];
        }
    }
    return n_right;
}

__attribute__((always_inline)) static inline
int u16part_core_neon(uint16_t *codes_la, int n, int depth,
                                  uint8_t *bm, const uint8_t *bm_in,
                                  uint16_t *right_out,
                                  int BUILD, int EMIT_RIGHT, int EMIT_LEFT)
{
    int n_left = 0, n_right = 0, j = 0;
    int neg_shift_d = -(15 - depth);

    /* V5 wide path: 64 codes/iter, 8 independent chunks -- same prefix-sum
     * cursor decoupling + vpaddq 8-mask build as build_bitmap_partition_full_
     * neon, specialized here for the one-sided (right/left/none) emit cases.
     * BUILD=1 only (all live callers build the bitmap; the BUILD=0 from-bitmap
     * share is unused, falls through to the 8-code loop).  EMIT_RIGHT/
     * EMIT_LEFT are compile-time, so the unused scatter + cursor drop out. */
    if (BUILD) {
        int shift_d = 15 - depth;
        for (; j + 64 <= n; j += 64) {
            uint16x8_t cv0=vld1q_u16(codes_la+j),    cv1=vld1q_u16(codes_la+j+8),
                       cv2=vld1q_u16(codes_la+j+16), cv3=vld1q_u16(codes_la+j+24),
                       cv4=vld1q_u16(codes_la+j+32), cv5=vld1q_u16(codes_la+j+40),
                       cv6=vld1q_u16(codes_la+j+48), cv7=vld1q_u16(codes_la+j+56);
            uint64_t mask_word = enc_masks8x8_codes_la_neon(
                cv0,cv1,cv2,cv3,cv4,cv5,cv6,cv7, shift_d);
            memcpy(bm + (j >> 3), &mask_word, 8);
            uint8x8_t pc_v = vcnt_u8(vcreate_u8(mask_word));
            uint64_t pc_word = vget_lane_u64(vreinterpret_u64_u8(pc_v), 0);
            uint64_t pfx = pc_word * 0x0101010101010101ULL;
#define _V5_PART1(K_, CV) do {                                                \
            uint32_t cr = (K_)==0 ? 0u : (uint32_t)((pfx >> (8*((K_)-1))) & 0xFF);\
            if (EMIT_RIGHT || EMIT_LEFT) {                                     \
                uint8_t M = (uint8_t)(mask_word >> (8*(K_)));                  \
                const uint8_t *tab = compress_tab[M];                         \
                uint8x16_t data = vreinterpretq_u8_u16(CV);                   \
                if (EMIT_RIGHT) vst1q_u8((uint8_t *)(right_out + n_right + cr),\
                                          vqtbl1q_u8(data, vld1q_u8(tab)));     \
                if (EMIT_LEFT)  vst1q_u8((uint8_t *)(codes_la + n_left         \
                                          + (8u*(K_) - cr)),                   \
                                          vqtbl1q_u8(data, vld1q_u8(tab + 16)));\
            }                                                                 \
        } while (0)
            _V5_PART1(0,cv0); _V5_PART1(1,cv1); _V5_PART1(2,cv2); _V5_PART1(3,cv3);
            _V5_PART1(4,cv4); _V5_PART1(5,cv5); _V5_PART1(6,cv6); _V5_PART1(7,cv7);
#undef _V5_PART1
            uint32_t total_r = (uint32_t)(pfx >> 56);
            n_right += total_r; n_left += 64 - total_r;
        }
    }

    for (; j + 8 <= n; j += 8) {
        uint16x8_t code_vec = vld1q_u16(codes_la + j);
        uint8_t mask;
        if (BUILD) { mask = enc_mask8_codes_la_neon(code_vec, neg_shift_d); bm[j >> 3] = mask; }
        else         mask = bm_in[j >> 3];
        const uint8_t *tab = compress_tab[mask];
        uint8x16_t data = vreinterpretq_u8_u16(code_vec);
        if (EMIT_RIGHT) vst1q_u8((uint8_t *)(right_out + n_right),
                                 vqtbl1q_u8(data, vld1q_u8(tab)));
        if (EMIT_LEFT)  vst1q_u8((uint8_t *)(codes_la + n_left),
                                 vqtbl1q_u8(data, vld1q_u8(tab + 16)));
        int nr = compress_popcnt[mask];
        n_right += nr;
        n_left  += 8 - nr;
    }
    if (j < n) {
        int tail = n - j, shift_d = 15 - depth;
        uint16_t tail_buf[8];
        for (int k = 0; k < tail; k++) tail_buf[k] = codes_la[j + k];
        uint8_t mask;
        if (BUILD) {
            mask = 0;
            for (int k = 0; k < tail; k++)
                mask |= (uint8_t)(((tail_buf[k] >> shift_d) & 1) << k);
            bm[j >> 3] = mask;
        } else mask = bm_in[j >> 3];
        for (int k = 0; k < tail; k++) {
            if (mask & (1 << k)) { if (EMIT_RIGHT) right_out[n_right] = tail_buf[k]; n_right++; }
            else                 { if (EMIT_LEFT)  codes_la[n_left]   = tail_buf[k]; n_left++;  }
        }
    }
    return n_right;
}

/* ---------- Encode primitives (init) ----------
 *
 * u16init_neon — gather per-symbol left-aligned codes into codes_la.
 * `code_la_lut` is table->code_la (256 uint16 entries).
 *
 * Today this is a straight scalar loop; the compiler often auto-
 * vectorises it via vqtbl1q_u8 over a 256-entry LUT, but the codegen
 * is fragile and the LSU is the bottleneck in either form (microbench
 * at extras/bench/bench_enc_init.c established the NEON TBL pattern buys
 * only ~11% over the scalar loop on M4 -- not worth the source
 * complexity).  Kept here as a primitive so AVX-512's actual SIMD
 * win via vpermi2w (commit 7c08c19) has a contract slot to fill. */
static inline void u16init_neon(uint16_t *codes_la, int n,
                                   const uint8_t *symbols,
                                   const uint16_t *code_la_lut)
{
    for (int i = 0; i < n; i++) codes_la[i] = code_la_lut[symbols[i]];
}

/* ---------- Encode primitives (flat-subtree pack) ----------
 *
 * Per-D SIMD bit-pack helpers.  Each reads D-bit codes from codes_la
 * (each lane holds the left-aligned Huffman code -- bit-d of the
 * original code is at position 15-d) and packs them LSB-first into
 * the output byte stream.
 *
 * The legacy ergonomic: each `u16pack_dN_neon` helper takes `right_shift
 * = 16 - depth - D`, applies it via vshlq_u16 with a runtime negative
 * shift vector, ANDs to (1<<D)-1, then packs.  Each helper OVERPACKS
 * (processes ceil(n / stride) * stride elements) so callers can drop
 * the scalar tail entirely if they pre-zero `codes_la[n .. n+15]`.
 * The dispatcher pack_dN_dispatch_neon below handles the residual
 * scalar tail when overpacking isn't possible.
 *
 * For the codec.c contract, u16enc_pack_dN(codes_la, n, D, depth, out_packed)
 * forwards to u16pack_dN_neon(out_packed, codes_la, n, D, depth). */

/* D=2: 16 codes -> 4 bytes (4 codes per byte, no byte crossings). */
static inline int u16pack_d2_neon(uint8_t *out, const uint16_t *codes_la,
                                 int n, int right_shift)
{
    static const int8_t shifts_d2[16] = {
        0, 2, 4, 6,  0, 2, 4, 6,  0, 2, 4, 6,  0, 2, 4, 6
    };
    int i = 0;
    for (; i + 16 <= n; i += 16) {
        uint16x8_t v0 = vshlq_u16(vld1q_u16(codes_la + i    ),
                                   vdupq_n_s16((int16_t)-right_shift));
        uint16x8_t v1 = vshlq_u16(vld1q_u16(codes_la + i + 8),
                                   vdupq_n_s16((int16_t)-right_shift));
        v0 = vandq_u16(v0, vdupq_n_u16(0x3));
        v1 = vandq_u16(v1, vdupq_n_u16(0x3));
        uint8x16_t bytes = vcombine_u8(vmovn_u16(v0), vmovn_u16(v1));
        bytes = vshlq_u8(bytes, vld1q_s8(shifts_d2));
        /* Sum groups of 4 via two paired-adds; low 4 lanes = 4 output bytes. */
        uint8x16_t s1 = vpaddq_u8(bytes, bytes);
        uint8x16_t s2 = vpaddq_u8(s1, s1);
        uint32_t packed4 = vgetq_lane_u32(vreinterpretq_u32_u8(s2), 0);
        memcpy(out + (i * 2 / 8), &packed4, 4);
    }
    return i;
}

/* D=3: 8 codes -> 24 bits.  Cross byte boundaries; uint32 accumulator
 * (max shift 7*3 = 21).  Overpacks to ceil(n/8)*8. */
static inline int u16pack_d3_neon(uint8_t *out, const uint16_t *codes_la,
                                 int n, int right_shift)
{
    static const int32_t shifts_lo[4] = {0,   3,  6,  9};
    static const int32_t shifts_hi[4] = {12, 15, 18, 21};
    int i = 0;
    for (; i + 8 <= n; i += 8) {
        uint16x8_t v = vshlq_u16(vld1q_u16(codes_la + i),
                                  vdupq_n_s16((int16_t)-right_shift));
        v = vandq_u16(v, vdupq_n_u16(0x7));
        uint32x4_t lo = vshlq_u32(vmovl_u16(vget_low_u16(v)),
                                   vld1q_s32(shifts_lo));
        uint32x4_t hi = vshlq_u32(vmovl_u16(vget_high_u16(v)),
                                   vld1q_s32(shifts_hi));
        uint32x4_t sum = vaddq_u32(lo, hi);
        uint32_t packed = vaddvq_u32(sum);
        int bi = i * 3 / 8;
        out[bi    ] = (uint8_t)(packed       & 0xff);
        out[bi + 1] = (uint8_t)((packed >> 8 ) & 0xff);
        out[bi + 2] = (uint8_t)((packed >> 16) & 0xff);
    }
    return i;
}

/* D=4: 16 codes -> 8 bytes.  Pair (c[2k], c[2k+1]) into one byte each. */
static inline int u16pack_d4_neon(uint8_t *out, const uint16_t *codes_la,
                                 int n, int right_shift)
{
    static const int8_t shifts_d4[16] = {
        0, 4, 0, 4, 0, 4, 0, 4, 0, 4, 0, 4, 0, 4, 0, 4
    };
    int i = 0;
    for (; i + 16 <= n; i += 16) {
        uint16x8_t v0 = vshlq_u16(vld1q_u16(codes_la + i    ),
                                   vdupq_n_s16((int16_t)-right_shift));
        uint16x8_t v1 = vshlq_u16(vld1q_u16(codes_la + i + 8),
                                   vdupq_n_s16((int16_t)-right_shift));
        v0 = vandq_u16(v0, vdupq_n_u16(0xF));
        v1 = vandq_u16(v1, vdupq_n_u16(0xF));
        uint8x16_t bytes = vcombine_u8(vmovn_u16(v0), vmovn_u16(v1));
        bytes = vshlq_u8(bytes, vld1q_s8(shifts_d4));
        uint8x16_t paired = vpaddq_u8(bytes, bytes);
        vst1_u8(out + (i * 4 / 8), vget_low_u8(paired));
    }
    return i;
}

/* D=5/6/7 pack moved to pivco_huffman_neon_pack.h (ryg multiply-as-shift,
 * 16 codes/iter via byte-laid intermediate). */

/* D=8: 16 codes -> 16 bytes.  Byte-aligned; one shift+AND pass. */
static inline int u16pack_d8_neon(uint8_t *out, const uint16_t *codes_la,
                                 int n, int right_shift)
{
    int i = 0;
    for (; i + 16 <= n; i += 16) {
        uint16x8_t v0 = vshlq_u16(vld1q_u16(codes_la + i    ),
                                   vdupq_n_s16((int16_t)-right_shift));
        uint16x8_t v1 = vshlq_u16(vld1q_u16(codes_la + i + 8),
                                   vdupq_n_s16((int16_t)-right_shift));
        uint8x16_t bytes = vcombine_u8(vmovn_u16(v0), vmovn_u16(v1));
        vst1q_u8(out + i, bytes);
    }
    return i;
}

/* Dispatcher: pack n D-bit codes from codes_la into out[].  Selects the
 * SIMD per-D pack helper, then handles any residual scalar tail (the
 * per-D helpers stride at 8 or 16; callers that haven't pre-zero-padded
 * codes_la beyond n need the tail).
 *
 * codec.c contract: codes_la is the per-block left-aligned-codes array
 * (NOT mutated across recursion levels), `depth` is the current tree
 * depth.  The D-bit local code at a flat-subtree node lives at
 * positions [15-depth .. 15-depth-D+1] of each codes_la[i] = bits
 * shifted right by (16 - depth - D). */
static inline void u16pack_dN_neon(uint8_t *out, const uint16_t *codes_la,
                                  int n, int D, int depth)
{
    int total_bytes = (n * D + 7) >> 3;
    if (total_bytes > 0) out[total_bytes - 1] = 0;
    int right_shift = 16 - depth - D;

    int i = 0;
    switch (D) {
    case 2: i = u16pack_d2_neon(out, codes_la, n, right_shift); break;
    case 3: i = u16pack_d3_neon(out, codes_la, n, right_shift); break;
    case 4: i = u16pack_d4_neon(out, codes_la, n, right_shift); break;
    case 5: i = u16pack_d5_neon(out, codes_la, n, right_shift); break;
    case 6: i = u16pack_d6_neon(out, codes_la, n, right_shift); break;
    case 7: i = u16pack_d7_neon(out, codes_la, n, right_shift); break;
    case 8: i = u16pack_d8_neon(out, codes_la, n, right_shift); break;
    default: break;  /* D >= 9: scalar tail below handles it
                      * (shouldn't happen with PIVCO_MAX_CODE_LEN = 11) */
    }

    /* With overpacking, i = ceil(n / stride) * stride >= n.  Clamp the
     * counter to avoid uint64 underflow in PROF_COUNT_ONLY. */
    int simd_n = i > n ? n : i;
    PROF_COUNT_ONLY(PROF_ENC_FLAT_SIMD_ELEMS, simd_n);
    PROF_COUNT_ONLY(PROF_ENC_FLAT_TAIL_ELEMS, n - simd_n);
    (void)simd_n;  /* unused when PIVCO_PROF=0 */

    if (i >= n) return;

    /* Scalar tail: pick up where the SIMD path left off. */
    uint32_t mask = (1u << D) - 1;
    int bit_pos = i * D;
    int byte_idx = bit_pos >> 3;
    int bits_in_buf = bit_pos & 7;
    uint64_t buf = bits_in_buf > 0
        ? (uint64_t)out[byte_idx] & ((1u << bits_in_buf) - 1)
        : 0;
    for (; i < n; i++) {
        uint32_t local = ((uint32_t)codes_la[i] >> right_shift) & mask;
        buf |= ((uint64_t)local) << bits_in_buf;
        bits_in_buf += D;
        while (bits_in_buf >= 8) {
            out[byte_idx++] = (uint8_t)(buf & 0xff);
            buf >>= 8;
            bits_in_buf -= 8;
        }
    }
    if (bits_in_buf > 0) {
        out[byte_idx] = (uint8_t)(buf & ((1u << bits_in_buf) - 1));
    }
}

PIVCO_PRIM_ALWAYS_INLINE void u16enc_init(uint16_t *codes_la, int n,
                                              const uint8_t *symbols,
                                              const uint16_t *code_la_lut)
{ u16init_neon(codes_la, n, symbols, code_la_lut); }

PIVCO_PRIM_ALWAYS_INLINE int u16enc_partition_full(uint16_t *codes_la,
                                                      int n, int depth,
                                                      uint8_t *bm,
                                                      uint16_t *right_out)
{ return u16part_full_neon(codes_la, n, depth, bm, right_out); }

PIVCO_PRIM_ALWAYS_INLINE int u16enc_partition_right(uint16_t *codes_la,
                                                      int n, int depth,
                                                      uint8_t *bm,
                                                      uint16_t *right_out)
{ return u16part_core_neon(codes_la, n, depth, bm, NULL, right_out, 1, 1, 0); }

PIVCO_PRIM_ALWAYS_INLINE int u16enc_partition_left(uint16_t *codes_la,
                                                     int n, int depth,
                                                     uint8_t *bm)
{ return u16part_core_neon(codes_la, n, depth, bm, NULL, NULL, 1, 0, 1); }

PIVCO_PRIM_ALWAYS_INLINE int u16enc_partition_none(uint16_t *codes_la,
                                                     int n, int depth,
                                                     uint8_t *bm)
{ return u16part_core_neon(codes_la, n, depth, bm, NULL, NULL, 1, 0, 0); }

PIVCO_PRIM_ALWAYS_INLINE void u16enc_pack_dN(const uint16_t *codes_la,
                                             int n, int D, int depth,
                                             uint8_t *out_packed)
{ u16pack_dN_neon(out_packed, codes_la, n, D, depth); }
#endif /* __aarch64__ */


#if defined(__SSE4_1__) && !defined(__AVX512VBMI2__)

#ifdef PIVCO_HAS_AVX2
/* Retired u16 AVX2 ryg pack (moved out of pivco_huffman_avx2_pack.h; uses the
 * shared PIVCO_PACK_AVX2_COMPACT_D* macros still defined there).  Gated on
 * PIVCO_HAS_AVX2 to match where those macros (and the dispatcher below) live. */
/* Load 32 left-aligned u16 codes, right-shift, narrow to one ymm of 32
 * bytes (1 code/byte), mask to D bits.
 *
 * NB: vpackuswb SATURATES (u16 > 255 -> 255).  We mask the high byte of
 * each u16 to zero before the pack so it behaves like truncation. */
static inline __m256i pivco_u16pack_load_avx2(const uint16_t *codes_la,
                                                       int right_shift,
                                                       uint8_t code_mask)
{
    const __m256i lo_byte_mask = _mm256_set1_epi16(0x00FF);
    __m256i a = _mm256_loadu_si256((const __m256i *)(codes_la));
    __m256i b = _mm256_loadu_si256((const __m256i *)(codes_la + 16));
    a = _mm256_and_si256(_mm256_srli_epi16(a, right_shift), lo_byte_mask);
    b = _mm256_and_si256(_mm256_srli_epi16(b, right_shift), lo_byte_mask);
    /* vpackuswb on ymm operates per-128-bit-lane; permute to fix order. */
    __m256i packed = _mm256_packus_epi16(a, b);
    __m256i bytes  = _mm256_permute4x64_epi64(packed, 0xD8);
    return _mm256_and_si256(bytes, _mm256_set1_epi8((char)code_mask));
}

#define PIVCO_U16PACK_AVX2_DN(NAME, D_VAL, COMPACT_SHUF)                           \
static inline int NAME(uint8_t *out, const uint16_t *codes_la,                  \
                       int n, int right_shift)                                   \
{                                                                                \
    const __m256i c0 = _mm256_set1_epi16(                                        \
        (int16_t)(((1 << (D_VAL)) << 8) | 1));                                   \
    const __m256i c1 = _mm256_set1_epi32(                                        \
        (int32_t)(((int32_t)1 << (2*(D_VAL))) << 16) | 1);                       \
    const __m256i c3 = _mm256_set1_epi64x(                                       \
        (int64_t)(((int64_t)1 << (4*(D_VAL))) - 1));                             \
    const __m256i compact = COMPACT_SHUF;                                        \
    int i = 0;                                                                   \
    for (; i + 32 <= n; i += 32) {                                               \
        __m256i cb = pivco_u16pack_load_avx2(                            \
            codes_la + i, right_shift, (uint8_t)((1 << (D_VAL)) - 1));           \
        __m256i x  = _mm256_maddubs_epi16(c0, cb);                               \
        x = _mm256_madd_epi16(x, c1);                                            \
        __m256i xs = _mm256_srli_epi64(x, 32 - 4*(D_VAL));                       \
        x = _mm256_or_si256(_mm256_and_si256(x, c3),                             \
                             _mm256_andnot_si256(c3, xs));                       \
        __m256i out_y = _mm256_shuffle_epi8(x, compact);                         \
        int base = (i * (D_VAL)) >> 3;                                           \
        _mm_storeu_si128((__m128i *)(out + base),                                \
                          _mm256_castsi256_si128(out_y));                        \
        _mm_storeu_si128((__m128i *)(out + base + 2*(D_VAL)),                    \
                          _mm256_extracti128_si256(out_y, 1));                   \
    }                                                                            \
    return i;                                                                    \
}
PIVCO_U16PACK_AVX2_DN(u16pack_d2_avx2_x86, 2, PIVCO_PACK_AVX2_COMPACT_D2)
PIVCO_U16PACK_AVX2_DN(u16pack_d3_avx2_x86, 3, PIVCO_PACK_AVX2_COMPACT_D3)
PIVCO_U16PACK_AVX2_DN(u16pack_d5_avx2_x86, 5, PIVCO_PACK_AVX2_COMPACT_D5)
PIVCO_U16PACK_AVX2_DN(u16pack_d6_avx2_x86, 6, PIVCO_PACK_AVX2_COMPACT_D6)
PIVCO_U16PACK_AVX2_DN(u16pack_d7_avx2_x86, 7, PIVCO_PACK_AVX2_COMPACT_D7)
#undef PIVCO_U16PACK_AVX2_DN
#endif /* PIVCO_HAS_AVX2 */

/* ---------- Encode primitives (bitmap + partition) ----------
 *
 * Dense-codes mask build via the classic SSE movemask trick.
 * code_vec holds 8 left-aligned 16-bit Huffman codes.  At tree depth d,
 * bit d of the original code is at position (15 - d) of code_la; we
 * shift LEFT by d to move that bit to position 15 (= sign bit of each
 * int16 lane).  _mm_packs_epi16 with signed saturation then collapses
 * each int16 lane to an int8 byte where bit 7 is the sign bit, and
 * _mm_movemask_epi8 reads bit 7 of each byte into an 8-bit bitmask --
 * the per-element bit slice we want.
 *
 * Cost: vpsllw (1) + vpacksw (1) + vpmovmskb (1) = 3 SSE ops + 1 mask. */
static inline uint8_t enc_mask8_codes_la_x86(__m128i code_vec,
                                               __m128i shift_count)
{
    __m128i shifted = _mm_sll_epi16(code_vec, shift_count);
    __m128i bytes   = _mm_packs_epi16(shifted, _mm_setzero_si128());
    return (uint8_t)_mm_movemask_epi8(bytes);
}

/* Stride-16 SIMD main path: 2x-unrolled load + partition.  Two 8-code
 * chunks per outer iter with all per-chunk deps independent until the
 * cursor math.  The second store of each (right, left) pair overlaps
 * the first and overwrites its trailing pshufb junk with chunk 1's
 * valid prefix — safe because the junk lives past the popcount of
 * chunk 0.  Stride-8 residual handles n mod 16 ∈ [8, 16); scalar tail
 * handles the final 1..7.
 *
 * +9% on Zen 3 (c6a), +14% on Cascade Lake (c5) vs the previous
 * stride-8-only inner loop.  Wire format byte-for-byte identical. */
static inline int u16part_full_x86(uint16_t *codes_la, int n,
                                               int depth,
                                               uint8_t *bm,
                                               uint16_t *right_out)
{
    uint16_t *lp = codes_la, *rp = right_out;
    int j = 0;
    __m128i shift_count = _mm_cvtsi32_si128(depth);

    for (; j + 16 <= n; j += 16) {
        __m128i code0 = _mm_loadu_si128((const __m128i *)(codes_la + j    ));
        __m128i code1 = _mm_loadu_si128((const __m128i *)(codes_la + j + 8));

        uint8_t m0 = enc_mask8_codes_la_x86(code0, shift_count);
        uint8_t m1 = enc_mask8_codes_la_x86(code1, shift_count);
        bm[j >> 3]       = m0;
        bm[(j >> 3) + 1] = m1;

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

        _mm_storeu_si128((__m128i *)rp,                r0);
        _mm_storeu_si128((__m128i *)(rp + nr0),        r1);
        _mm_storeu_si128((__m128i *)lp,                l0);
        _mm_storeu_si128((__m128i *)(lp + (8 - nr0)),  l1);

        rp += nr0 + nr1;
        lp += (8 - nr0) + (8 - nr1);
    }

    /* Stride-8 residual (n mod 16 ∈ [8, 16)). */
    if (j + 8 <= n) {
        __m128i code_vec = _mm_loadu_si128((const __m128i *)(codes_la + j));
        uint8_t mask = enc_mask8_codes_la_x86(code_vec, shift_count);
        bm[j >> 3] = mask;

        const uint8_t *tab = compress_tab[mask];
        __m128i shuf_r = _mm_load_si128((const __m128i *)tab);
        __m128i shuf_l = _mm_load_si128((const __m128i *)(tab + 16));
        __m128i right  = _mm_shuffle_epi8(code_vec, shuf_r);
        __m128i left   = _mm_shuffle_epi8(code_vec, shuf_l);
        int nr = compress_popcnt[mask];
        _mm_storeu_si128((__m128i *)rp, right);
        _mm_storeu_si128((__m128i *)lp, left);
        rp += nr;
        lp += (8 - nr);
        j += 8;
    }

    /* Scalar tail.  Read all tail codes into a temporary before writing
     * back, since the in-place left write can overlap the read when
     * the left cursor + 8 > j. */
    if (j < n) {
        int tail = n - j;
        uint16_t tail_buf[8];
        for (int k = 0; k < tail; k++) tail_buf[k] = codes_la[j + k];
        uint8_t mask = 0;
        int shift_d = 15 - depth;
        for (int k = 0; k < tail; k++) {
            int bit = (tail_buf[k] >> shift_d) & 1;
            mask |= (uint8_t)(bit << k);
        }
        bm[j >> 3] = mask;
        for (int k = 0; k < tail; k++) {
            if (mask & (1 << k))
                *rp++ = tail_buf[k];
            else
                *lp++ = tail_buf[k];
        }
    }
    return (int)(rp - right_out);
}

/* u16part_core_x86 — shared partition loop for the right/left/none variants (and
 * the from-bitmap BUILD=0 form, kept for a future TD-decode share).  FULL stays
 * hand-written in u16part_full_x86 (matching the NEON rationale:
 * the generic 1,1,1 form can schedule worse on the hot common path).
 * always_inline + compile-time-constant flags => each wrapper specializes. */
__attribute__((always_inline)) static inline
int u16part_core_x86(uint16_t *codes_la, int n, int depth,
                  uint8_t *bm, const uint8_t *bm_in, uint16_t *right_out,
                  int BUILD, int EMIT_RIGHT, int EMIT_LEFT)
{
    int n_left = 0, n_right = 0, j = 0;
    __m128i shift_count = _mm_cvtsi32_si128(depth);
    for (; j + 8 <= n; j += 8) {
        __m128i code_vec = _mm_loadu_si128((const __m128i *)(codes_la + j));
        uint8_t mask;
        if (BUILD) { mask = enc_mask8_codes_la_x86(code_vec, shift_count); bm[j >> 3] = mask; }
        else         mask = bm_in[j >> 3];
        const uint8_t *tab = compress_tab[mask];
        if (EMIT_RIGHT)
            _mm_storeu_si128((__m128i *)(right_out + n_right),
                             _mm_shuffle_epi8(code_vec, _mm_load_si128((const __m128i *)tab)));
        if (EMIT_LEFT)
            _mm_storeu_si128((__m128i *)(codes_la + n_left),
                             _mm_shuffle_epi8(code_vec, _mm_load_si128((const __m128i *)(tab + 16))));
        int nr = compress_popcnt[mask];
        n_right += nr;
        n_left  += 8 - nr;
    }
    if (j < n) {
        int tail = n - j, shift_d = 15 - depth;
        uint16_t tail_buf[8];
        for (int k = 0; k < tail; k++) tail_buf[k] = codes_la[j + k];
        uint8_t mask;
        if (BUILD) {
            mask = 0;
            for (int k = 0; k < tail; k++)
                mask |= (uint8_t)(((tail_buf[k] >> shift_d) & 1) << k);
            bm[j >> 3] = mask;
        } else mask = bm_in[j >> 3];
        for (int k = 0; k < tail; k++) {
            if (mask & (1 << k)) { if (EMIT_RIGHT) right_out[n_right] = tail_buf[k]; n_right++; }
            else                 { if (EMIT_LEFT)  codes_la[n_left]   = tail_buf[k]; n_left++;  }
        }
    }
    return n_right;
}

/* ---------- Encode primitives (init) ---------- */

/* u16init_x86 — gather per-symbol left-aligned codes into codes_la.
 * Today this is a straight scalar loop; the compiler auto-vectorises
 * it well enough on AVX2 hosts that a hand-rolled vpermi2w / vpgatherq
 * version doesn't materially help (the LSU is the bottleneck either
 * way).  AVX-512 has an actual SIMD win via vpermi2w -- see
 * primitives_avx512.h. */
static inline void u16init_x86(uint16_t *codes_la, int n,
                                  const uint8_t *symbols,
                                  const uint16_t *code_la_lut)
{
    for (int i = 0; i < n; i++) codes_la[i] = code_la_lut[symbols[i]];
}

/* ---------- Encode primitives (flat-subtree pack) ----------
 *
 * Per-D SIMD bit-pack helpers.  Each reads D-bit codes from codes_la
 * (each lane holds the left-aligned Huffman code -- bit-d of the
 * original code is at position 15-d) and packs them LSB-first into
 * the output byte stream.  Each helper OVERPACKS (processes
 * ceil(n / stride) * stride elements).  The dispatcher u16pack_dN_x86
 * below handles the residual scalar tail. */

/* AVX2 D=2,3,5,6,7: 32 codes per ymm iter via ryg's multiply-as-shift
 * pack (pmaddubsw + pmaddwd + psrlq + and/andn/or + vpshufb compact
 * + 2x movdqu).  Helpers live in pivco_huffman_avx2_pack.h.  Beats the
 * prior sllv+reduce_add path by 3-3.5x on Zen 3 (c6a).  D=4 stays on
 * the SSE _mm_maddubs_epi16 "2 codes per byte" path which is intrinsically
 * cheap and beats v3 by ~20%. */

/* SSE4.1 D=2: 16 codes -> 4 bytes.  _mm_maddubs_epi16 weighted pair-add
 * with weights {1, 4, 16, 64} (int8 max 127, so 64 fits). */
static inline int u16pack_d2_sse_x86(uint8_t *out, const uint16_t *codes_la,
                                    int n, int right_shift)
{
    const __m128i weights = _mm_setr_epi8(1, 4, 16, 64, 1, 4, 16, 64,
                                           1, 4, 16, 64, 1, 4, 16, 64);
    int i = 0;
    for (; i + 16 <= n; i += 16) {
        __m128i v0 = _mm_loadu_si128((const __m128i *)(codes_la + i    ));
        __m128i v1 = _mm_loadu_si128((const __m128i *)(codes_la + i + 8));
        v0 = _mm_srli_epi16(v0, right_shift);
        v1 = _mm_srli_epi16(v1, right_shift);
        v0 = _mm_and_si128(v0, _mm_set1_epi16(0x3));
        v1 = _mm_and_si128(v1, _mm_set1_epi16(0x3));
        __m128i bytes = _mm_packus_epi16(v0, v1);
        __m128i step1 = _mm_maddubs_epi16(bytes, weights);
        __m128i step2 = _mm_hadd_epi16(step1, _mm_setzero_si128());
        __m128i out_bytes = _mm_packus_epi16(step2, _mm_setzero_si128());
        uint32_t packed4 = (uint32_t)_mm_cvtsi128_si32(out_bytes);
        memcpy(out + (i * 2 / 8), &packed4, 4);
    }
    return i;
}

/* SSE4.1 D=4: 16 codes -> 8 bytes.  _mm_maddubs_epi16 with weights {1, 16}. */
static inline int u16pack_d4_sse_x86(uint8_t *out, const uint16_t *codes_la,
                                    int n, int right_shift)
{
    const __m128i weights = _mm_setr_epi8(1, 16, 1, 16, 1, 16, 1, 16,
                                           1, 16, 1, 16, 1, 16, 1, 16);
    int i = 0;
    for (; i + 16 <= n; i += 16) {
        __m128i v0 = _mm_loadu_si128((const __m128i *)(codes_la + i    ));
        __m128i v1 = _mm_loadu_si128((const __m128i *)(codes_la + i + 8));
        v0 = _mm_srli_epi16(v0, right_shift);
        v1 = _mm_srli_epi16(v1, right_shift);
        v0 = _mm_and_si128(v0, _mm_set1_epi16(0xF));
        v1 = _mm_and_si128(v1, _mm_set1_epi16(0xF));
        __m128i bytes = _mm_packus_epi16(v0, v1);
        __m128i step1 = _mm_maddubs_epi16(bytes, weights);
        __m128i out_bytes = _mm_packus_epi16(step1, _mm_setzero_si128());
        _mm_storel_epi64((__m128i *)(out + (i * 4 / 8)), out_bytes);
    }
    return i;
}

/* SSE4.1 D=8: 16 codes -> 16 bytes, byte-aligned. */
static inline int u16pack_d8_sse_x86(uint8_t *out, const uint16_t *codes_la,
                                    int n, int right_shift)
{
    int i = 0;
    for (; i + 16 <= n; i += 16) {
        __m128i v0 = _mm_loadu_si128((const __m128i *)(codes_la + i    ));
        __m128i v1 = _mm_loadu_si128((const __m128i *)(codes_la + i + 8));
        /* Mask to 8 bits before the *saturating* packus: codes_la may carry
         * the flat-root prefix in the bits above the D=8 code (depth>0), and
         * packus would clamp those to 255 instead of dropping them (NEON's
         * vmovn truncates).  The mask matches the truncate semantics. */
        v0 = _mm_and_si128(_mm_srli_epi16(v0, right_shift), _mm_set1_epi16(0x00FF));
        v1 = _mm_and_si128(_mm_srli_epi16(v1, right_shift), _mm_set1_epi16(0x00FF));
        __m128i bytes = _mm_packus_epi16(v0, v1);
        _mm_storeu_si128((__m128i *)(out + i), bytes);
    }
    return i;
}

/* SSE4.1 D=3: 8 codes -> 24 bits via _mm_mullo_epi32 multiply-as-shift.
 * SSE4.1 lacks _mm_sllv_epi32; multiplying uint32 by 2^k achieves the
 * same per-lane left shift. */
static inline int u16pack_d3_sse_x86(uint8_t *out, const uint16_t *codes_la,
                                    int n, int right_shift)
{
    const __m128i mlo = _mm_setr_epi32(1, 8, 64, 512);
    const __m128i mhi = _mm_setr_epi32(4096, 32768, 262144, 2097152);
    int i = 0;
    for (; i + 8 <= n; i += 8) {
        __m128i v = _mm_loadu_si128((const __m128i *)(codes_la + i));
        v = _mm_srli_epi16(v, right_shift);
        v = _mm_and_si128(v, _mm_set1_epi16(0x7));
        __m128i vlo = _mm_unpacklo_epi16(v, _mm_setzero_si128());
        __m128i vhi = _mm_unpackhi_epi16(v, _mm_setzero_si128());
        vlo = _mm_mullo_epi32(vlo, mlo);
        vhi = _mm_mullo_epi32(vhi, mhi);
        __m128i s = _mm_add_epi32(vlo, vhi);
        s = _mm_hadd_epi32(s, s);
        s = _mm_hadd_epi32(s, s);
        uint32_t packed = (uint32_t)_mm_cvtsi128_si32(s);
        int bi = i * 3 / 8;
        out[bi    ] = (uint8_t)(packed       );
        out[bi + 1] = (uint8_t)(packed >>  8);
        out[bi + 2] = (uint8_t)(packed >> 16);
    }
    return i;
}

/* Dispatcher: pack n D-bit codes from codes_la into out[].  Selects the
 * SIMD per-D pack helper, then handles any residual scalar tail.
 *
 * D=2/4/8 use SSE4.1 directly; D=3/5/6/7 use AVX2 sllv where available,
 * falling back to scalar on SSE4.1-only hosts (D=3 has an SSE multiply-
 * as-shift version, D=5/6/7 stay scalar since SSE has no uint64 per-
 * lane shift). */
static inline void u16pack_dN_x86(uint8_t *out, const uint16_t *codes_la,
                                 int n, int D, int depth)
{
    int total_bytes = (n * D + 7) >> 3;
    if (total_bytes > 0) out[total_bytes - 1] = 0;
    int right_shift = 16 - depth - D;

    /* NB: BMI2 pext pack is NOT used here.  pext is microcoded-slow on AMD
     * pre-Zen4 (measured 2x worse than the AVX2 spread on c6a/Zen3), and the
     * x86 backend runs on those parts.  pext pack is gated to the AVX-512
     * backend instead (Intel Xeon + AMD Zen4, both fast-pext). */
    int i = 0;
    switch (D) {
    case 4: i = u16pack_d4_sse_x86(out, codes_la, n, right_shift); break;
    case 8: i = u16pack_d8_sse_x86(out, codes_la, n, right_shift); break;
#ifdef PIVCO_HAS_AVX2
    case 2: i = u16pack_d2_avx2_x86(out, codes_la, n, right_shift); break;
    case 3: i = u16pack_d3_avx2_x86(out, codes_la, n, right_shift); break;
    case 5: i = u16pack_d5_avx2_x86(out, codes_la, n, right_shift); break;
    case 6: i = u16pack_d6_avx2_x86(out, codes_la, n, right_shift); break;
    case 7: i = u16pack_d7_avx2_x86(out, codes_la, n, right_shift); break;
#else
    case 2: i = u16pack_d2_sse_x86(out, codes_la, n, right_shift); break;
    case 3: i = u16pack_d3_sse_x86(out, codes_la, n, right_shift); break;
    /* D=5,6,7 fall through to scalar tail on SSE4.1-only hosts. */
#endif
    default: break;
    }

    int simd_n = i > n ? n : i;
    PROF_COUNT_ONLY(PROF_ENC_FLAT_SIMD_ELEMS, simd_n);
    PROF_COUNT_ONLY(PROF_ENC_FLAT_TAIL_ELEMS, n - simd_n);
    (void)simd_n;  /* unused when PIVCO_PROF=0 (PROF_COUNT_ONLY expands away) */

    if (i >= n) return;

    /* Scalar tail. */
    uint32_t mask = (1u << D) - 1;
    int bit_pos = i * D;
    int byte_idx = bit_pos >> 3;
    int bits_in_buf = bit_pos & 7;
    uint64_t buf = bits_in_buf > 0
        ? (uint64_t)out[byte_idx] & ((1u << bits_in_buf) - 1)
        : 0;
    for (; i < n; i++) {
        uint32_t local = ((uint32_t)codes_la[i] >> right_shift) & mask;
        buf |= ((uint64_t)local) << bits_in_buf;
        bits_in_buf += D;
        while (bits_in_buf >= 8) {
            out[byte_idx++] = (uint8_t)(buf & 0xff);
            buf >>= 8;
            bits_in_buf -= 8;
        }
    }
    if (bits_in_buf > 0) {
        out[byte_idx] = (uint8_t)(buf & ((1u << bits_in_buf) - 1));
    }
}


PIVCO_PRIM_ALWAYS_INLINE void u16enc_init(uint16_t *codes_la, int n,
                                              const uint8_t *symbols,
                                              const uint16_t *code_la_lut)
{ u16init_x86(codes_la, n, symbols, code_la_lut); }

PIVCO_PRIM_ALWAYS_INLINE int u16enc_partition_full(uint16_t *codes_la,
                                                      int n, int depth,
                                                      uint8_t *bm,
                                                      uint16_t *right_out)
{ return u16part_full_x86(codes_la, n, depth, bm, right_out); }

PIVCO_PRIM_ALWAYS_INLINE int u16enc_partition_right(uint16_t *codes_la,
                                                      int n, int depth,
                                                      uint8_t *bm,
                                                      uint16_t *right_out)
{ return u16part_core_x86(codes_la, n, depth, bm, NULL, right_out, 1, 1, 0); }

PIVCO_PRIM_ALWAYS_INLINE int u16enc_partition_left(uint16_t *codes_la,
                                                     int n, int depth,
                                                     uint8_t *bm)
{ return u16part_core_x86(codes_la, n, depth, bm, NULL, NULL, 1, 0, 1); }

PIVCO_PRIM_ALWAYS_INLINE int u16enc_partition_none(uint16_t *codes_la,
                                                     int n, int depth,
                                                     uint8_t *bm)
{ return u16part_core_x86(codes_la, n, depth, bm, NULL, NULL, 1, 0, 0); }

PIVCO_PRIM_ALWAYS_INLINE void u16enc_pack_dN(const uint16_t *codes_la,
                                             int n, int D, int depth,
                                             uint8_t *out_packed)
{ u16pack_dN_x86(out_packed, codes_la, n, D, depth); }
#endif /* __SSE4_1__ && !__AVX512VBMI2__ */


#if defined(__AVX512VBMI2__)

/* Retired u16 AVX-512 pack (moved out of pivco_huffman_avx512_pack.h; uses the
 * shared pivco_pack_compact_d{5,6,7} tables still defined there). */
/* Load 64 left-aligned u16 codes, right-shift to align code into low D
 * bits, narrow u16 -> u8, assemble into one zmm (1 code per byte).
 *
 * NB: the returned bytes carry whatever sat in the low byte of each
 * codes_la lane after right-shift — high bits above D may be GARBAGE.
 * Callers must mask to D bits if their subsequent pipeline would leak
 * those bits.  The multishift-based D=3/5/6/7 helpers don't mask here
 * because their per-group `_mm512_and_si512(cb, mX)` already clips both
 * the unwanted bytes (zero outside the group) AND the high bits within
 * each group byte (0x07 / 0x1F / 0x3F / 0x7F).  The vpermb-stride D=2
 * and D=4 helpers DO mask because their shift-then-OR step would
 * otherwise leak high bits across byte boundaries within each u32. */
static inline __m512i pivco_u16pack_load(const uint16_t *codes_la,
                                                  int right_shift)
{
    __m512i lo16 = _mm512_loadu_si512((const __m512i *)(codes_la));
    __m512i hi16 = _mm512_loadu_si512((const __m512i *)(codes_la + 32));
    __m256i lo_b = _mm512_cvtepi16_epi8(_mm512_srli_epi16(lo16, right_shift));
    __m256i hi_b = _mm512_cvtepi16_epi8(_mm512_srli_epi16(hi16, right_shift));
    return _mm512_inserti64x4(_mm512_castsi256_si512(lo_b), hi_b, 1);
}

/* D=2 (4 codes per byte): 4 groups by code mod 4, gather + shift + OR. */
static inline int u16pack_d2_avx512(uint8_t *out, const uint16_t *codes_la,
                                   int n, int right_shift)
{
    /* Group g (g in 0..3) gathers codes (g, g+4, g+8, ..., g+60) into the
     * low 16 output bytes; group g's bits land at position 2g within each
     * output byte. */
    const __m512i shuf0 = _mm512_set_epi8(
        0,0,0,0,0,0,0,0, 0,0,0,0,0,0,0,0, 0,0,0,0,0,0,0,0, 0,0,0,0,0,0,0,0,
        0,0,0,0,0,0,0,0, 0,0,0,0,0,0,0,0,
        60,56,52,48,44,40,36,32, 28,24,20,16,12,8,4,0);
    const __m512i shuf1 = _mm512_set_epi8(
        0,0,0,0,0,0,0,0, 0,0,0,0,0,0,0,0, 0,0,0,0,0,0,0,0, 0,0,0,0,0,0,0,0,
        0,0,0,0,0,0,0,0, 0,0,0,0,0,0,0,0,
        61,57,53,49,45,41,37,33, 29,25,21,17,13,9,5,1);
    const __m512i shuf2 = _mm512_set_epi8(
        0,0,0,0,0,0,0,0, 0,0,0,0,0,0,0,0, 0,0,0,0,0,0,0,0, 0,0,0,0,0,0,0,0,
        0,0,0,0,0,0,0,0, 0,0,0,0,0,0,0,0,
        62,58,54,50,46,42,38,34, 30,26,22,18,14,10,6,2);
    const __m512i shuf3 = _mm512_set_epi8(
        0,0,0,0,0,0,0,0, 0,0,0,0,0,0,0,0, 0,0,0,0,0,0,0,0, 0,0,0,0,0,0,0,0,
        0,0,0,0,0,0,0,0, 0,0,0,0,0,0,0,0,
        63,59,55,51,47,43,39,35, 31,27,23,19,15,11,7,3);
    int i = 0;
    for (; i + 64 <= n; i += 64) {
        /* D=2 needs the explicit mask: the slli-then-OR below would otherwise
         * leak high bits across byte boundaries within each u32 lane. */
        __m512i cb = _mm512_and_si512(
            pivco_u16pack_load(codes_la + i, right_shift),
            _mm512_set1_epi8(0x03));
        __m512i g0 = _mm512_permutexvar_epi8(shuf0, cb);
        __m512i g1 = _mm512_permutexvar_epi8(shuf1, cb);
        __m512i g2 = _mm512_permutexvar_epi8(shuf2, cb);
        __m512i g3 = _mm512_permutexvar_epi8(shuf3, cb);
        __m512i packed = _mm512_or_si512(
            _mm512_or_si512(g0, _mm512_slli_epi32(g1, 2)),
            _mm512_or_si512(_mm512_slli_epi32(g2, 4), _mm512_slli_epi32(g3, 6)));
        _mm512_mask_storeu_epi8(out + ((i * 2) >> 3),
                                 (__mmask64)0xFFFFULL, packed);
    }
    return i;
}

/* D=3: 4 groups (codes mod 4).  Each chunk of 8 codes -> 3 output bytes. */
static inline int u16pack_d3_avx512(uint8_t *out, const uint16_t *codes_la,
                                   int n, int right_shift)
{
    const __m512i mA = _mm512_set1_epi64((int64_t)0x0000000700000007ULL); /* bytes 0,4 */
    const __m512i mB = _mm512_set1_epi64((int64_t)0x0000070000000700ULL); /* bytes 1,5 */
    const __m512i mC = _mm512_set1_epi64((int64_t)0x0007000000070000ULL); /* bytes 2,6 */
    const __m512i mD = _mm512_set1_epi64((int64_t)0x0700000007000000ULL); /* bytes 3,7 */
    /* Per-byte multishift ctrls (lo->hi byte order).  Byte 2 of cA reads
     * a zero region of lane_A (Group A doesn't contribute to output byte
     * 2; pulling from a masked-zero byte avoids leaking code 0 in). */
    const __m512i cA = _mm512_set1_epi64((int64_t)0x0000000000081C00ULL); /* {0,28,8,...} */
    const __m512i cB = _mm512_set1_epi64((int64_t)0x0000000000292105ULL); /* {5,33,41,...} */
    const __m512i cC = _mm512_set1_epi64((int64_t)0x00000000002E120AULL); /* {10,18,46,...} */
    const __m512i cD = _mm512_set1_epi64((int64_t)0x0000000000331700ULL); /* {0,23,51,...} */
    int i = 0;
    for (; i + 64 <= n; i += 64) {
        __m512i cb = pivco_u16pack_load(codes_la + i, right_shift);
        __m512i a = _mm512_multishift_epi64_epi8(cA, _mm512_and_si512(cb, mA));
        __m512i b = _mm512_multishift_epi64_epi8(cB, _mm512_and_si512(cb, mB));
        __m512i c = _mm512_multishift_epi64_epi8(cC, _mm512_and_si512(cb, mC));
        __m512i d = _mm512_multishift_epi64_epi8(cD, _mm512_and_si512(cb, mD));
        __m512i packed = _mm512_or_si512(_mm512_or_si512(a, b),
                                          _mm512_or_si512(c, d));
        __m512i compact = _mm512_permutexvar_epi8(
            _mm512_load_si512((const __m512i *)pivco_pack_compact_d3), packed);
        _mm512_mask_storeu_epi8(out + ((i * 3) >> 3),
                                 (__mmask64)0xFFFFFFULL, compact);
    }
    return i;
}

/* D=4 (2 codes per byte): 2 groups (even/odd), gather + shift + OR. */
static inline int u16pack_d4_avx512(uint8_t *out, const uint16_t *codes_la,
                                   int n, int right_shift)
{
    const __m512i shuf0 = _mm512_set_epi8(
        0,0,0,0,0,0,0,0, 0,0,0,0,0,0,0,0, 0,0,0,0,0,0,0,0, 0,0,0,0,0,0,0,0,
        62,60,58,56,54,52,50,48, 46,44,42,40,38,36,34,32,
        30,28,26,24,22,20,18,16, 14,12,10, 8, 6, 4, 2, 0);
    const __m512i shuf1 = _mm512_set_epi8(
        0,0,0,0,0,0,0,0, 0,0,0,0,0,0,0,0, 0,0,0,0,0,0,0,0, 0,0,0,0,0,0,0,0,
        63,61,59,57,55,53,51,49, 47,45,43,41,39,37,35,33,
        31,29,27,25,23,21,19,17, 15,13,11, 9, 7, 5, 3, 1);
    int i = 0;
    for (; i + 64 <= n; i += 64) {
        /* D=4 needs the explicit mask: see D=2 comment. */
        __m512i cb = _mm512_and_si512(
            pivco_u16pack_load(codes_la + i, right_shift),
            _mm512_set1_epi8(0x0F));
        __m512i g0 = _mm512_permutexvar_epi8(shuf0, cb);
        __m512i g1 = _mm512_permutexvar_epi8(shuf1, cb);
        __m512i packed = _mm512_or_si512(g0, _mm512_slli_epi32(g1, 4));
        _mm512_mask_storeu_epi8(out + ((i * 4) >> 3),
                                 (__mmask64)0xFFFFFFFFULL, packed);
    }
    return i;
}

/* D=5/6/7 pack via ryg multiply-as-shift (port of AVX2 a1aa6b9):
 *   - mask byte-laid codes to D bits
 *   - vpmaddubsw c0   word[i] = cb[2i]   + cb[2i+1]   * 2^D    (2D bits)
 *   - vpmaddwd   c1   dword[i] = word[2i] + word[2i+1] * 2^(2D) (4D bits)
 *   - vpsrlq + vpternlogq 0xE4 to merge dword[2i+1] into dword[2i]'s
 *     u64 lane: bits [0..4D-1] = dword[2i], bits [4D..8D-1] = dword[2i+1]
 *   - vpermb compact + masked store
 *
 * Beats the per-group multishift path on Intel (Granite Rapids -16 to
 * -23% cyc/elem, Sapphire Rapids -10 to -18%); ties or marginally loses
 * on AMD (Zen 4 D=5 -24% else tied, Zen 5 D=5 -10% / D=6,7 +5%).  See
 * scratch bench results in the commit message.
 *
 * For D=5 only, the two-byte mults of vpmaddubsw can produce u16 lanes
 * up to 31 + 31*32 = 1023 (fits u16), then vpmaddwd up to 1023 + 1023 *
 * 1024 ≈ 1.05M (fits u32) -- 4D = 20 bits is the maximum used.  Same
 * envelope analysis for D=6 (24 bits) and D=7 (28 bits). */
#define PIVCO_U16PACK_AVX512_RYG_DN(NAME, D_VAL, COMPACT_TAB, STORE_MASK)            \
static inline int NAME(uint8_t *out, const uint16_t *codes_la,                  \
                       int n, int right_shift)                                    \
{                                                                                 \
    const __m512i c0 = _mm512_set1_epi16(                                         \
        (int16_t)(((1 << (D_VAL)) << 8) | 1));                                    \
    const __m512i c1 = _mm512_set1_epi32(                                         \
        (int32_t)(((int32_t)1 << (2*(D_VAL))) << 16) | 1);                        \
    const __m512i c3 = _mm512_set1_epi64(                                         \
        (int64_t)(((int64_t)1 << (4*(D_VAL))) - 1));                              \
    const __m512i d_mask = _mm512_set1_epi8((char)((1 << (D_VAL)) - 1));          \
    int i = 0;                                                                    \
    for (; i + 64 <= n; i += 64) {                                                \
        __m512i cb = pivco_u16pack_load(codes_la + i, right_shift);       \
        cb = _mm512_and_si512(cb, d_mask);                                        \
        __m512i x  = _mm512_maddubs_epi16(c0, cb);                                \
        x = _mm512_madd_epi16(x, c1);                                             \
        __m512i xs = _mm512_srli_epi64(x, 32 - 4*(D_VAL));                        \
        /* (x & c3) | (xs & ~c3)  via vpternlogq 0xE4 */                          \
        x = _mm512_ternarylogic_epi64(x, xs, c3, 0xE4);                           \
        __m512i compact = _mm512_permutexvar_epi8(                                \
            _mm512_load_si512((const __m512i *)COMPACT_TAB), x);                  \
        _mm512_mask_storeu_epi8(out + ((i * (D_VAL)) >> 3),                       \
                                 (__mmask64)(STORE_MASK), compact);               \
    }                                                                             \
    return i;                                                                     \
}
PIVCO_U16PACK_AVX512_RYG_DN(u16pack_d5_avx512, 5, pivco_pack_compact_d5, 0xFFFFFFFFFFULL)
PIVCO_U16PACK_AVX512_RYG_DN(u16pack_d6_avx512, 6, pivco_pack_compact_d6, 0xFFFFFFFFFFFFULL)
PIVCO_U16PACK_AVX512_RYG_DN(u16pack_d7_avx512, 7, pivco_pack_compact_d7, 0x00FFFFFFFFFFFFFFULL)
#undef PIVCO_U16PACK_AVX512_RYG_DN
/* ---------- Encode primitives (bitmap + partition) ----------
 *
 * Stride-32 main loop: load 32 left-aligned codes, build the 32-bit
 * mask via vpsllw + vpmovw2m (the AVX-512 analog of SSE's vpsllw +
 * vpacksw + vpmovmskb sequence -- single instruction).  Partition with
 * vpcompressw, two stores.  SSE-stride-8 tail uses _mm_maskz_compress_
 * epi16 (VL).  No shuffle table needed at either tier.
 *
 * codes_la is depth-threaded: not shifted across recursion levels.
 * The current-depth partition bit is at position (15 - depth) of each
 * lane; we left-shift by `depth` to move it to bit 15 (sign bit of
 * int16) so vpmovw2m reads it directly. */

static inline uint32_t enc_mask32_codes_la_avx512(__m512i code_vec, int depth)
{
    __m512i shifted = _mm512_slli_epi16(code_vec, depth);
    return (uint32_t)_mm512_movepi16_mask(shifted);
}

static inline int u16part_full_avx512(uint16_t *codes_la, int n,
                                                  int depth,
                                                  uint8_t *bm,
                                                  uint16_t *right_out)
{
    int n_left = 0, n_right = 0;
    int j = 0;

    for (; j + 32 <= n; j += 32) {
        __m512i code_vec = _mm512_loadu_si512((const __m512i *)(codes_la + j));
        uint32_t mask = enc_mask32_codes_la_avx512(code_vec, depth);
        memcpy(bm + (j >> 3), &mask, 4);

        __m512i right_v = _mm512_maskz_compress_epi16((__mmask32) mask, code_vec);
        __m512i left_v  = _mm512_maskz_compress_epi16((__mmask32)~mask, code_vec);
        _mm512_storeu_si512((__m512i *)(right_out      + n_right), right_v);
        _mm512_storeu_si512((__m512i *)(codes_la + n_left ), left_v);
        int nr = __builtin_popcount(mask);
        n_right += nr;
        n_left  += (32 - nr);
    }
    /* SSE-stride-8 remainder via _mm_maskz_compress_epi16 (VL). */
    __m128i shift_count = _mm_cvtsi32_si128(depth);
    for (; j + 8 <= n; j += 8) {
        __m128i code_vec = _mm_loadu_si128((const __m128i *)(codes_la + j));
        __m128i shifted  = _mm_sll_epi16(code_vec, shift_count);
        __m128i bytes    = _mm_packs_epi16(shifted, _mm_setzero_si128());
        uint8_t mask     = (uint8_t)_mm_movemask_epi8(bytes);
        bm[j >> 3] = mask;

        __m128i right_v = _mm_maskz_compress_epi16((__mmask8) mask, code_vec);
        __m128i left_v  = _mm_maskz_compress_epi16((__mmask8)~mask, code_vec);
        _mm_storeu_si128((__m128i *)(right_out      + n_right), right_v);
        _mm_storeu_si128((__m128i *)(codes_la + n_left ), left_v);
        int nr = __builtin_popcount(mask);
        n_right += nr;
        n_left  += (8 - nr);
    }
    /* Scalar tail. */
    if (j < n) {
        int tail = n - j;
        uint16_t tail_buf[8];
        for (int k = 0; k < tail; k++) tail_buf[k] = codes_la[j + k];
        uint8_t mask = 0;
        int shift_d = 15 - depth;
        for (int k = 0; k < tail; k++) {
            int bit = (tail_buf[k] >> shift_d) & 1;
            mask |= (uint8_t)(bit << k);
        }
        bm[j >> 3] = mask;
        for (int k = 0; k < tail; k++) {
            if (mask & (1 << k))
                right_out[n_right++] = tail_buf[k];
            else
                codes_la[n_left++] = tail_buf[k];
        }
    }
    return n_right;
}

/* u16part_core_avx512 — shared partition loop for the right/left/none variants
 * (and the from-bitmap BUILD=0 form for a future TD-decode share).  FULL stays
 * hand-written in u16part_full_avx512 (same rationale as NEON/x86).
 * 32-wide vpcompressw main path + 8-wide VL remainder + scalar tail; stores
 * gated by compile-time EMIT flags. */
__attribute__((always_inline)) static inline
int u16part_core_avx512(uint16_t *codes_la, int n, int depth,
                     uint8_t *bm, const uint8_t *bm_in, uint16_t *right_out,
                     int BUILD, int EMIT_RIGHT, int EMIT_LEFT)
{
    int n_left = 0, n_right = 0, j = 0;
    for (; j + 32 <= n; j += 32) {
        __m512i code_vec = _mm512_loadu_si512((const __m512i *)(codes_la + j));
        uint32_t mask;
        if (BUILD) { mask = enc_mask32_codes_la_avx512(code_vec, depth); memcpy(bm + (j >> 3), &mask, 4); }
        else         memcpy(&mask, bm_in + (j >> 3), 4);
        if (EMIT_RIGHT) _mm512_storeu_si512((__m512i *)(right_out + n_right),
                            _mm512_maskz_compress_epi16((__mmask32) mask, code_vec));
        if (EMIT_LEFT)  _mm512_storeu_si512((__m512i *)(codes_la + n_left),
                            _mm512_maskz_compress_epi16((__mmask32)~mask, code_vec));
        int nr = __builtin_popcount(mask);
        n_right += nr;
        n_left  += 32 - nr;
    }
    __m128i shift_count = _mm_cvtsi32_si128(depth);
    for (; j + 8 <= n; j += 8) {
        __m128i code_vec = _mm_loadu_si128((const __m128i *)(codes_la + j));
        uint8_t mask;
        if (BUILD) {
            __m128i shifted = _mm_sll_epi16(code_vec, shift_count);
            __m128i bytes   = _mm_packs_epi16(shifted, _mm_setzero_si128());
            mask = (uint8_t)_mm_movemask_epi8(bytes);
            bm[j >> 3] = mask;
        } else mask = bm_in[j >> 3];
        if (EMIT_RIGHT) _mm_storeu_si128((__m128i *)(right_out + n_right),
                            _mm_maskz_compress_epi16((__mmask8) mask, code_vec));
        if (EMIT_LEFT)  _mm_storeu_si128((__m128i *)(codes_la + n_left),
                            _mm_maskz_compress_epi16((__mmask8)~mask, code_vec));
        int nr = __builtin_popcount(mask);
        n_right += nr;
        n_left  += 8 - nr;
    }
    if (j < n) {
        int tail = n - j, shift_d = 15 - depth;
        uint16_t tail_buf[8];
        for (int k = 0; k < tail; k++) tail_buf[k] = codes_la[j + k];
        uint8_t mask;
        if (BUILD) {
            mask = 0;
            for (int k = 0; k < tail; k++)
                mask |= (uint8_t)(((tail_buf[k] >> shift_d) & 1) << k);
            bm[j >> 3] = mask;
        } else mask = bm_in[j >> 3];
        for (int k = 0; k < tail; k++) {
            if (mask & (1 << k)) { if (EMIT_RIGHT) right_out[n_right] = tail_buf[k]; n_right++; }
            else                 { if (EMIT_LEFT)  codes_la[n_left]   = tail_buf[k]; n_left++;  }
        }
    }
    return n_right;
}

/* ---------- Encode primitives (init) ----------
 *
 * u16init_avx512 — gather per-symbol left-aligned codes via byte-split
 * vpermex2var_epi8 (AVX-512 VBMI).  64 chars per iter, 12 ops, ~0.19
 * ops/char.  See the comment block for the lookup geometry.
 *
 * Lifted verbatim (minus the surrounding encoder driver) from the
 * legacy pivco_huffman_avx512.c.  All design notes preserved. */
static inline void u16init_avx512(uint16_t *codes_la, int n,
                                     const uint8_t *symbols,
                                     const uint16_t *code_la_lut)
{
    /* Build lo/hi byte tables from the uint16 code_la table.  Each
     * byte-table chunk holds 64 sequential entries' lo (or hi) bytes;
     * two chunks per byte-half pair the chars [0,128) and [128,256)
     * regions for vpermex2var_epi8. */
    __m512i u0 = _mm512_loadu_si512((const __m512i *)&code_la_lut[  0]);
    __m512i u1 = _mm512_loadu_si512((const __m512i *)&code_la_lut[ 32]);
    __m512i u2 = _mm512_loadu_si512((const __m512i *)&code_la_lut[ 64]);
    __m512i u3 = _mm512_loadu_si512((const __m512i *)&code_la_lut[ 96]);
    __m512i u4 = _mm512_loadu_si512((const __m512i *)&code_la_lut[128]);
    __m512i u5 = _mm512_loadu_si512((const __m512i *)&code_la_lut[160]);
    __m512i u6 = _mm512_loadu_si512((const __m512i *)&code_la_lut[192]);
    __m512i u7 = _mm512_loadu_si512((const __m512i *)&code_la_lut[224]);

    __m512i lo_c0p1 = _mm512_inserti64x4(
        _mm512_castsi256_si512(_mm512_cvtepi16_epi8(u0)),
        _mm512_cvtepi16_epi8(u1), 1);
    __m512i lo_c0p2 = _mm512_inserti64x4(
        _mm512_castsi256_si512(_mm512_cvtepi16_epi8(u2)),
        _mm512_cvtepi16_epi8(u3), 1);
    __m512i lo_c1p1 = _mm512_inserti64x4(
        _mm512_castsi256_si512(_mm512_cvtepi16_epi8(u4)),
        _mm512_cvtepi16_epi8(u5), 1);
    __m512i lo_c1p2 = _mm512_inserti64x4(
        _mm512_castsi256_si512(_mm512_cvtepi16_epi8(u6)),
        _mm512_cvtepi16_epi8(u7), 1);

    __m512i hi_c0p1 = _mm512_inserti64x4(
        _mm512_castsi256_si512(_mm512_cvtepi16_epi8(_mm512_srli_epi16(u0, 8))),
        _mm512_cvtepi16_epi8(_mm512_srli_epi16(u1, 8)), 1);
    __m512i hi_c0p2 = _mm512_inserti64x4(
        _mm512_castsi256_si512(_mm512_cvtepi16_epi8(_mm512_srli_epi16(u2, 8))),
        _mm512_cvtepi16_epi8(_mm512_srli_epi16(u3, 8)), 1);
    __m512i hi_c1p1 = _mm512_inserti64x4(
        _mm512_castsi256_si512(_mm512_cvtepi16_epi8(_mm512_srli_epi16(u4, 8))),
        _mm512_cvtepi16_epi8(_mm512_srli_epi16(u5, 8)), 1);
    __m512i hi_c1p2 = _mm512_inserti64x4(
        _mm512_castsi256_si512(_mm512_cvtepi16_epi8(_mm512_srli_epi16(u6, 8))),
        _mm512_cvtepi16_epi8(_mm512_srli_epi16(u7, 8)), 1);

    static const uint8_t inter_sel0_tab[64] __attribute__((aligned(64))) = {
         0, 64,  1, 65,  2, 66,  3, 67,  4, 68,  5, 69,  6, 70,  7, 71,
         8, 72,  9, 73, 10, 74, 11, 75, 12, 76, 13, 77, 14, 78, 15, 79,
        16, 80, 17, 81, 18, 82, 19, 83, 20, 84, 21, 85, 22, 86, 23, 87,
        24, 88, 25, 89, 26, 90, 27, 91, 28, 92, 29, 93, 30, 94, 31, 95
    };
    static const uint8_t inter_sel1_tab[64] __attribute__((aligned(64))) = {
        32, 96, 33, 97, 34, 98, 35, 99, 36,100, 37,101, 38,102, 39,103,
        40,104, 41,105, 42,106, 43,107, 44,108, 45,109, 46,110, 47,111,
        48,112, 49,113, 50,114, 51,115, 52,116, 53,117, 54,118, 55,119,
        56,120, 57,121, 58,122, 59,123, 60,124, 61,125, 62,126, 63,127
    };
    __m512i sel0 = _mm512_load_si512((const __m512i *)inter_sel0_tab);
    __m512i sel1 = _mm512_load_si512((const __m512i *)inter_sel1_tab);

    PROF_TIC();
    int i = 0;
    for (; i + 64 <= n; i += 64) {
        __m512i chars = _mm512_loadu_si512((const __m512i *)(symbols + i));
        __mmask64 hi_chunk = _mm512_movepi8_mask(chars);

        __m512i lo0 = _mm512_permutex2var_epi8(lo_c0p1, chars, lo_c0p2);
        __m512i lo1 = _mm512_permutex2var_epi8(lo_c1p1, chars, lo_c1p2);
        __m512i lo  = _mm512_mask_blend_epi8(hi_chunk, lo0, lo1);

        __m512i hi0 = _mm512_permutex2var_epi8(hi_c0p1, chars, hi_c0p2);
        __m512i hi1 = _mm512_permutex2var_epi8(hi_c1p1, chars, hi_c1p2);
        __m512i hi  = _mm512_mask_blend_epi8(hi_chunk, hi0, hi1);

        __m512i out0 = _mm512_permutex2var_epi8(lo, sel0, hi);
        __m512i out1 = _mm512_permutex2var_epi8(lo, sel1, hi);

        _mm512_storeu_si512((__m512i *)(codes_la + i     ), out0);
        _mm512_storeu_si512((__m512i *)(codes_la + i + 32), out1);
    }
    /* Scalar tail (PIVCO_BLOCK_SIZE is a multiple of 64 on AVX-512
     * hosts so this is currently dead, but kept defensively). */
    for (; i < n; i++) codes_la[i] = code_la_lut[symbols[i]];
    PROF_TOC(PROF_ENC_INIT, n);
}

/* ---------- Encode primitives (flat-subtree pack) ----------
 *
 * D=2..7: 64 codes per zmm iter via byte-laid intermediate +
 * vpmultishiftqb (D=3,5,6,7) or vpermb-stride gather + shift (D=2,4).
 * See pivco_huffman_avx512_pack.h for the per-D helpers.
 * D=8: byte-aligned, 32 codes per iter via vpmovwb narrow + store. */

static inline int u16pack_d8_avx512(uint8_t *out, const uint16_t *codes_la,
                                   int n, int right_shift)
{
    int i = 0;
    for (; i + 32 <= n; i += 32) {
        __m512i v = _mm512_loadu_si512((const __m512i *)(codes_la + i));
        v = _mm512_srli_epi16(v, right_shift);
        __m256i bytes = _mm512_cvtepi16_epi8(v);
        _mm256_storeu_si256((__m256i *)(out + i), bytes);
    }
    return i;
}

/* Dispatcher: pack n D-bit codes from codes_la into out[].  Selects
 * the SIMD per-D pack helper; scalar tail picks up the residual. */
static inline void u16pack_dN_avx512(uint8_t *out, const uint16_t *codes_la,
                                    int n, int D, int depth)
{
    int total_bytes = (n * D + 7) >> 3;
    if (total_bytes > 0) out[total_bytes - 1] = 0;
    int right_shift = 16 - depth - D;

    int i = 0;
    switch (D) {
    case 2: i = u16pack_d2_avx512(out, codes_la, n, right_shift); break;
    case 3: i = u16pack_d3_avx512(out, codes_la, n, right_shift); break;
    case 4: i = u16pack_d4_avx512(out, codes_la, n, right_shift); break;
    case 5: i = u16pack_d5_avx512(out, codes_la, n, right_shift); break;
    case 6: i = u16pack_d6_avx512(out, codes_la, n, right_shift); break;
    case 7: i = u16pack_d7_avx512(out, codes_la, n, right_shift); break;
    case 8: i = u16pack_d8_avx512(out, codes_la, n, right_shift); break;
    default: break;
    }

    int simd_n = i > n ? n : i;
    PROF_COUNT_ONLY(PROF_ENC_FLAT_SIMD_ELEMS, simd_n);
    PROF_COUNT_ONLY(PROF_ENC_FLAT_TAIL_ELEMS, n - simd_n);
    (void)simd_n;  /* unused when PIVCO_PROF=0 */

    if (i >= n) return;

    /* Scalar tail (fires only for D >= 9, currently impossible with
     * PIVCO_MAX_CODE_LEN = 11 and flat-D <= depth bound). */
    uint32_t mask = (1u << D) - 1;
    int bit_pos = i * D;
    int byte_idx = bit_pos >> 3;
    int bits_in_buf = bit_pos & 7;
    uint64_t buf = bits_in_buf > 0
        ? (uint64_t)out[byte_idx] & ((1u << bits_in_buf) - 1)
        : 0;
    for (; i < n; i++) {
        uint32_t local = ((uint32_t)codes_la[i] >> right_shift) & mask;
        buf |= ((uint64_t)local) << bits_in_buf;
        bits_in_buf += D;
        while (bits_in_buf >= 8) {
            out[byte_idx++] = (uint8_t)(buf & 0xff);
            buf >>= 8;
            bits_in_buf -= 8;
        }
    }
    if (bits_in_buf > 0) {
        out[byte_idx] = (uint8_t)(buf & ((1u << bits_in_buf) - 1));
    }
}


PIVCO_PRIM_ALWAYS_INLINE void u16enc_init(uint16_t *codes_la, int n,
                                              const uint8_t *symbols,
                                              const uint16_t *code_la_lut)
{ u16init_avx512(codes_la, n, symbols, code_la_lut); }

PIVCO_PRIM_ALWAYS_INLINE int u16enc_partition_full(uint16_t *codes_la,
                                                      int n, int depth,
                                                      uint8_t *bm,
                                                      uint16_t *right_out)
{ return u16part_full_avx512(codes_la, n, depth, bm, right_out); }

PIVCO_PRIM_ALWAYS_INLINE int u16enc_partition_right(uint16_t *codes_la,
                                                      int n, int depth,
                                                      uint8_t *bm,
                                                      uint16_t *right_out)
{ return u16part_core_avx512(codes_la, n, depth, bm, NULL, right_out, 1, 1, 0); }

PIVCO_PRIM_ALWAYS_INLINE int u16enc_partition_left(uint16_t *codes_la,
                                                     int n, int depth,
                                                     uint8_t *bm)
{ return u16part_core_avx512(codes_la, n, depth, bm, NULL, NULL, 1, 0, 1); }

PIVCO_PRIM_ALWAYS_INLINE int u16enc_partition_none(uint16_t *codes_la,
                                                     int n, int depth,
                                                     uint8_t *bm)
{ return u16part_core_avx512(codes_la, n, depth, bm, NULL, NULL, 1, 0, 0); }

PIVCO_PRIM_ALWAYS_INLINE void u16enc_pack_dN(const uint16_t *codes_la,
                                             int n, int D, int depth,
                                             uint8_t *out_packed)
{ u16pack_dN_avx512(out_packed, codes_la, n, D, depth); }
#endif /* __AVX512VBMI2__ */

#endif /* PIVCO_HUFFMAN_U16ENC_H */
