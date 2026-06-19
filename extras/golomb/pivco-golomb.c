/*
 * Reproduction of fgiesen's "Simple batch decoding of unary codes"
 *   https://fgiesen.wordpress.com/2026/05/30/simple-batch-decoding-of-unary-codes/
 *
 * Code from the post is marked verbatim where used; the encoder, refill loops,
 * harness, and types around it are reconstructed locally.
 *
 * Unary code convention used here (matches the post): value v -> v zero bits
 * then a single 1 bit, written LSB-first into a little-endian byte stream.
 */

#define _POSIX_C_SOURCE 200809L   /* expose CLOCK_MONOTONIC under -std=c11 (glibc) */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <time.h>
#include <math.h>
#include <assert.h>

typedef uint8_t  uint8;
typedef uint32_t uint32;
typedef uint64_t uint64;
typedef uint64_t U64;

/* ---------- helpers ---------- */

static inline uint64 read64LE(const uint8 *p) {
    uint64 v;
    memcpy(&v, p, sizeof v);  /* host is LE on M4/x86 */
    return v;
}
static inline void write64LE(uint8 *p, uint64 v) {
    memcpy(p, &v, sizeof v);
}
static inline int ctz64(uint64 x) { return __builtin_ctzll(x); }
static inline int popcnt32(uint32 x) { return __builtin_popcount(x); }

static void die(const char *msg) { fprintf(stderr, "error: %s\n", msg); exit(1); }
#define ERROR_RETURN(s) do { fprintf(stderr, "decode error: %s\n", s); return -1; } while (0)


/* ---------- encoder (not in post; reconstructed) ----------
 * Bit-packs unary codes LSB-first into a byte buffer.  Each value v writes v
 * zero bits then a 1 bit.  We pad the tail with enough zero bytes that the
 * decoders can safely do their 8-byte refill past the last real bit.
 */

static size_t encode_unary(const uint32 *values, size_t n,
                           uint8 *out, size_t out_cap) {
    uint64 buf = 0;
    int bits = 0;
    size_t opos = 0;
    for (size_t i = 0; i < n; i++) {
        uint32 v = values[i];
        /* v zero bits then a 1 bit: just the 1 bit shifted left by v */
        int len = (int)v + 1;
        if (bits + len > 63) {
            /* drain whole bytes */
            while (bits >= 8) {
                if (opos >= out_cap) die("out of buffer");
                out[opos++] = (uint8)(buf & 0xff);
                buf >>= 8;
                bits -= 8;
            }
        }
        if (len > 63) die("unary code too long for this naive encoder");
        buf |= (uint64)1 << (bits + v);
        bits += len;
    }
    /* flush whole bytes */
    while (bits > 0) {
        if (opos >= out_cap) die("out of buffer");
        out[opos++] = (uint8)(buf & 0xff);
        buf >>= 8;
        bits -= 8;
    }
    /* tail padding for safe 8-byte refill */
    for (int i = 0; i < 16; i++) {
        if (opos >= out_cap) die("out of buffer");
        out[opos++] = 0;
    }
    return opos;
}

/* ============================================================
 * Decoder 1 — naive serial (post §"Basic Serial Unary Decoder")
 * ============================================================
 *
 * Post snippet wrapped in a per-code loop.  The body of the loop (refill,
 * check, decode-one, consume) is verbatim from the post; the surrounding
 * `for` and the initial state are inferred.
 */
static int decode_serial(const uint8 *bitptr, uint8 *decoded, size_t n) {
    uint64 bitbuf = 0;
    int bitcount = 0;

    for (size_t i = 0; i < n; i++) {
        /* --- verbatim from post --- */
        // refill
        uint64 next = read64LE(bitptr);
        bitbuf |= next << bitcount;
        bitptr += (63 - bitcount) >> 3;
        bitcount |= 56;

        if ((bitbuf & ((1ull << 56) - 1)) == 0) {
            ERROR_RETURN("too many 0s in a row");
        }

        // unary code value = trailing zero count
        uint64 code = ctz64(bitbuf);
        decoded[i] = (uint8)code;

        // consume bits
        // code len=coded value + 1 bit for trailing 1
        uint64 len = code + 1;
        bitcount -= len;
        bitbuf >>= len;
        /* --- end verbatim --- */
    }
    return 0;
}

/* ============================================================
 * Decoder 2 — two codes per iteration (post §"Optimized Bit Detection")
 * ============================================================
 *
 * The branch using `bitbuf & (bitbuf - 1)` is the post's idea; the actual
 * two-code arithmetic and loop scaffolding are inferred from the prose.
 *
 *   If `bitbuf` has >= 2 set bits:
 *     code0  = ctz(bitbuf)
 *     code1  = ctz(bitbuf & (bitbuf-1)) - code0 - 1
 *     consumed bits = ctz(bitbuf & (bitbuf-1)) + 1
 */
static int decode_pair(const uint8 *bitptr, uint8 *decoded, size_t n) {
    uint64 bitbuf = 0;
    int bitcount = 0;
    size_t i = 0;

    while (i < n) {
        // refill (same as serial)
        uint64 next_word = read64LE(bitptr);
        bitbuf |= next_word << bitcount;
        bitptr += (63 - bitcount) >> 3;
        bitcount |= 56;

        if ((bitbuf & ((1ull << 56) - 1)) == 0) {
            ERROR_RETURN("too many 0s in a row");
        }

        /* --- verbatim shape from post + guard for long-code regime --- */
        // clear lowest set bit in bitbuf:
        uint64 next = bitbuf & (bitbuf - 1);
        uint64 c1_pos = next ? (uint64)ctz64(next) : 64;
        if (next != 0 && c1_pos < 56 && i + 1 < n) {
            // two codes fit safely in the 56-bit refill window
            uint64 c0_pos = ctz64(bitbuf);
            decoded[i++] = (uint8)c0_pos;
            decoded[i++] = (uint8)(c1_pos - c0_pos - 1);
            uint64 len = c1_pos + 1;
            bitcount -= len;
            bitbuf >>= len;
        } else {
            // single decode (fallback)
            uint64 code = ctz64(bitbuf);
            decoded[i++] = (uint8)code;
            uint64 len = code + 1;
            bitcount -= len;
            bitbuf >>= len;
        }
    }
    return 0;
}

/* ============================================================
 * Decoder 3 — Tunstall-style table, struct entries
 *             (post §"Tunstall-Style Table-Based Decoder")
 * ============================================================
 *
 * The post's snippet uses `TableEntry { value[8], count, carry }`.  The
 * loop body is verbatim; the table itself is built from the same generator
 * shown for the 64-bit packed variant, then unpacked into the struct form.
 */
typedef struct {
    uint8 value[8];
    uint8 count;
    uint8 carry;
} TableEntry;

static TableEntry g_unary_table_struct[256];
static uint64    g_unary_table[256];   /* used by decoders 3+4 and built by build_table() */

static void build_table(void) {
    /* --- verbatim from post --- */
    for (int byte = 0; byte < 256; byte++)
    {
        U64 remainder = byte | 256; // add a dummy 1 bit at top
        U64 shift = 0;
        U64 table_entry = 0;

        for (;;)
        {
            // determine next code value (we always have a set bit,
            // so this always works)
            U64 code = ctz64(remainder);

            // shift that bit out
            remainder >>= code + 1;

            if (remainder == 0)
            {
                // if there was only 1 set bit left, that's the sentinel
                // we inserted; our code is the carry.
                //
                // if and only if byte==255, we already have a real value
                // at that position, but in that case, both that final
                // value and the carry are 0, so it doesn't matter.
                table_entry |= code << 56;
                break;
            }
            else
            {
                // not yet done; add this value to the list
                table_entry |= code << shift;
                shift += 8;
            }
        }

        g_unary_table[byte] = table_entry;
    }
    /* --- end verbatim --- */

    /* unpack into struct form for decoder 3 */
    for (int b = 0; b < 256; b++) {
        uint64 t = g_unary_table[b];
        TableEntry e = {{0}, 0, 0};
        e.count = (uint8)__builtin_popcount((uint32)b);
        for (int j = 0; j < e.count; j++) {
            e.value[j] = (uint8)((t >> (8 * j)) & 0xff);
        }
        e.carry = (uint8)((t >> 56) & 0xff);
        g_unary_table_struct[b] = e;
    }
}

static int decode_tunstall(const uint8 *bitptr, uint8 *decoded, size_t n) {
    size_t i = 0;
    uint32 carry = 0;

    while (i < n) {
        /* --- verbatim shape from post --- */
        uint8 byte = *bitptr++;
        if (byte != 0) {
            // at least 1 bit set -> we emit values
            TableEntry tab = g_unary_table_struct[byte];

            // first value has carry added
            decoded[i] = (uint8)(carry + tab.value[0]);

            // remaining values are copied directly
            for (int j = 1; j < tab.count; j++) {
                decoded[i + j] = tab.value[j];
            }

            i += tab.count;
            carry = tab.carry;
        } else {
            // run of zeros: no code emitted just yet,
            // add 8 to current carry
            carry += 8;
            if (carry >= 57) {
                ERROR_RETURN("too many 0s in a row");
            }
        }
        /* --- end verbatim --- */
    }
    return 0;
}

/* ============================================================
 * Decoder 4 — Tunstall-style table, 64-bit packed entries, single store
 *             (post §"Optimized Tunstall Decoder")
 * ============================================================
 */
static int decode_tunstall64(const uint8 *bitptr, uint8 *decoded8, size_t n_bytes) {
    /*
     * NB: post stores 8-byte chunks of values directly with write64LE.  That
     * implies the decoded stream is 1 byte per value (uint8), so the codes
     * must fit in a byte (carry < 57 enforces this).  We use a uint8 output
     * buffer here and convert to uint32 in the harness.
     */
    size_t i = 0;
    uint64 carry = 0;

    while (i < n_bytes) {
        /* --- verbatim from post --- */
        uint8 byte = *bitptr++;
        if (byte != 0) {
            uint64 values = g_unary_table[byte];
            // emit values from table, adding carry into the first value (bottom byte)
            write64LE(decoded8 + i, values + carry);
            // next carry is in top byte of table
            carry = values >> 56;
            // number of values emitted is the population count
            i += popcnt32(byte);
        } else {
            carry += 8;
            if (carry >= 57) {
                ERROR_RETURN("too many 0s in a row");
            }
        }
        /* --- end verbatim --- */
    }
    return 0;
}

/* ============================================================
 * Decoder 5 — branch-free Tunstall64.  Same table as decoder 4 but the
 *             per-byte `if (byte != 0)` branch is replaced with an
 *             unconditional write + a ternary carry update that the
 *             compiler folds into a CMOV / CSEL.  Eliminates the branch
 *             mispredict tax that decoder 4 pays once the input has
 *             enough zero-bytes that the predictor can't lock onto a
 *             direction (see the per-p analysis we did earlier).
 *
 *             For byte == 0, `values` from the all-zero table entry is
 *             also 0, so the `values + carry` store harmlessly writes 8
 *             carry-padded values that are immediately overwritten by
 *             the next nonzero byte's emission (popcnt(0) == 0, so `i`
 *             doesn't advance).  The cost: ~1 wasted 8-byte store per
 *             zero byte — much cheaper than a mispredict.
 * ============================================================ */
static int decode_tunstall64_bf(const uint8 *bitptr, uint8 *decoded8, size_t n_bytes) {
    size_t i = 0;
    uint64 carry = 0;

    while (i < n_bytes) {
        uint8 byte = *bitptr++;
        uint64 values = g_unary_table[byte];
        write64LE(decoded8 + i, values + carry);
        i += popcnt32(byte);
        carry = (byte != 0) ? (values >> 56) : (carry + 8);
    }
    if (carry >= 57) ERROR_RETURN("too many 0s in a row");
    return 0;
}

/* ---------- harness ---------- */

/* ============================================================
 * "pivco" layout for unary codes
 * ============================================================
 *
 * Per-level bitmap layout (analogous to pivco-Huffman on a left-spine tree):
 *
 *   level k bitmap holds the k-th bit of every code whose first k bits were 0
 *   -> length is count(v_i >= k); bit value is 1 iff v_i == k
 *
 * Stored levels: 0 .. max_d - 1.  Level max_d would be all 1s (every code
 * still active at depth max_d terminates here by definition), so it's skipped;
 * its count is recovered as count[max_d-1] - popcount(level[max_d-1]).
 *
 * Decode is bottom-up, identical in shape to pivco-Huffman:
 *   deepest leaf yields all-0 values for the count[max_d] active codes;
 *   at each level above, output[i] = bitmap[i] ? 0 : (child[child_idx++] + 1).
 * That's a merge_vec_cst with constant=0 and a +1 baked into the vec side.
 *
 * Header: max_d (u8) + N (u32).  Per-level bitmap is bit-packed; length in
 * bits = count[k], in bytes = ceil(count[k] / 8).
 */

#define PIVCO_MAX_D 64
typedef struct {
    int   max_d;
    int   counts[PIVCO_MAX_D + 1];   /* counts[k] = #codes reaching level k */
    uint8 *levels[PIVCO_MAX_D];      /* bit-packed bitmaps, levels[0..max_d-1] */
    size_t total_bits;
} PivcoStream;

static void pivco_free(PivcoStream *s) {
    for (int k = 0; k < s->max_d; k++) free(s->levels[k]);
    memset(s, 0, sizeof *s);
}

static void encode_pivco(const uint32 *vals, size_t n, PivcoStream *s) {
    memset(s, 0, sizeof *s);
    int max_d = 0;
    for (size_t i = 0; i < n; i++) {
        if ((int)vals[i] > max_d) max_d = (int)vals[i];
        if ((int)vals[i] >= PIVCO_MAX_D) die("value exceeds PIVCO_MAX_D");
    }
    s->max_d = max_d;

    /* counts[k] = # of values >= k */
    s->counts[0] = (int)n;
    for (int k = 1; k <= max_d; k++) {
        int c = 0;
        for (size_t i = 0; i < n; i++) c += (vals[i] >= (uint32)k);
        s->counts[k] = c;
    }

    /* per-level packed bitmaps (only levels 0..max_d-1) */
    size_t bits = 0;
    for (int k = 0; k < max_d; k++) {
        int nbits = s->counts[k];
        size_t nbytes = (size_t)(nbits + 7) / 8;
        s->levels[k] = calloc(nbytes ? nbytes : 1, 1);
        int pos = 0;
        for (size_t i = 0; i < n; i++) {
            if (vals[i] >= (uint32)k) {
                int bit = (vals[i] == (uint32)k);
                s->levels[k][pos >> 3] |= (uint8)(bit << (pos & 7));
                pos++;
            }
        }
        bits += (size_t)nbits;
    }
    s->total_bits = bits;
}

static int decode_pivco(const PivcoStream *s, uint8 *out, size_t n) {
    if (s->counts[0] != (int)n) ERROR_RETURN("count mismatch");
    int max_d = s->max_d;

    uint8 *child = calloc(n ? n : 1, 1);
    uint8 *next  = calloc(n ? n : 1, 1);
    if (!child || !next) die("oom");

    /* deepest level: count[max_d] codes, all value 0 (leaf produces zeros) */
    int child_n = (max_d == 0) ? (int)n : s->counts[max_d];
    /* child[] is already zeroed */

    /* fold each stored level into the running output, deepest first */
    for (int k = max_d - 1; k >= 0; k--) {
        const uint8 *bm = s->levels[k];
        int n_at_k = s->counts[k];
        int ci = 0;
        for (int i = 0; i < n_at_k; i++) {
            int bit = (bm[i >> 3] >> (i & 7)) & 1;
            if (bit) {
                next[i] = 0;
            } else {
                if (ci >= child_n) { free(child); free(next); ERROR_RETURN("child underrun"); }
                next[i] = (uint8)(child[ci++] + 1);
            }
        }
        if (ci != child_n) { free(child); free(next); ERROR_RETURN("child not fully consumed"); }
        uint8 *t = child; child = next; next = t;
        child_n = n_at_k;
    }

    if (child_n != (int)n) { free(child); free(next); ERROR_RETURN("final size mismatch"); }
    memcpy(out, child, n);
    free(child); free(next);
    return 0;
}

/* Single scratch buffer for the pivco SIMD decoders.  The other half of the
 * ping-pong is the caller's `out` buffer (we pick which-is-which based on the
 * parity of max_d so the final result lands in `out` without a memcpy).
 * Allocated + page-touched in main() before the timed runs so we don't pay
 * first-touch faults inside the BENCH region. */
static uint8_t *g_pivco_child = NULL;

/* ============================================================
 * NEON SIMD pivco decoder
 * ============================================================
 *
 * Each level does a "merge_vec_cst with +1 on the vec side":
 *
 *   output[i] = bitmap[i] ? 0 : (child[idx++] + 1)
 *
 * The +1 is folded into the merge by setting the constant to 0xFF
 * (i.e. -1 in uint8) and adding 1 to the whole output vector after the
 * shuffle: the vec lane becomes child+1, the cst lane wraps 0xFF -> 0.
 * Values stay in uint8 across the whole bottom-up pass; we widen to
 * uint32 only at the very end.  Safe as long as max_d < 256.
 *
 * Shuffle tables (expand_tab + expand_popcnt + expand_tab_pre) are
 * the same V4-strategy tables used by pivco-Huffman; the init code is
 * copied verbatim from src/pivco_huffman_neon_tables.c.
 */
#if defined(__aarch64__) && defined(__ARM_NEON)
#include <arm_neon.h>

static uint8_t g_expand_tab    [256][8]    __attribute__((aligned(32)));
static uint8_t g_expand_tab_pre[9][256][8] __attribute__((aligned(64)));
static uint8_t g_expand_popcnt [256]       __attribute__((aligned(64)));
static int     g_expand_ready = 0;

static void init_expand_tables(void) {
    if (g_expand_ready) return;
    for (int m = 0; m < 256; m++) {
        int n_zeros = 0, n_ones = 0;
        for (int k = 0; k < 8; k++) {
            if (m & (1 << k)) { g_expand_tab[m][k] = (uint8_t)(8 + n_ones); n_ones++; }
            else              { g_expand_tab[m][k] = (uint8_t)n_zeros;      n_zeros++; }
        }
        g_expand_popcnt[m] = (uint8_t)n_ones;
    }
    for (int nr0 = 0; nr0 <= 8; nr0++) {
        for (int m = 0; m < 256; m++) {
            for (int k = 0; k < 8; k++) {
                uint8_t raw = g_expand_tab[m][k];
                g_expand_tab_pre[nr0][m][k] =
                    (raw < 8) ? (uint8_t)(raw + (8 - nr0))
                              : (uint8_t)(raw + 8 + nr0);
            }
        }
    }
    g_expand_ready = 1;
}

/* merge_vec_cst with constant=-1, then +1: produces
 *   output[i] = bitmap[i] ? 0 : (child[idx++] + 1)   (all uint8)
 *
 * Bit convention: bit==0 -> vec (left), bit==1 -> cst (right) = 0xFF, +1 wraps to 0.
 * Adapted from merge_vec_cst_neon in src/pivco_huffman_primitives_neon.h.
 */
static void merge_vec_cst_plus1_neon(const uint8_t *bm, int K,
                                     const uint8_t *vec, uint8_t *out) {
    int lc = 0;
    int j  = 0;
    const uint8x8_t  Rbcast   = vdup_n_u8(0xFF);
    const uint8x16_t Rbcast_q = vdupq_n_u8(0xFF);
    const uint8x8_t  one8     = vdup_n_u8(1);

    /* V5 COM64 main path (ported from production merge_vec_cst_neon, 5cccccc):
     * 64 codes / iter as 4 independent 16-code chunks.  The left (vec) cursor
     * for each chunk is precomputed from a byte-wise popcount prefix sum
     * (pc * 0x0101..), so there's no loop-carried cursor between chunks — the
     * dependency the old per-16 form (lc += 16-nr0-nr1) carried.  Constant
     * right = 0xFF; the post-merge +1 wraps 0xFF->0 for the bit==1 lanes. */
    for (; j + 64 <= K; j += 64) {
        uint64_t mask_u64;
        memcpy(&mask_u64, bm + (j >> 3), 8);
        uint8x8_t bm_v = vcreate_u8(mask_u64);
        uint8x8_t pc_v = vcnt_u8(bm_v);
        uint64_t pc_u64 = vget_lane_u64(vreinterpret_u64_u8(pc_v), 0);
        uint64_t pfx = pc_u64 * 0x0101010101010101ULL;

        uint8_t cr0=0,                  cr1=(uint8_t)(pfx>> 8);
        uint8_t cr2=(uint8_t)(pfx>>24), cr3=(uint8_t)(pfx>>40);
        uint8_t in0=(uint8_t)pc_u64,        in1=(uint8_t)(pc_u64>>16);
        uint8_t in2=(uint8_t)(pc_u64>>32),  in3=(uint8_t)(pc_u64>>48);

        uint8_t m0=(uint8_t) mask_u64,        m1=(uint8_t)(mask_u64>> 8);
        uint8_t m2=(uint8_t)(mask_u64>>16),   m3=(uint8_t)(mask_u64>>24);
        uint8_t m4=(uint8_t)(mask_u64>>32),   m5=(uint8_t)(mask_u64>>40);
        uint8_t m6=(uint8_t)(mask_u64>>48),   m7=(uint8_t)(mask_u64>>56);

#define _VC_CHUNK_P1(idx, cr, in, ma, mb) do {                               \
            uint8_t cl = (uint8_t)((idx)*16 - (cr));                          \
            uint8x16_t L = vld1q_u8(vec + lc + cl);                           \
            uint8x16_t both = vcombine_u8(vget_low_u8(L), Rbcast);           \
            uint8x8_t  s0   = vld1_u8(g_expand_tab[ma]);                      \
            vst1_u8(out + j + (idx)*16,     vadd_u8(vqtbl1_u8(both, s0), one8)); \
            uint8x16x2_t src = {{ L, Rbcast_q }};                            \
            uint8x8_t s1 = vld1_u8(g_expand_tab_pre[in][mb]);                 \
            vst1_u8(out + j + (idx)*16 + 8, vadd_u8(vqtbl2_u8(src, s1), one8)); \
        } while (0)
        _VC_CHUNK_P1(0, cr0, in0, m0, m1);
        _VC_CHUNK_P1(1, cr1, in1, m2, m3);
        _VC_CHUNK_P1(2, cr2, in2, m4, m5);
        _VC_CHUNK_P1(3, cr3, in3, m6, m7);
#undef _VC_CHUNK_P1
        lc += 64 - (uint8_t)(pfx >> 56);
    }
    /* 8-element tail: single-source vqtbl2 against {L, Rbcast_q}. */
    for (; j + 8 <= K; j += 8) {
        uint8_t m = bm[j >> 3];
        uint8x16_t L_full = vcombine_u8(vld1_u8(vec + lc), vdup_n_u8(0));
        uint8x16x2_t src = {{ L_full, Rbcast_q }};
        uint8x8_t shuf = vld1_u8(g_expand_tab_pre[8][m]);
        uint8x8_t o    = vqtbl2_u8(src, shuf);
        vst1_u8(out + j, vadd_u8(o, one8));
        lc += (8 - g_expand_popcnt[m]);
    }
    for (; j < K; j++) {
        int mb = (bm[j >> 3] >> (j & 7)) & 1;
        out[j] = mb ? 0 : (uint8_t)(vec[lc++] + 1);
    }
}

static int decode_pivco_neon(const PivcoStream *s, uint8 *out, size_t n) {
    if (s->counts[0] != (int)n) ERROR_RETURN("count mismatch");
    init_expand_tables();
    int max_d = s->max_d;

    /* Use `out` as one half of the ping-pong so we don't need a final
     * memcpy.  After max_d swaps, child == (initial child) iff max_d is
     * even.  We want child == out at the end, so:
     *   max_d even: start child = out      (the zero'd bottom-level buffer)
     *   max_d odd : start next  = out      (becomes child after the 1st swap)
     */
    uint8_t *child, *next;
    if (max_d % 2 == 0) { child = out;             next  = g_pivco_child; }
    else                { child = g_pivco_child;   next  = out;           }

    /* deepest level supplies counts[max_d] zero bytes; that's all the next
     * merge will read from child.  Don't memset the full N. */
    memset(child, 0, (size_t)s->counts[max_d]);
    for (int k = max_d - 1; k >= 0; k--) {
        merge_vec_cst_plus1_neon(s->levels[k], s->counts[k], child, next);
        uint8_t *t = child; child = next; next = t;
    }
    return 0;
}
#endif  /* aarch64 + NEON */

/* ============================================================
 * AVX-512 VBMI2 SIMD pivco decoder
 * ============================================================
 *
 * Stride-64 variant of the same merge_vec_cst_plus1 idea, lifted from
 * merge_vec_cst_avx512 in src/pivco_huffman_primitives_avx512.h.
 *
 * Per 64 codes:
 *   - load 64 bm bits as a kmask
 *   - vpexpandb child bytes into the 0-bit positions (1-bit positions get 0)
 *   - masked +1: bumps 0-bit lanes by 1, leaves 1-bit lanes at 0
 *   - one 64-byte store
 *
 * Requires AVX-512 BW (mask_blend_epi8) + VBMI2 (maskz_expandloadu_epi8).
 */
#if defined(__x86_64__) && defined(__AVX512VBMI2__)
#include <immintrin.h>

static void merge_vec_cst_plus1_avx512(const uint8_t *bm, int K,
                                       const uint8_t *vec, uint8_t *out) {
    int lc = 0;
    int j  = 0;
    const __m512i one64 = _mm512_set1_epi8(1);

    for (; j + 64 <= K; j += 64) {
        uint64_t mask;
        memcpy(&mask, bm + (j >> 3), 8);
        __mmask64 m  = (__mmask64)mask;
        __mmask64 nm = ~m;
        /* vpexpandb: gather 64 bytes from vec into the 0-bit positions */
        __m512i L = _mm512_maskz_expandloadu_epi8(nm, vec + lc);
        /* masked add: child+1 in 0-bit lanes, 0 in 1-bit lanes */
        __m512i o = _mm512_mask_add_epi8(L, nm, L, one64);
        _mm512_storeu_si512((__m512i *)(out + j), o);
        lc += 64 - __builtin_popcountll(mask);
    }
    /* tail: 16-byte and scalar */
    for (; j + 16 <= K; j += 16) {
        uint16_t m16;
        memcpy(&m16, bm + (j >> 3), 2);
        __mmask16 m  = (__mmask16)m16;
        __mmask16 nm = ~m;
        __m128i L = _mm_maskz_expandloadu_epi8(nm, vec + lc);
        __m128i o = _mm_mask_add_epi8(L, nm, L, _mm_set1_epi8(1));
        _mm_storeu_si128((__m128i *)(out + j), o);
        lc += 16 - __builtin_popcount(m16);
    }
    for (; j < K; j++) {
        int mb = (bm[j >> 3] >> (j & 7)) & 1;
        out[j] = mb ? 0 : (uint8_t)(vec[lc++] + 1);
    }
}

static int decode_pivco_avx512(const PivcoStream *s, uint8 *out, size_t n) {
    if (s->counts[0] != (int)n) ERROR_RETURN("count mismatch");
    int max_d = s->max_d;

    uint8_t *child, *next;
    if (max_d % 2 == 0) { child = out;             next  = g_pivco_child; }
    else                { child = g_pivco_child;   next  = out;           }

    memset(child, 0, (size_t)s->counts[max_d]);
    for (int k = max_d - 1; k >= 0; k--) {
        merge_vec_cst_plus1_avx512(s->levels[k], s->counts[k], child, next);
        uint8_t *t = child; child = next; next = t;
    }
    return 0;
}
#endif  /* x86_64 + AVX-512 VBMI2 */

/* ---------- harness ---------- */

static uint64 nanos_now(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64)ts.tv_sec * 1000000000ull + ts.tv_nsec;
}

/* Geometric-ish small values so the unary stream has realistic mix. */
static uint32 sample_geom(uint64 *state, double p) {
    /* xorshift64* */
    uint64 x = *state;
    x ^= x >> 12; x ^= x << 25; x ^= x >> 27;
    *state = x;
    double u = (double)((x * 2685821657736338717ull) >> 11) / (double)(1ull << 53);
    if (u <= 0.0) u = 1e-12;
    /* floor(log(u)/log(1-p)) */
    double q = 1.0 - p;
    double v = log(u) / log(q);
    if (v < 0) v = 0;
    /* clamp at 55: serial/pair/tunstall decoders need < 56 zeros in a row to
     * detect end-of-stream.  PIVCO_MAX_D = 64 leaves headroom for the
     * truncated tail of the distribution. */
    if (v > 55) v = 55;
    return (uint32)v;
}

int main(int argc, char **argv) {
    double p = (argc > 1) ? atof(argv[1]) : 0.5;
    if (p <= 0.0 || p >= 1.0) die("p must be in (0,1)");
    build_table();

    enum { N = 1 << 20 };          /* ~1M codes */
    enum { OUT_CAP = N * 8 + 64 };

    uint32 *src      = malloc(sizeof(uint32) * N);  /* keep u32 for encoder API + verify */
    uint8  *dec_s    = calloc((size_t)N, 1);
    uint8  *dec_p    = calloc((size_t)N, 1);
    uint8  *dec_t    = calloc((size_t)N, 1);
    uint8  *dec_t64b = calloc((size_t)N + 64, 1);
    uint8  *dec_t64bf = calloc((size_t)N + 64, 1);
    uint8  *dec_pv   = calloc((size_t)N, 1);
    uint8  *dec_pvn  = calloc((size_t)N, 1);
    uint8  *enc      = malloc(OUT_CAP);
    if (!src || !dec_s || !dec_p || !dec_t || !dec_t64b || !dec_t64bf || !dec_pv || !dec_pvn || !enc) die("oom");

    uint64 rng = 0x243F6A8885A308D3ull;
    for (int i = 0; i < N; i++) src[i] = sample_geom(&rng, p);
    printf("# geometric p=%.3f  (avg code length ~ %.2f bits)\n", p, 1.0 + (1.0 - p) / p);

    size_t enc_bytes = encode_unary(src, N, enc, OUT_CAP);
    printf("standard unary: %d codes -> %zu bytes (%.2f bits/code)\n",
           N, enc_bytes, (double)enc_bytes * 8 / N);

    PivcoStream ps;
    encode_pivco(src, N, &ps);
    printf("pivco unary  : max_d=%d, %zu bits across %d levels (%.2f bits/code)\n",
           ps.max_d, ps.total_bits, ps.max_d, (double)ps.total_bits / N);

    /* Pre-allocate the pivco scratch buffer and page-touch every 1 MB output
     * buffer so the BENCH timing doesn't include first-touch page faults
     * (calloc returns lazy-zeroed mmap pages on Linux/macOS). */
    g_pivco_child = malloc(N);
    if (!g_pivco_child) die("oom");
    uint8 *touch[] = { g_pivco_child, dec_s, dec_p, dec_t, dec_t64b, dec_t64bf, dec_pv, dec_pvn };
    for (size_t i = 0; i < sizeof(touch)/sizeof(touch[0]); i++) memset(touch[i], 0, N);

    /* run each decoder REPS times back-to-back; print every run so warm vs
     * cold is visible.  Useful for seeing first-call output-buffer RFO cost. */
    #define REPS 3
    #define BENCH(name, fn) do {                                          \
        for (int r = 0; r < REPS; r++) {                                  \
            uint64 t0 = nanos_now();                                      \
            if (fn) die(name " returned error");                          \
            double ns = (double)(nanos_now() - t0);                       \
            printf("  %-20s [%d/%d] %7.2f ns/code  %7.2f Gcode/s\n",      \
                   name, r + 1, REPS, ns / N, N / ns);                    \
        }                                                                 \
    } while (0)

    BENCH("decode_serial",     decode_serial(enc, dec_s, N));
    BENCH("decode_pair",       decode_pair(enc, dec_p, N));
    BENCH("decode_tunstall",   decode_tunstall(enc, dec_t, N));
    BENCH("decode_tunstall64", decode_tunstall64(enc, dec_t64b, N));
    BENCH("decode_tunstall64_bf", decode_tunstall64_bf(enc, dec_t64bf, N));
    BENCH("decode_pivco",      decode_pivco(&ps, dec_pv, N));
#if defined(__aarch64__) && defined(__ARM_NEON)
    BENCH("decode_pivco_neon", decode_pivco_neon(&ps, dec_pvn, N));
#endif
#if defined(__x86_64__) && defined(__AVX512VBMI2__)
    BENCH("decode_pivco_avx512", decode_pivco_avx512(&ps, dec_pvn, N));
#endif

    /* verify against the source — fail fast on first mismatch */
    #define CHECK(buf) do {                                                          \
        for (int i = 0; i < N; i++)                                                  \
            if (buf[i] != src[i]) {                                                  \
                printf("FAIL " #buf " @%d: %u vs %u\n", i, buf[i], src[i]);          \
                return 1;                                                            \
            }                                                                        \
    } while (0)
    CHECK(dec_s); CHECK(dec_p); CHECK(dec_t); CHECK(dec_t64b); CHECK(dec_t64bf); CHECK(dec_pv); CHECK(dec_pvn);
    printf("OK: all decoders agree with encoder over %d codes\n", N);

    pivco_free(&ps);
    free(g_pivco_child);
    free(src); free(dec_s); free(dec_p); free(dec_t); free(dec_t64b); free(dec_t64bf); free(dec_pv); free(dec_pvn); free(enc);
    return 0;
}
