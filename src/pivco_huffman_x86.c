#include "pivco_huffman.h"
#include "pivco_huffman_common.h"
#include <string.h>

#ifdef PIVCO_HAS_SSE4
#include <smmintrin.h>  /* SSE4.1 */
#ifdef PIVCO_HAS_AVX2
#include <immintrin.h>  /* AVX2 */
#endif
#include "pivco_huffman_x86_flat.h"

/* ---------- SSE4.1 Compress Shuffle Table ----------
 *
 * Identical to the NEON version: for each 8-bit mask, a 16-byte
 * shuffle that packs selected uint16_t elements to the front.
 * pshufb (_mm_shuffle_epi8) is the x86 equivalent of NEON TBL.
 */
/* Combined shuffle table: [256][32] where bytes 0-15 are the shuffle
   for mask (right) and bytes 16-31 are for ~mask (left).
   Loaded as two aligned 16-byte loads from contiguous memory. */
static uint8_t compress_tab[256][32] __attribute__((aligned(32)));
static uint8_t compress_popcnt[256] __attribute__((aligned(64)));
static int     compress_table_ready = 0;

static void init_compress_table(void)
{
    if (compress_table_ready) return;
    for (int mask = 0; mask < 256; mask++) {
        int out_r = 0;
        for (int i = 0; i < 8; i++) {
            if (mask & (1 << i)) {
                compress_tab[mask][out_r * 2]     = (uint8_t)(i * 2);
                compress_tab[mask][out_r * 2 + 1] = (uint8_t)(i * 2 + 1);
                out_r++;
            }
        }
        compress_popcnt[mask] = (uint8_t)out_r;
        for (int j = out_r * 2; j < 16; j++)
            compress_tab[mask][j] = 0x80;

        int out_l = 0;
        for (int i = 0; i < 8; i++) {
            if (!(mask & (1 << i))) {
                compress_tab[mask][16 + out_l * 2]     = (uint8_t)(i * 2);
                compress_tab[mask][16 + out_l * 2 + 1] = (uint8_t)(i * 2 + 1);
                out_l++;
            }
        }
        for (int j = out_l * 2; j < 16; j++)
            compress_tab[mask][16 + j] = 0x80;
    }
    compress_table_ready = 1;
}

/* Partition 8 uint16_t by an 8-bit mask using SSE4.1 pshufb.
   bit=1 → right_out, bit=0 → left_out.
   Source is loaded first, so left_out may overlap src (n_left <= j).
   Returns count of right (bit=1) elements. */
static inline int partition_8_sse(const uint16_t *src,
                                   uint8_t mask,
                                   uint16_t *left_out,
                                   uint16_t *right_out)
{
    __m128i data = _mm_loadu_si128((const __m128i *)src);

    /* Load both shuffle patterns from combined table (contiguous) */
    const uint8_t *tab = compress_tab[mask];
    __m128i shuf_r = _mm_load_si128((const __m128i *)tab);
    __m128i shuf_l = _mm_load_si128((const __m128i *)(tab + 16));

    __m128i right = _mm_shuffle_epi8(data, shuf_r);
    __m128i left  = _mm_shuffle_epi8(data, shuf_l);

    int n_right = compress_popcnt[mask];

    _mm_storeu_si128((__m128i *)right_out, right);
    _mm_storeu_si128((__m128i *)left_out, left);

    return n_right;
}

/* ---------- Leaf scatter-write (SSE4.1) ---------- */

static inline void scatter_write_sse(uint8_t *symbols,
                                      const uint16_t *indices, int n,
                                      uint8_t sym)
{
    int j = 0;
    for (; j + 8 <= n; j += 8) {
        __m128i idx = _mm_loadu_si128((const __m128i *)(indices + j));
        symbols[_mm_extract_epi16(idx, 0)] = sym;
        symbols[_mm_extract_epi16(idx, 1)] = sym;
        symbols[_mm_extract_epi16(idx, 2)] = sym;
        symbols[_mm_extract_epi16(idx, 3)] = sym;
        symbols[_mm_extract_epi16(idx, 4)] = sym;
        symbols[_mm_extract_epi16(idx, 5)] = sym;
        symbols[_mm_extract_epi16(idx, 6)] = sym;
        symbols[_mm_extract_epi16(idx, 7)] = sym;
    }
    for (; j < n; j++) {
        symbols[indices[j]] = sym;
    }
}

/* ---------- x86 Encode (Tree-Walk) ---------- */

/* Pack n values of D bits (D<=8 typical, up to 15) into out, LSB-first.
 * Each element's local code = codes[indices[i]] & ((1<<D)-1).
 * Used for the flat-subtree fast path.  Writes ceil(n*D/8) bytes. */
static inline void pack_D_bits_x86(uint8_t *out, int n, int D,
                                    const uint16_t *indices,
                                    const uint16_t *codes)
{
    uint32_t mask = (1u << D) - 1;
    uint64_t buf = 0;
    int bits_in_buf = 0;
    int byte_idx = 0;
    for (int i = 0; i < n; i++) {
        uint32_t local = (uint32_t)codes[indices[i]] & mask;
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

/* Extract D bits at bit position `bit_pos`.  D <= 16. */
static inline uint32_t extract_D_bits_x86(const uint8_t *in,
                                           int bit_pos, int D)
{
    int byte_idx = bit_pos >> 3;
    int bit_off  = bit_pos & 7;
    uint32_t val = (uint32_t)in[byte_idx];
    if (bit_off + D > 8)  val |= ((uint32_t)in[byte_idx + 1]) << 8;
    if (bit_off + D > 16) val |= ((uint32_t)in[byte_idx + 2]) << 16;
    return (val >> bit_off) & ((1u << D) - 1);
}

/* Only D=4 gets a SIMD fast path on pure SSE4.1.
 *
 * - D=2, D=3, D=5, D=6, D=7 all require either per-byte variable shifts
 *   (AVX2's _mm_srlv_*) or vpmultishiftqb (AVX-512 VBMI2) to build the
 *   per-byte code values efficiently.  Without those, the unpack would
 *   need ~4-8 separate pshufb + immediate shifts + blends, which
 *   benchmarked slower than the scalar FLAT_UNPACK_SWITCH_IDX on AVX-512
 *   (where even vpmultishiftqb wasn't enough for D=3/5/6), so the
 *   SSE4.1 variant is definitely not viable.
 * - D=4 is the special case where the unpack is simple: duplicate +
 *   mask + single-immediate-shift + blend gives (b_i & 0x0F, b_i >> 4)
 *   per input byte without any variable-shift primitive.
 * - D=8 has 256-entry c2s, too big for pshufb; scalar LDR wins.
 *
 * For real-world Zen-3-style hosts stuck on SSE4.1, the IDEAS.md
 * "Zen-3 hybrid block decoder" fallback (per-table selection between
 * PIVCO and trad_huffman_decode_4s) is the right escape for
 * bell_* / proba02 / english / zipfian. */

/* flat_d4_unpack_x86 lives in pivco_huffman_x86_flat.h (shared with
 * bench/bench_micro.c). */

/* Decode n elements through a D-bit packed region + code_to_sym table,
 * scattering to symbols[indices[i]].  Same per-D specialised unpackers
 * as NEON; scalar-fast on x86. */
static inline void flat_decode_scatter_x86(uint8_t *symbols,
                                            const uint16_t *indices, int n,
                                            const uint8_t *bm, int D,
                                            const uint8_t *c2s)
{
    if (D == 4) {
        /* c2s has 16 entries — exactly fills a pshufb register. */
        __m128i c2s_vec = _mm_loadu_si128((const __m128i *)c2s);
        int i = 0;
        for (; i + 16 <= n; i += 16) {
            __m128i codes = flat_d4_unpack_x86(bm + (i >> 1));
            __m128i syms  = _mm_shuffle_epi8(c2s_vec, codes);
            symbols[indices[i     ]] = (uint8_t)_mm_extract_epi8(syms, 0);
            symbols[indices[i +  1]] = (uint8_t)_mm_extract_epi8(syms, 1);
            symbols[indices[i +  2]] = (uint8_t)_mm_extract_epi8(syms, 2);
            symbols[indices[i +  3]] = (uint8_t)_mm_extract_epi8(syms, 3);
            symbols[indices[i +  4]] = (uint8_t)_mm_extract_epi8(syms, 4);
            symbols[indices[i +  5]] = (uint8_t)_mm_extract_epi8(syms, 5);
            symbols[indices[i +  6]] = (uint8_t)_mm_extract_epi8(syms, 6);
            symbols[indices[i +  7]] = (uint8_t)_mm_extract_epi8(syms, 7);
            symbols[indices[i +  8]] = (uint8_t)_mm_extract_epi8(syms, 8);
            symbols[indices[i +  9]] = (uint8_t)_mm_extract_epi8(syms, 9);
            symbols[indices[i + 10]] = (uint8_t)_mm_extract_epi8(syms, 10);
            symbols[indices[i + 11]] = (uint8_t)_mm_extract_epi8(syms, 11);
            symbols[indices[i + 12]] = (uint8_t)_mm_extract_epi8(syms, 12);
            symbols[indices[i + 13]] = (uint8_t)_mm_extract_epi8(syms, 13);
            symbols[indices[i + 14]] = (uint8_t)_mm_extract_epi8(syms, 14);
            symbols[indices[i + 15]] = (uint8_t)_mm_extract_epi8(syms, 15);
        }
        for (; i + 2 <= n; i += 2) {
            uint8_t b = bm[i >> 1];
            symbols[indices[i    ]] = c2s[b & 0x0F];
            symbols[indices[i + 1]] = c2s[b >> 4];
        }
        for (; i < n; i++) {
            uint32_t code = extract_D_bits_x86(bm, i * D, D);
            symbols[indices[i]] = c2s[code];
        }
        return;
    }
    int i = 0;
    switch (D) {
    case 2:
        for (; i + 4 <= n; i += 4) {
            uint8_t b = bm[i >> 2];
            symbols[indices[i    ]] = c2s[(b     ) & 3];
            symbols[indices[i + 1]] = c2s[(b >> 2) & 3];
            symbols[indices[i + 2]] = c2s[(b >> 4) & 3];
            symbols[indices[i + 3]] = c2s[(b >> 6) & 3];
        }
        break;
    case 3:
        for (; i + 8 <= n; i += 8) {
            const uint8_t *p = bm + ((i * 3) >> 3);
            uint32_t w = (uint32_t)p[0] | ((uint32_t)p[1] << 8) | ((uint32_t)p[2] << 16);
            symbols[indices[i    ]] = c2s[(w      ) & 7];
            symbols[indices[i + 1]] = c2s[(w >>  3) & 7];
            symbols[indices[i + 2]] = c2s[(w >>  6) & 7];
            symbols[indices[i + 3]] = c2s[(w >>  9) & 7];
            symbols[indices[i + 4]] = c2s[(w >> 12) & 7];
            symbols[indices[i + 5]] = c2s[(w >> 15) & 7];
            symbols[indices[i + 6]] = c2s[(w >> 18) & 7];
            symbols[indices[i + 7]] = c2s[(w >> 21) & 7];
        }
        break;
    case 4:
        for (; i + 2 <= n; i += 2) {
            uint8_t b = bm[i >> 1];
            symbols[indices[i    ]] = c2s[b & 0x0F];
            symbols[indices[i + 1]] = c2s[b >> 4];
        }
        break;
    case 5:
        for (; i + 8 <= n; i += 8) {
            const uint8_t *p = bm + ((i * 5) >> 3);
            uint64_t w = (uint64_t)p[0] | ((uint64_t)p[1] << 8)
                       | ((uint64_t)p[2] << 16) | ((uint64_t)p[3] << 24)
                       | ((uint64_t)p[4] << 32);
            symbols[indices[i    ]] = c2s[(w      ) & 0x1F];
            symbols[indices[i + 1]] = c2s[(w >>  5) & 0x1F];
            symbols[indices[i + 2]] = c2s[(w >> 10) & 0x1F];
            symbols[indices[i + 3]] = c2s[(w >> 15) & 0x1F];
            symbols[indices[i + 4]] = c2s[(w >> 20) & 0x1F];
            symbols[indices[i + 5]] = c2s[(w >> 25) & 0x1F];
            symbols[indices[i + 6]] = c2s[(w >> 30) & 0x1F];
            symbols[indices[i + 7]] = c2s[(w >> 35) & 0x1F];
        }
        break;
    case 6:
        for (; i + 4 <= n; i += 4) {
            const uint8_t *p = bm + ((i * 6) >> 3);
            uint32_t w = (uint32_t)p[0] | ((uint32_t)p[1] << 8) | ((uint32_t)p[2] << 16);
            symbols[indices[i    ]] = c2s[(w      ) & 0x3F];
            symbols[indices[i + 1]] = c2s[(w >>  6) & 0x3F];
            symbols[indices[i + 2]] = c2s[(w >> 12) & 0x3F];
            symbols[indices[i + 3]] = c2s[(w >> 18) & 0x3F];
        }
        break;
    case 7:
        for (; i + 8 <= n; i += 8) {
            const uint8_t *p = bm + ((i * 7) >> 3);
            uint64_t w = (uint64_t)p[0] | ((uint64_t)p[1] << 8)
                       | ((uint64_t)p[2] << 16) | ((uint64_t)p[3] << 24)
                       | ((uint64_t)p[4] << 32) | ((uint64_t)p[5] << 40)
                       | ((uint64_t)p[6] << 48);
            symbols[indices[i    ]] = c2s[(w      ) & 0x7F];
            symbols[indices[i + 1]] = c2s[(w >>  7) & 0x7F];
            symbols[indices[i + 2]] = c2s[(w >> 14) & 0x7F];
            symbols[indices[i + 3]] = c2s[(w >> 21) & 0x7F];
            symbols[indices[i + 4]] = c2s[(w >> 28) & 0x7F];
            symbols[indices[i + 5]] = c2s[(w >> 35) & 0x7F];
            symbols[indices[i + 6]] = c2s[(w >> 42) & 0x7F];
            symbols[indices[i + 7]] = c2s[(w >> 49) & 0x7F];
        }
        break;
    case 8:
        for (; i < n; i++) symbols[indices[i]] = c2s[bm[i]];
        break;
    }
    for (; i < n; i++) {
        uint32_t code = extract_D_bits_x86(bm, i * D, D);
        symbols[indices[i]] = c2s[code];
    }
}

/* Same as flat_decode_scatter_x86 but writes to symbols[i] directly
 * (used for root-flat where indices are identity). */
static inline void flat_decode_direct_x86(uint8_t *symbols, int n,
                                           const uint8_t *bm, int D,
                                           const uint8_t *c2s)
{
    if (D == 4) {
        __m128i c2s_vec = _mm_loadu_si128((const __m128i *)c2s);
        int i = 0;
        for (; i + 16 <= n; i += 16) {
            __m128i codes = flat_d4_unpack_x86(bm + (i >> 1));
            __m128i syms  = _mm_shuffle_epi8(c2s_vec, codes);
            _mm_storeu_si128((__m128i *)(symbols + i), syms);
        }
        for (; i + 2 <= n; i += 2) {
            uint8_t b = bm[i >> 1];
            symbols[i    ] = c2s[b & 0x0F];
            symbols[i + 1] = c2s[b >> 4];
        }
        for (; i < n; i++) {
            uint32_t code = extract_D_bits_x86(bm, i * D, D);
            symbols[i] = c2s[code];
        }
        return;
    }
    int i = 0;
    switch (D) {
    case 2:
        for (; i + 4 <= n; i += 4) {
            uint8_t b = bm[i >> 2];
            symbols[i    ] = c2s[(b     ) & 3];
            symbols[i + 1] = c2s[(b >> 2) & 3];
            symbols[i + 2] = c2s[(b >> 4) & 3];
            symbols[i + 3] = c2s[(b >> 6) & 3];
        }
        break;
    case 3:
        for (; i + 8 <= n; i += 8) {
            const uint8_t *p = bm + ((i * 3) >> 3);
            uint32_t w = (uint32_t)p[0] | ((uint32_t)p[1] << 8) | ((uint32_t)p[2] << 16);
            symbols[i    ] = c2s[(w      ) & 7];
            symbols[i + 1] = c2s[(w >>  3) & 7];
            symbols[i + 2] = c2s[(w >>  6) & 7];
            symbols[i + 3] = c2s[(w >>  9) & 7];
            symbols[i + 4] = c2s[(w >> 12) & 7];
            symbols[i + 5] = c2s[(w >> 15) & 7];
            symbols[i + 6] = c2s[(w >> 18) & 7];
            symbols[i + 7] = c2s[(w >> 21) & 7];
        }
        break;
    case 4:
        for (; i + 2 <= n; i += 2) {
            uint8_t b = bm[i >> 1];
            symbols[i    ] = c2s[b & 0x0F];
            symbols[i + 1] = c2s[b >> 4];
        }
        break;
    case 5:
        for (; i + 8 <= n; i += 8) {
            const uint8_t *p = bm + ((i * 5) >> 3);
            uint64_t w = (uint64_t)p[0] | ((uint64_t)p[1] << 8)
                       | ((uint64_t)p[2] << 16) | ((uint64_t)p[3] << 24)
                       | ((uint64_t)p[4] << 32);
            symbols[i    ] = c2s[(w      ) & 0x1F];
            symbols[i + 1] = c2s[(w >>  5) & 0x1F];
            symbols[i + 2] = c2s[(w >> 10) & 0x1F];
            symbols[i + 3] = c2s[(w >> 15) & 0x1F];
            symbols[i + 4] = c2s[(w >> 20) & 0x1F];
            symbols[i + 5] = c2s[(w >> 25) & 0x1F];
            symbols[i + 6] = c2s[(w >> 30) & 0x1F];
            symbols[i + 7] = c2s[(w >> 35) & 0x1F];
        }
        break;
    case 6:
        for (; i + 4 <= n; i += 4) {
            const uint8_t *p = bm + ((i * 6) >> 3);
            uint32_t w = (uint32_t)p[0] | ((uint32_t)p[1] << 8) | ((uint32_t)p[2] << 16);
            symbols[i    ] = c2s[(w      ) & 0x3F];
            symbols[i + 1] = c2s[(w >>  6) & 0x3F];
            symbols[i + 2] = c2s[(w >> 12) & 0x3F];
            symbols[i + 3] = c2s[(w >> 18) & 0x3F];
        }
        break;
    case 7:
        for (; i + 8 <= n; i += 8) {
            const uint8_t *p = bm + ((i * 7) >> 3);
            uint64_t w = (uint64_t)p[0] | ((uint64_t)p[1] << 8)
                       | ((uint64_t)p[2] << 16) | ((uint64_t)p[3] << 24)
                       | ((uint64_t)p[4] << 32) | ((uint64_t)p[5] << 40)
                       | ((uint64_t)p[6] << 48);
            symbols[i    ] = c2s[(w      ) & 0x7F];
            symbols[i + 1] = c2s[(w >>  7) & 0x7F];
            symbols[i + 2] = c2s[(w >> 14) & 0x7F];
            symbols[i + 3] = c2s[(w >> 21) & 0x7F];
            symbols[i + 4] = c2s[(w >> 28) & 0x7F];
            symbols[i + 5] = c2s[(w >> 35) & 0x7F];
            symbols[i + 6] = c2s[(w >> 42) & 0x7F];
            symbols[i + 7] = c2s[(w >> 49) & 0x7F];
        }
        break;
    case 8:
        for (; i < n; i++) symbols[i] = c2s[bm[i]];
        break;
    }
    for (; i < n; i++) {
        uint32_t code = extract_D_bits_x86(bm, i * D, D);
        symbols[i] = c2s[code];
    }
}

static void encode_node_x86(const pivco_huffman_table_t *table,
                              int16_t node_id,
                              uint16_t *indices, int n,
                              int depth,
                              const uint16_t *codes, const uint8_t *lens,
                              uint8_t **out_ptr,
                              uint16_t *tmp)
{
    if (n == 0) return;

    const pivco_tree_node_t *node = &table->tree[node_id];
    if (node->symbol >= 0) return; /* leaf */

    /* Flat-subtree fast path: emit n*D packed bits instead of D levels
       of bitmaps.  Detected at build_table time. */
    if (table->flat_depth[node_id] >= 2) {
        int D = table->flat_depth[node_id];
        int total_bytes = (n * D + 7) >> 3;
        uint8_t *out = *out_ptr;
        if (total_bytes > 0) out[total_bytes - 1] = 0;
        pack_D_bits_x86(out, n, D, indices, codes);
        *out_ptr += total_bytes;
        return;
    }

    /* Write n code bits */
    int nbytes = bitmap_bytes(n);
    uint8_t *bm = *out_ptr;
    memset(bm, 0, (size_t)nbytes);

    for (int j = 0; j < n; j++) {
        int idx = indices[j];
        int bit = (codes[idx] >> (lens[idx] - 1 - depth)) & 1;
        if (bit) bitmap_set(bm, j);
    }
    *out_ptr += nbytes;

    /* SSE partition in-place */
    int n_left = 0, n_right = 0;
    int j = 0;

    for (; j + 8 <= n; j += 8) {
        uint8_t mask = bm[j >> 3];
        int nr = partition_8_sse(indices + j, mask,
                                  indices + n_left, tmp + n_right);
        n_right += nr;
        n_left += (8 - nr);
    }
    for (; j < n; j++) {
        if (bitmap_get(bm, j)) {
            tmp[n_right++] = indices[j];
        } else {
            indices[n_left++] = indices[j];
        }
    }

    encode_node_x86(table, node->left, indices, n_left,
                     depth + 1, codes, lens, out_ptr, tmp + n_right);
    encode_node_x86(table, node->right, tmp, n_right,
                     depth + 1, codes, lens, out_ptr, tmp + n_right);
}

int pivco_huffman_encode_x86(const uint8_t *symbols,
                              const pivco_huffman_table_t *table,
                              uint8_t *out, size_t *out_len)
{
    if (!symbols || !table || !out || !out_len) return PIVCO_ERR_NULL;

    init_compress_table();

    const int N = PIVCO_BLOCK_SIZE;

    uint16_t codes[PIVCO_BLOCK_SIZE];
    uint8_t  lens[PIVCO_BLOCK_SIZE];
    for (int i = 0; i < N; i++) {
        codes[i] = table->code[symbols[i]];
        lens[i]  = table->code_len[symbols[i]];
    }

    uint16_t indices[PIVCO_BLOCK_SIZE];
    for (int i = 0; i < N; i++) indices[i] = (uint16_t)i;

    uint16_t tmp[PIVCO_BLOCK_SIZE * 2];
    uint8_t *ptr = out;

    encode_node_x86(table, table->tree_root, indices, N,
                     0, codes, lens, &ptr, tmp);

    *out_len = (size_t)(ptr - out);
    return PIVCO_OK;
}

/* ---------- x86 Decode (Tree-Walk with SSE Partition) ---------- */

/* Half-partition helpers: extract only one side */
static inline int partition_8_sse_right(const uint16_t *src,
                                         uint8_t mask,
                                         uint16_t *right_out)
{
    __m128i data = _mm_loadu_si128((const __m128i *)src);
    __m128i shuf_r = _mm_load_si128((const __m128i *)compress_tab[mask]);
    _mm_storeu_si128((__m128i *)right_out, _mm_shuffle_epi8(data, shuf_r));
    return compress_popcnt[mask];
}

static inline int partition_8_sse_left(const uint16_t *src,
                                        uint8_t mask,
                                        uint16_t *left_out)
{
    __m128i data = _mm_loadu_si128((const __m128i *)src);
    __m128i shuf_l = _mm_load_si128((const __m128i *)(compress_tab[mask] + 16));
    _mm_storeu_si128((__m128i *)left_out, _mm_shuffle_epi8(data, shuf_l));
    return 8 - compress_popcnt[mask];
}

static void decode_node_x86(const pivco_huffman_table_t *table,
                              int16_t node_id,
                              uint16_t *indices, int n,
                              uint8_t *symbols,
                              const uint8_t **in_ptr,
                              uint16_t *tmp,
                              int16_t skip_node)
{
    if (n == 0) return;
    if (node_id == skip_node) return;  /* prefilled by memset */

    const pivco_tree_node_t *node = &table->tree[node_id];
    if (node->symbol >= 0) {
        scatter_write_sse(symbols, indices, n, (uint8_t)node->symbol);
        return;
    }

    /* Flat-subtree fast path. */
    if (table->flat_depth[node_id] >= 2) {
        int D = table->flat_depth[node_id];
        int total_bytes = (n * D + 7) >> 3;
        const uint8_t *bm = *in_ptr;
        *in_ptr += total_bytes;
        const uint8_t *c2s = &table->flat_code_to_sym[table->flat_offset[node_id]];
        flat_decode_scatter_x86(symbols, indices, n, bm, D, c2s);
        return;
    }

    /* Read n code bits */
    int nbytes = bitmap_bytes(n);
    const uint8_t *bm = *in_ptr;
    *in_ptr += nbytes;

    /* Check children for stage fusion */
    const pivco_tree_node_t *left_child  = &table->tree[node->left];
    const pivco_tree_node_t *right_child = &table->tree[node->right];
    int left_leaf  = (left_child->symbol >= 0);
    int right_leaf = (right_child->symbol >= 0);

    if (left_leaf && right_leaf
        && node->left != skip_node && node->right != skip_node) {
        /* Both children are leaves (neither prefilled) — scatter directly */
        uint8_t syms[2] = {(uint8_t)left_child->symbol,
                           (uint8_t)right_child->symbol};
        int j = 0;
        for (; j + 8 <= n; j += 8) {
            uint8_t mask = bm[j >> 3];
            __m128i idx = _mm_loadu_si128((const __m128i *)(indices + j));
            symbols[_mm_extract_epi16(idx, 0)] = syms[(mask >> 0) & 1];
            symbols[_mm_extract_epi16(idx, 1)] = syms[(mask >> 1) & 1];
            symbols[_mm_extract_epi16(idx, 2)] = syms[(mask >> 2) & 1];
            symbols[_mm_extract_epi16(idx, 3)] = syms[(mask >> 3) & 1];
            symbols[_mm_extract_epi16(idx, 4)] = syms[(mask >> 4) & 1];
            symbols[_mm_extract_epi16(idx, 5)] = syms[(mask >> 5) & 1];
            symbols[_mm_extract_epi16(idx, 6)] = syms[(mask >> 6) & 1];
            symbols[_mm_extract_epi16(idx, 7)] = syms[(mask >> 7) & 1];
        }
        for (; j < n; j++) {
            symbols[indices[j]] = syms[bitmap_get(bm, j)];
        }
        return;
    }

    if (left_leaf && node->left == skip_node) {
        /* Left is prefilled leaf — half-partition right only */
        int n_right = 0;
        int j = 0;
        for (; j + 8 <= n; j += 8)
            n_right += partition_8_sse_right(indices + j, bm[j >> 3],
                                              tmp + n_right);
        for (; j < n; j++)
            if (bitmap_get(bm, j)) tmp[n_right++] = indices[j];
        decode_node_x86(table, node->right, tmp, n_right,
                         symbols, in_ptr, tmp + n_right, skip_node);
    } else if (right_leaf && node->right == skip_node) {
        /* Right is prefilled leaf — half-partition left only */
        int n_left = 0;
        int j = 0;
        for (; j + 8 <= n; j += 8)
            n_left += partition_8_sse_left(indices + j, bm[j >> 3],
                                            indices + n_left);
        for (; j < n; j++)
            if (!bitmap_get(bm, j)) indices[n_left++] = indices[j];
        decode_node_x86(table, node->left, indices, n_left,
                         symbols, in_ptr, tmp, skip_node);
    } else {
        /* Full partition */
        int n_left = 0, n_right = 0;
        int j = 0;
        for (; j + 8 <= n; j += 8) {
            uint8_t mask = bm[j >> 3];
            int nr = partition_8_sse(indices + j, mask,
                                      indices + n_left, tmp + n_right);
            n_right += nr;
            n_left += (8 - nr);
        }
        for (; j < n; j++) {
            if (bitmap_get(bm, j))
                tmp[n_right++] = indices[j];
            else
                indices[n_left++] = indices[j];
        }

        /* Recurse into both; child's entry handles leaf/skip_node. */
        decode_node_x86(table, node->left, indices, n_left,
                         symbols, in_ptr, tmp + n_right, skip_node);
        decode_node_x86(table, node->right, tmp, n_right,
                         symbols, in_ptr, tmp + n_right, skip_node);
    }
}

int pivco_huffman_decode_x86(const uint8_t *in, size_t in_len,
                              const pivco_huffman_table_t *table,
                              uint8_t *symbols, size_t *consumed)
{
    if (!in || !table || !symbols || !consumed) return PIVCO_ERR_NULL;

    init_compress_table();

    const int N = PIVCO_BLOCK_SIZE;
    (void)in_len;
    const uint8_t *ptr = in;

    const pivco_tree_node_t *root = &table->tree[table->tree_root];

    /* Root is leaf — fill everything */
    if (root->symbol >= 0) {
        memset(symbols, (uint8_t)root->symbol, (size_t)N);
        *consumed = 0;
        return PIVCO_OK;
    }

    /* Root is a flat subtree (whole tree flat, D>=2) — write symbols[i]
       directly, no prefill, no indices[]. */
    if (table->flat_depth[table->tree_root] >= 2) {
        int D = table->flat_depth[table->tree_root];
        int total_bytes = (N * D + 7) >> 3;
        const uint8_t *bm = ptr;
        ptr += total_bytes;
        const uint8_t *c2s = &table->flat_code_to_sym[table->flat_offset[table->tree_root]];
        flat_decode_direct_x86(symbols, N, bm, D, c2s);
        *consumed = (size_t)(ptr - in);
        return PIVCO_OK;
    }

    /* Read root bitmap */
    int nbytes = bitmap_bytes(N);
    const uint8_t *bm = ptr;
    ptr += nbytes;

    const pivco_tree_node_t *left_child  = &table->tree[root->left];
    const pivco_tree_node_t *right_child = &table->tree[root->right];
    int left_leaf  = (left_child->symbol >= 0);
    int right_leaf = (right_child->symbol >= 0);

    if (left_leaf && right_leaf) {
        /* Both-leaves at root — sequential stores, no scatter */
        uint8_t syms[2] = {(uint8_t)left_child->symbol,
                           (uint8_t)right_child->symbol};
        for (int j = 0; j < N; j++)
            symbols[j] = syms[(bm[j >> 3] >> (j & 7)) & 1];
        *consumed = (size_t)(ptr - in);
        return PIVCO_OK;
    }

    /* Prefill output with most frequent symbol */
    uint8_t prefill_sym = table->prefill_sym;
    int16_t skip_node = table->prefill_node;
    memset(symbols, prefill_sym, (size_t)N);

    /* Partition at root — generate identity indices in-place */
    uint16_t indices[PIVCO_BLOCK_SIZE];
    uint16_t tmp[PIVCO_BLOCK_SIZE * 2];

    if (left_leaf && root->left == skip_node) {
        /* Left is prefilled — half-partition right only at root */
        int n_right = 0;
        for (int j = 0; j + 8 <= N; j += 8) {
            /* Generate identity indices [j..j+7] and partition right */
            uint16_t id[8];
            for (int k = 0; k < 8; k++) id[k] = (uint16_t)(j + k);
            n_right += partition_8_sse_right(id, bm[j >> 3], tmp + n_right);
        }
        decode_node_x86(table, root->right, tmp, n_right,
                         symbols, &ptr, tmp + n_right, skip_node);
    } else if (right_leaf && root->right == skip_node) {
        /* Right is prefilled — half-partition left only at root */
        int n_left = 0;
        for (int j = 0; j + 8 <= N; j += 8) {
            uint16_t id[8];
            for (int k = 0; k < 8; k++) id[k] = (uint16_t)(j + k);
            n_left += partition_8_sse_left(id, bm[j >> 3], indices + n_left);
        }
        decode_node_x86(table, root->left, indices, n_left,
                         symbols, &ptr, tmp, skip_node);
    } else {
        /* Full partition at root */
        int n_left = 0, n_right = 0;
        for (int j = 0; j + 8 <= N; j += 8) {
            uint16_t id[8];
            for (int k = 0; k < 8; k++) id[k] = (uint16_t)(j + k);
            uint8_t mask = bm[j >> 3];
            int nr = partition_8_sse(id, mask,
                                      indices + n_left, tmp + n_right);
            n_right += nr;
            n_left += (8 - nr);
        }

        /* Recurse into both; child's entry handles leaf/skip_node. */
        decode_node_x86(table, root->left, indices, n_left,
                         symbols, &ptr, tmp + n_right, skip_node);
        decode_node_x86(table, root->right, tmp, n_right,
                         symbols, &ptr, tmp + n_right, skip_node);
    }

    *consumed = (size_t)(ptr - in);
    return PIVCO_OK;
}

#endif /* PIVCO_HAS_SSE4 */
