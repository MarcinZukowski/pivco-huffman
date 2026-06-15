/* bench_partition_avx2.c — partition primitive: compress_tab[256] table
 * lookup vs BMI2 pext-based no-table approach.
 *
 *   - tab    : current production path (1 load from compress_tab, 2 pshufb).
 *   - pext   : derive pshufb control on the fly via _pdep_u32/_pext_u32 on
 *              a packed-indices vector.  No table.  Uses prefix-popcount
 *              indirectly: pext compresses the indices [0..7] vector by
 *              mask, which is conceptually "give me the input positions
 *              of the 1-bits in mask, in order".
 *
 * Both process 8 u16 codes per iter (matches production stride).
 *
 * Build (c6a / c4 / c5):
 *   cc -O3 -march=native -o bench_partition_avx2 extras/bench/bench_partition_avx2.c
 */

#define _POSIX_C_SOURCE 199309L
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <time.h>

#if !defined(__AVX2__) || !defined(__BMI2__) || !defined(__SSE4_1__)
int main(void) { puts("bench_partition_avx2: needs SSE4.1 + AVX2 + BMI2"); return 0; }
#else

#include <immintrin.h>
#include <smmintrin.h>

#define N    8192
#define REPS 100000

static double now_sec(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

/* ============================================================
 *   compress_tab: 256 * 32 bytes, low half = right-pack indices,
 *   high half = left-pack indices.  Built lazily at startup. */
static uint8_t compress_tab[256][32] __attribute__((aligned(32)));
static uint8_t compress_popcnt[256];

static void init_compress_tab(void) {
    for (int mask = 0; mask < 256; mask++) {
        int out_r = 0, out_l = 0;
        for (int i = 0; i < 8; i++) {
            if (mask & (1 << i)) {
                compress_tab[mask][out_r * 2]     = (uint8_t)(i * 2);
                compress_tab[mask][out_r * 2 + 1] = (uint8_t)(i * 2 + 1);
                out_r++;
            } else {
                compress_tab[mask][16 + out_l * 2]     = (uint8_t)(i * 2);
                compress_tab[mask][16 + out_l * 2 + 1] = (uint8_t)(i * 2 + 1);
                out_l++;
            }
        }
        for (int j = out_r * 2; j < 16; j++)
            compress_tab[mask][j] = 0xFF;
        for (int j = out_l * 2; j < 16; j++)
            compress_tab[mask][16 + j] = 0xFF;
        compress_popcnt[mask] = (uint8_t)out_r;
    }
}

/* ============================================================
 *   Scalar reference partition (for correctness).
 *   Input: codes_la[0..n) — left-aligned u16 codes, bit (15-depth) decides side.
 *   Output: left half is written in-place at codes_la[0..n_left),
 *           right half is written to right_out[0..n_right),
 *           bm[k] is the 8-bit mask for codes_la[8k..8k+7] (LSB-first).
 *   Returns n_right. */
static int partition_scalar(uint16_t *codes_la, int n, int depth,
                             uint8_t *bm, uint16_t *right_out)
{
    int n_left = 0, n_right = 0;
    int shift_d = 15 - depth;
    /* Read all input first, then write back (in-place left write would
     * overlap reads otherwise). */
    uint16_t *buf = malloc((size_t)n * sizeof(uint16_t));
    memcpy(buf, codes_la, (size_t)n * sizeof(uint16_t));
    for (int j = 0; j < n; j += 8) {
        int end = j + 8 < n ? j + 8 : n;
        uint8_t mask = 0;
        for (int k = j; k < end; k++) {
            int bit = (buf[k] >> shift_d) & 1;
            mask |= (uint8_t)(bit << (k - j));
        }
        bm[j >> 3] = mask;
        for (int k = j; k < end; k++) {
            int bit = (buf[k] >> shift_d) & 1;
            if (bit) right_out[n_right++] = buf[k];
            else     codes_la[n_left++]  = buf[k];
        }
    }
    free(buf);
    return n_right;
}

/* ============================================================
 *   v_tab: current production AVX2/SSE path (compress_tab + pshufb).
 *   8 codes/iter. */
static int partition_tab(uint16_t *codes_la, int n, int depth,
                          uint8_t *bm, uint16_t *right_out)
{
    int n_left = 0, n_right = 0;
    int j = 0;
    __m128i shift_count = _mm_cvtsi32_si128(depth);

    /* enc_mask8 reconstructed locally: shift each u16 left by `depth`
     * so the original bit-(15-depth) lands at bit 15, then packs/movemask. */
    for (; j + 8 <= n; j += 8) {
        __m128i code_vec = _mm_loadu_si128((const __m128i *)(codes_la + j));
        __m128i shifted = _mm_sll_epi16(code_vec, shift_count);
        __m128i packed  = _mm_packs_epi16(shifted, _mm_setzero_si128());
        uint8_t mask    = (uint8_t)_mm_movemask_epi8(packed);
        bm[j >> 3] = mask;

        const uint8_t *tab = compress_tab[mask];
        __m128i shuf_r = _mm_load_si128((const __m128i *)tab);
        __m128i shuf_l = _mm_load_si128((const __m128i *)(tab + 16));
        __m128i right  = _mm_shuffle_epi8(code_vec, shuf_r);
        __m128i left   = _mm_shuffle_epi8(code_vec, shuf_l);
        int nr = compress_popcnt[mask];
        _mm_storeu_si128((__m128i *)(right_out + n_right), right);
        _mm_storeu_si128((__m128i *)(codes_la  + n_left ), left);
        n_right += nr;
        n_left  += (8 - nr);
    }
    return n_right;
}

/* ============================================================
 *   v_pext: no compress_tab.  Derive pshufb control via BMI2 pdep/pext
 *   on a packed indices vector.
 *
 *   indices = 0x76543210 (8 4-bit indices: position 0 = 0, position 1 = 1, ...).
 *   pext mask: expand each set bit of `mask` to 4 bits = 0x_F_F_F (where each
 *   F-nibble is at a position where mask had a 1-bit).  pext compresses the
 *   indices accordingly.  Resulting 32-bit value has nr 4-bit indices packed
 *   into the low (4*nr) bits.
 *
 *   Then convert 4-bit indices to byte pairs (2k, 2k+1) for pshufb. */
static int partition_pext(uint16_t *codes_la, int n, int depth,
                            uint8_t *bm, uint16_t *right_out)
{
    int n_left = 0, n_right = 0;
    int j = 0;
    __m128i shift_count = _mm_cvtsi32_si128(depth);
    const __m128i dup_shuf = _mm_setr_epi8(0,0,1,1,2,2,3,3,4,4,5,5,6,6,7,7);
    const __m128i odd_offset = _mm_setr_epi8(0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1);

    for (; j + 8 <= n; j += 8) {
        __m128i code_vec = _mm_loadu_si128((const __m128i *)(codes_la + j));
        __m128i shifted = _mm_sll_epi16(code_vec, shift_count);
        __m128i packed  = _mm_packs_epi16(shifted, _mm_setzero_si128());
        uint8_t mask    = (uint8_t)_mm_movemask_epi8(packed);
        bm[j >> 3] = mask;

        /* indices = 0x76543210 (8 nibbles, LSB first = 0,1,2,...,7) */
        uint32_t indices = 0x76543210u;
        uint32_t mask_ex_r = _pdep_u32((uint32_t)mask,           0x11111111u) * 0x0Fu;
        uint32_t mask_ex_l = _pdep_u32((uint32_t)(uint8_t)~mask, 0x11111111u) * 0x0Fu;
        uint32_t comp_r    = _pext_u32(indices, mask_ex_r);  /* nr 4-bit indices */
        uint32_t comp_l    = _pext_u32(indices, mask_ex_l);  /* 8-nr 4-bit indices */

        /* Spread each 4-bit index to a byte (low nibble), then double + odd
         * offset to produce the pshufb byte-pair control. */
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
    return n_right;
}

/* ============================================================
 *   Driver
 * ============================================================ */
typedef int (*part_fn)(uint16_t *, int, int, uint8_t *, uint16_t *);

static double time_fn(part_fn fn, uint16_t *scratch, const uint16_t *orig,
                      int n, int depth, uint8_t *bm, uint16_t *right_out,
                      int reps)
{
    double t0 = now_sec();
    for (int r = 0; r < reps; r++) {
        memcpy(scratch, orig, (size_t)n * sizeof(uint16_t));
        (void)fn(scratch, n, depth, bm, right_out);
    }
    double t1 = now_sec();
    return (t1 - t0) / ((double)n * reps) * 1e9;
}

static int verify_partition(uint16_t *a_left, uint16_t *a_right, int a_nr,
                              uint16_t *b_left, uint16_t *b_right, int b_nr,
                              int n)
{
    if (a_nr != b_nr) return -1;
    int nl = n - a_nr;
    for (int i = 0; i < nl; i++) if (a_left[i]  != b_left[i])  return i;
    for (int i = 0; i < a_nr; i++) if (a_right[i] != b_right[i]) return 1000 + i;
    return -1;
}

int main(int argc, char **argv) {
    int reps = argc > 1 ? atoi(argv[1]) : REPS;
    init_compress_tab();
    printf("bench_partition_avx2: N=%d, REPS=%d\n\n", N, reps);

    uint16_t *orig    = aligned_alloc(64, N * 2 + 128);
    uint16_t *scratch = aligned_alloc(64, N * 2 + 128);
    uint16_t *right_out = aligned_alloc(64, N * 2 + 128);
    uint16_t *right_ref = aligned_alloc(64, N * 2 + 128);
    uint16_t *left_ref  = aligned_alloc(64, N * 2 + 128);
    uint8_t  *bm_ref    = aligned_alloc(64, N / 8 + 32);
    uint8_t  *bm        = aligned_alloc(64, N / 8 + 32);

    /* Try a few depths to sample the bitmap-density spectrum.  Pick u16
     * values with random low bits so the bitmap is roughly uniform-50%. */
    srand(42);
    for (int i = 0; i < N; i++) orig[i] = (uint16_t)rand();

    int depths[] = {0, 1, 2, 4, 8, 12};
    printf("%-6s %10s %10s %10s   %s\n",
           "depth", "scalar", "tab", "pext", "check");
    for (size_t d = 0; d < sizeof(depths)/sizeof(depths[0]); d++) {
        int depth = depths[d];

        /* Reference partition (correctness). */
        memcpy(left_ref, orig, (size_t)N * sizeof(uint16_t));
        int nr_ref = partition_scalar(left_ref, N, depth, bm_ref, right_ref);

        memcpy(scratch, orig, (size_t)N * sizeof(uint16_t));
        int nr_tab = partition_tab(scratch, N, depth, bm, right_out);
        int e_tab = verify_partition(left_ref, right_ref, nr_ref,
                                       scratch, right_out, nr_tab, N);

        memcpy(scratch, orig, (size_t)N * sizeof(uint16_t));
        int nr_pext = partition_pext(scratch, N, depth, bm, right_out);
        int e_pext = verify_partition(left_ref, right_ref, nr_ref,
                                        scratch, right_out, nr_pext, N);

        const char *chk = (e_tab < 0 && e_pext < 0) ? "ok" : "FAIL";

        /* Time scalar separately (no fn-pointer call overhead). */
        double t0 = now_sec();
        for (int r = 0; r < 500; r++) {
            memcpy(scratch, orig, (size_t)N * sizeof(uint16_t));
            (void)partition_scalar(scratch, N, depth, bm, right_out);
        }
        double t1 = now_sec();
        double ts = (t1 - t0) / ((double)N * 500) * 1e9;

        double t_tab  = time_fn(partition_tab,  scratch, orig, N, depth, bm, right_out, reps);
        double t_pext = time_fn(partition_pext, scratch, orig, N, depth, bm, right_out, reps);

        printf("%-6d %8.3fns %8.3fns %8.3fns   %s\n",
               depth, ts, t_tab, t_pext, chk);
        if (e_tab  >= 0) printf("    tab  mismatch at %d (nr_ref=%d, nr=%d)\n",
                                  e_tab,  nr_ref, nr_tab);
        if (e_pext >= 0) printf("    pext mismatch at %d (nr_ref=%d, nr=%d)\n",
                                  e_pext, nr_ref, nr_pext);
    }

    free(orig); free(scratch); free(right_out); free(right_ref); free(left_ref);
    free(bm_ref); free(bm);
    return 0;
}

#endif /* AVX2 + BMI2 + SSE4.1 */
