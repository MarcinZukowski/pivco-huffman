/* bench_golomb — Rice (Golomb power-of-2 M) vs fast FSE on skewed
 * binary bitmaps.  Motivated by the IDEAS.md "Golomb/Rice for skewed
 * bitmaps" entry: per-node bitmaps are geometric in their minority-
 * bit run lengths, which is the regime Rice was built for.
 *
 *   ./build/pivco_bench_golomb           # default 50k iters
 *   ./build/pivco_bench_golomb 200000
 *
 * For each p_major in {0.50, 0.55, 0.60, ..., 0.95, 0.99}, generates
 * a fixed-size IID bitmap and reports:
 *   - encoded size (bytes) and ratio (bits/bit input) for both Rice
 *     and FSE x=8 (the "fast FSE" wide-cursor decoder from
 *     bench_fse_xy_micro.c — beats stock x=2 by 2.5-3x on the per-
 *     node primitive).
 *   - encode/decode MB/s (min of 5 batches × N iters).
 *
 * Rice parameter k auto-selected per p_major:
 *   k = max(0, floor(log2(p / (1-p))))
 * which matches the standard optimal-k formula for geometric run
 * lengths.  k is encoded into a 1-byte header alongside the
 * majority-bit value and total minority-bit count.
 *
 * Bitmap size fixed at 1024 bytes (8192 bits — a typical ph block
 * size).  Reproducibility: PRNG seeded deterministically per cell.
 *
 * Build:   cmake --build build --target pivco_bench_golomb
 */

#define FSE_STATIC_LINKING_ONLY
#include "fse.h"
#include "bitstream.h"
#include "pivco_fse.h"
#include "pivco_fse_tables.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

/* Lemire's EWAH (Enhanced Word-Aligned Hybrid) — vendored at ext/ewah,
 * wrapped via extras/bench/bench_ewah_wrapper.cpp.  Built in when CMake
 * detects ext/ewah; otherwise the EWAH columns just stay empty. */
#ifdef PIVCO_HAS_EWAH
extern size_t ewah_encode(const uint8_t *bm, size_t n_bits,
                           int majority_bit,
                           uint8_t *out, size_t out_cap);
extern int    ewah_decode(const uint8_t *enc, size_t enc_len,
                           uint8_t *bm_out, size_t n_bits,
                           int majority_bit);
#endif

/* ============================================================
 *  Rice coding (Golomb with M = 2^k)
 * ============================================================ */

/* Pick the optimal Rice parameter k for a given p_major.
 * E[run_length] = p / (1-p); optimal k ≈ floor(log2(E[run]·ln 2)).
 * We use the simpler floor(log2(p/(1-p))) which is within 1 of
 * optimal across the range we care about (0.55..0.99). */
static int rice_pick_k(double p_major)
{
    if (p_major <= 0.5) return 0;
    double expected_run = p_major / (1.0 - p_major);
    int k = (int)floor(log2(expected_run));
    if (k < 0) k = 0;
    if (k > 16) k = 16;   /* cap — unary part would blow up otherwise */
    return k;
}

/* Wire format:
 *   byte 0  : header bits — k in [0..4], majority-bit in [5], 3 reserved
 *   bytes 1-2 : n_minority (uint16 LE)
 *   payload : Rice codes, one per minority bit, encoding the run of
 *             majority bits BEFORE it.  Unary part = q '1' bits then a
 *             terminating '0' bit; binary part = k bits of remainder.
 *             Bits packed LSB-first into bytes (matches pivco_fse).
 *
 * Trailing run (any majority bits after the last minority bit) is
 * implicit — the decoder fills majority once n_minority codes are
 * decoded.  This avoids the "what's the last code's run length"
 * ambiguity.
 */

static size_t rice_encode(const uint8_t *bm, size_t n_bits,
                          int k, int majority_bit,
                          uint8_t *out, size_t out_cap)
{
    if (out_cap < 3) return 0;
    /* Count minority bits first so we can write the header. */
    size_t n_minority = 0;
    for (size_t i = 0; i < n_bits; i++) {
        int b = (bm[i >> 3] >> (i & 7)) & 1;
        if (b != majority_bit) n_minority++;
    }
    if (n_minority > 0xFFFF) return 0;

    out[0] = (uint8_t)((k & 0x1F) | ((majority_bit & 1) << 5));
    out[1] = (uint8_t)(n_minority & 0xFF);
    out[2] = (uint8_t)((n_minority >> 8) & 0xFF);

    uint8_t *op = out + 3;
    uint8_t * const olim = out + out_cap;
    uint64_t acc = 0;
    int acc_bits = 0;

    /* Drain helper: spill full bytes from `acc` once `acc_bits >= 8`,
     * leaving acc_bits in [0..7].  Called between every phase so the
     * next OR can't overflow the 64-bit accumulator. */
    #define DRAIN() do { \
        while (acc_bits >= 8) { \
            if (op >= olim) return 0; \
            *op++ = (uint8_t)acc; \
            acc >>= 8; \
            acc_bits -= 8; \
        } \
    } while (0)

    size_t run = 0;
    for (size_t i = 0; i < n_bits; i++) {
        int b = (bm[i >> 3] >> (i & 7)) & 1;
        if (b == majority_bit) {
            run++;
        } else {
            /* Emit Rice code for `run`. */
            uint64_t q   = run >> k;
            uint64_t rem = run & (((uint64_t)1 << k) - 1);

            /* Unary: q '1' bits then a '0' bit terminator.  Chunked to
             * keep each OR within the 64-bit accumulator (we always
             * drain back to acc_bits < 8 between chunks, so adding up
             * to 56 bits stays under 64). */
            while (q >= 56) {
                acc |= ((uint64_t)((1ULL << 56) - 1)) << acc_bits;
                acc_bits += 56;
                q -= 56;
                DRAIN();
            }
            if (q > 0) {
                acc |= ((uint64_t)((1ULL << q) - 1)) << acc_bits;
                acc_bits += (int)q;
                DRAIN();
            }
            /* Terminating 0 bit (the bit stays 0 in acc; just bump). */
            acc_bits += 1;
            DRAIN();
            /* Binary remainder, k bits. */
            if (k > 0) {
                acc |= (rem & (((uint64_t)1 << k) - 1)) << acc_bits;
                acc_bits += k;
                DRAIN();
            }
            run = 0;
        }
    }
    #undef DRAIN
    /* trailing majority run intentionally not emitted; decoder fills it */

    /* Flush any remaining bits with zero padding. */
    if (acc_bits > 0) {
        if (op >= olim) return 0;
        *op++ = (uint8_t)acc;
    }
    return op - out;
}

static int rice_decode(const uint8_t *enc, size_t enc_len,
                       uint8_t *bm, size_t n_bits)
{
    if (enc_len < 3) return -1;
    int k             = enc[0] & 0x1F;
    int majority_bit  = (enc[0] >> 5) & 1;
    int minority_bit  = majority_bit ^ 1;
    size_t n_minority = (size_t)enc[1] | ((size_t)enc[2] << 8);

    /* Prefill bitmap with majority bits, then overlay minority. */
    memset(bm, majority_bit ? 0xFF : 0x00, (n_bits + 7) >> 3);
    /* Mask off bits past n_bits in the final byte. */
    size_t tail = n_bits & 7;
    if (tail && majority_bit) {
        size_t last = (n_bits + 7) >> 3;
        if (last > 0) bm[last - 1] &= (uint8_t)((1U << tail) - 1);
    }

    /* Bit reader: 64-bit window, LSB-first byte order. */
    const uint8_t *ip   = enc + 3;
    const uint8_t *ilim = enc + enc_len;
    uint64_t window = 0;
    int      have   = 0;

    #define FILL() \
        while (have <= 56 && ip < ilim) { \
            window |= ((uint64_t)*ip++) << have; \
            have += 8; \
        }

    size_t pos = 0;
    for (size_t m = 0; m < n_minority; m++) {
        FILL();
        /* Unary: count trailing 1 bits. */
        uint64_t q = 0;
        while (have > 0) {
            /* Find lowest 0-bit. */
            uint64_t inv = ~window;
            if (inv == 0) {
                /* All 1s in window; consume all and refill. */
                q += have;
                window = 0;
                have = 0;
                FILL();
                if (have == 0) return -2;  /* truncated */
                continue;
            }
            int ctz = __builtin_ctzll(inv);
            if (ctz >= have) {
                /* The 0 bit is past current window; consume all 1s
                 * and refill. */
                q += have;
                window = 0;
                have = 0;
                FILL();
                if (have == 0) return -3;
                continue;
            }
            q += ctz;
            /* Consume the 0 terminator bit. */
            window >>= (ctz + 1);
            have -= ctz + 1;
            break;
        }
        /* Read k-bit remainder. */
        FILL();
        if (have < k) return -4;
        uint64_t rem = window & (((uint64_t)1 << k) - 1);
        window >>= k;
        have -= k;
        uint64_t run = (q << k) + rem;

        pos += run;
        if (pos >= n_bits) return -5;  /* overflow */
        /* Write minority_bit at position `pos`. */
        if (minority_bit) bm[pos >> 3] |=  (uint8_t)(1U << (pos & 7));
        else              bm[pos >> 3] &= ~(uint8_t)(1U << (pos & 7));
        pos++;
    }
    #undef FILL
    return 0;
}

/* ============================================================
 *  FSE x=8 (wide-cursor) encode + decode.  Lifted from
 *  extras/bench/bench_fse_xy_micro.c — the "fast FSE" the IDEAS entry
 *  benchmarks against.  Stock FSE library ships x=2 in
 *  FSE_compress/FSE_decompress; the wide-cursor microbench shows
 *  x=8 closes most of the per-node decode gap to huf0_x2.
 * ============================================================ */

static FSE_CTable *g_ct;
static FSE_DTable *g_dt;
static int         g_tid;

static void build_tables_for_p(double p_major)
{
    int t_id = pivco_fse_select_table(p_major);
    g_tid = t_id;
    if (t_id < 1) { g_ct = NULL; g_dt = NULL; return; }
    g_ct = FSE_createCTable(PIVCO_FSE_MAX_SYMBOL, PIVCO_FSE_TABLE_LOG);
    g_dt = FSE_createDTable(PIVCO_FSE_TABLE_LOG);
    FSE_buildCTable(g_ct, pivco_fse_norm[t_id],
                     PIVCO_FSE_MAX_SYMBOL, PIVCO_FSE_TABLE_LOG);
    FSE_buildDTable(g_dt, pivco_fse_norm[t_id],
                     PIVCO_FSE_MAX_SYMBOL, PIVCO_FSE_TABLE_LOG);
}

static void free_tables(void)
{
    if (g_ct) { FSE_freeCTable(g_ct); g_ct = NULL; }
    if (g_dt) { FSE_freeDTable(g_dt); g_dt = NULL; }
}

/* x-cursor FSE encoder (n must be multiple of x). */
static size_t fse_encode_x(int x, const uint8_t *src, size_t n,
                            void *dst, size_t dst_cap,
                            const FSE_CTable *ct)
{
    if (x < 2 || x > 16) return 0;
    if (n % (size_t)x != 0) return 0;

    BIT_CStream_t bitC;
    if (FSE_isError(BIT_initCStream(&bitC, dst, dst_cap))) return 0;
    FSE_CState_t st[16];
    size_t i = n;
    for (int kk = x - 1; kk >= 0; kk--) {
        FSE_initCState2(&st[kk], ct, src[--i]);
    }
    while (i > 0) {
        int pushed = 0;
        for (int kk = x - 1; kk >= 0; kk--) {
            FSE_encodeSymbol(&bitC, &st[kk], src[--i]);
            pushed++;
            if (pushed == 5 && i > 0) {
                BIT_flushBitsFast(&bitC);
                pushed = 0;
            }
        }
        if (pushed > 0) BIT_flushBitsFast(&bitC);
    }
    for (int kk = x - 1; kk >= 0; kk--) {
        FSE_flushCState(&bitC, &st[kk]);
    }
    return BIT_closeCStream(&bitC);
}

/* x=8 y=1 decoder (FSE-reference tail pattern from bench_fse_xy_micro). */
static size_t fse_decode_x8(const void *src, size_t src_len,
                             uint8_t *dst, size_t dst_expected,
                             const FSE_DTable *dt)
{
    BIT_DStream_t bitD;
    if (FSE_isError(BIT_initDStream(&bitD, src, src_len))) return 0;
    FSE_DState_t s[8];
    for (int kk = 0; kk < 8; kk++) FSE_initDState(&s[kk], &bitD, dt);
    uint8_t *op = dst;
    uint8_t * const olim = dst + dst_expected;
    /* Main loop uses the safe FSE_decodeSymbol (vs Fast) because at
     * very high skew (p ≥ 0.95) the encoded stream is so small that
     * the mid-round reload between decodes 4 and 5 can return without
     * refilling.  Fast doesn't mask the resulting junk bits, the
     * decoder state goes out of DTable bounds, and the next decode
     * SEGVs on the table-lookup read.  Safe masks to keep state in
     * range; ~3-8% slower per decode but correct at every skew. */
    while ((BIT_reloadDStream(&bitD) == BIT_DStream_unfinished)
           & (op + 8 <= olim)) {
        op[0] = FSE_decodeSymbol(&s[0], &bitD);
        op[1] = FSE_decodeSymbol(&s[1], &bitD);
        op[2] = FSE_decodeSymbol(&s[2], &bitD);
        op[3] = FSE_decodeSymbol(&s[3], &bitD);
        BIT_reloadDStream(&bitD);
        op[4] = FSE_decodeSymbol(&s[4], &bitD);
        op[5] = FSE_decodeSymbol(&s[5], &bitD);
        op[6] = FSE_decodeSymbol(&s[6], &bitD);
        op[7] = FSE_decodeSymbol(&s[7], &bitD);
        op += 8;
    }
    while (op + 8 <= olim) {
        int overflowed = 0;
        for (int kk = 0; kk < 8; kk++) {
            *op++ = FSE_decodeSymbol(&s[kk], &bitD);
            if (BIT_reloadDStream(&bitD) == BIT_DStream_overflow) {
                for (int jj = kk + 1; jj < 8 && op < olim; jj++)
                    *op++ = FSE_decodeSymbol(&s[jj], &bitD);
                overflowed = 1;
                break;
            }
        }
        if (overflowed) break;
    }
    for (int kk = 0; kk < 8 && op < olim; kk++)
        *op++ = FSE_decodeSymbol(&s[kk], &bitD);
    return op - dst;
}


/* ============================================================
 *  Test fixture helpers.
 * ============================================================ */

static uint64_t xs_state = 0xc0ffee123456789ULL;
static uint64_t xs(void) {
    xs_state ^= xs_state << 13;
    xs_state ^= xs_state >> 7;
    xs_state ^= xs_state << 17;
    return xs_state;
}
static void xs_reseed(uint64_t seed) { xs_state = seed | 1; }

/* Fill `bytes` with IID bits where P(bit=0) = p_major. */
static void fill_pmajor(uint8_t *buf, size_t bytes, double p_major)
{
    for (size_t i = 0; i < bytes; i++) {
        uint8_t b = 0;
        for (int j = 0; j < 8; j++) {
            int one = ((double)(xs() & 0xFFFF) / 65535.0) > p_major;
            b |= ((uint8_t)one) << j;
        }
        buf[i] = b;
    }
}

/* Count empirical p_major from the actual generated bitmap. */
static double measure_p_major(const uint8_t *buf, size_t bytes)
{
    size_t total = bytes * 8;
    size_t zeros = 0;
    for (size_t i = 0; i < bytes; i++) {
        zeros += 8 - __builtin_popcount(buf[i]);
    }
    double p0 = (double)zeros / (double)total;
    return (p0 >= 0.5) ? p0 : (1.0 - p0);
}

static int measure_majority_bit(const uint8_t *buf, size_t bytes)
{
    size_t zeros = 0;
    for (size_t i = 0; i < bytes; i++) {
        zeros += 8 - __builtin_popcount(buf[i]);
    }
    return (zeros >= bytes * 4) ? 0 : 1;
}

static double now_ns(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec * 1e9 + (double)ts.tv_nsec;
}

#define N_BATCHES 5


/* ============================================================
 *  Main sweep
 * ============================================================ */

int main(int argc, char **argv)
{
    int iters = 50000;
    if (argc > 1) iters = atoi(argv[1]);
    if (iters < 1000) iters = 1000;

    pivco_fse_init();

    /* Probability sweep: 0.50, 0.55, ..., 0.95, 0.99. */
    static const double p_sweep[] = {
        0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 0.99
    };
    const int n_p = sizeof(p_sweep) / sizeof(p_sweep[0]);

    /* Bitmap size — 1024 bytes (8192 bits), a typical ph block size.
     * Must be a multiple of 8 for the x=8 FSE encoder. */
    const size_t BYTES = 1024;

    static uint8_t src[2048];
    static uint8_t rice_enc[8192];
    static uint8_t fse_enc[8192];
    static uint8_t ewah_enc[8192];
    static uint8_t dec[2048];

    printf("# bench_golomb — Rice (Golomb-k, M=2^k) vs fast FSE (x=8)\n");
    printf("# bitmap size: %zu B (%zu bits).  Iters/cell: %d.  Min of %d batches.\n",
            BYTES, BYTES * 8, iters, N_BATCHES);
    printf("# size columns are in BYTES.  Speed columns in MB/s of input bitmap.\n");
    printf("# entropy column is Shannon H(p_major) in bits/bit — the information\n");
    printf("# lower bound on encoded size.  ratio columns are encoded_bits / bitmap_bits.\n");
    printf("# Rice 'k' is auto-picked per p_major; column reports the chosen value.\n");
    printf("# FSE column: x=8 cursors, y=1 round — the fast-FSE wide-cursor variant.\n");
#ifdef PIVCO_HAS_EWAH
    printf("# EWAH column: Lemire's Enhanced Word-Aligned Hybrid (ext/ewah,\n");
    printf("#   github.com/lemire/EWAHBoolArray), uword = uint64_t.\n");
#endif
    printf("#\n");
    printf("%5s %4s %7s | %3s %5s %5s %7s %7s | %3s %5s %5s %7s %7s"
#ifdef PIVCO_HAS_EWAH
           " | %5s %5s %7s %7s"
#endif
           "\n",
           "pmaj", "tid", "entropy",
           "k", "Rsz", "Rrat", "RencMB", "RdecMB",
           "x", "Fsz", "Frat", "FencMB", "FdecMB"
#ifdef PIVCO_HAS_EWAH
           , "Esz", "Erat", "EencMB", "EdecMB"
#endif
           );
    fflush(stdout);

    for (int pi = 0; pi < n_p; pi++) {
        double p = p_sweep[pi];
        xs_reseed(0xb1ad5eed00000000ULL + (uint64_t)pi);
        fill_pmajor(src, BYTES, p);

        /* Empirical p (the generator is IID Bernoulli but small samples
         * drift a bit; report the measured majority frequency). */
        double p_emp = measure_p_major(src, BYTES);
        int    maj   = measure_majority_bit(src, BYTES);

        double H = (p_emp <= 0.0 || p_emp >= 1.0) ? 0.0 :
                    -p_emp * log2(p_emp) - (1.0 - p_emp) * log2(1.0 - p_emp);

        /* ---------- Rice ---------- */
        int k = rice_pick_k(p_emp);
        size_t rsz = rice_encode(src, BYTES * 8, k, maj,
                                  rice_enc, sizeof(rice_enc));
        if (rsz == 0) {
            printf("%5.2f  ERR (rice_encode failed)\n", p);
            continue;
        }
        /* Roundtrip check. */
        memset(dec, 0xAA, BYTES);
        if (rice_decode(rice_enc, rsz, dec, BYTES * 8) != 0
            || memcmp(src, dec, BYTES) != 0) {
            printf("%5.2f  ERR (rice roundtrip)\n", p);
            continue;
        }
        double r_ratio = (double)(rsz * 8) / (double)(BYTES * 8);

        /* Time Rice encode. */
        double r_enc_best = 0.0;
        for (int b = 0; b < N_BATCHES; b++) {
            volatile size_t sink = 0;
            double t0 = now_ns();
            for (int i = 0; i < iters; i++)
                sink ^= rice_encode(src, BYTES * 8, k, maj,
                                    rice_enc, sizeof(rice_enc));
            double t1 = now_ns();
            (void)sink;
            double mb = (double)BYTES * (double)iters / (t1 - t0) * 1e3;
            if (mb > r_enc_best) r_enc_best = mb;
        }
        /* Time Rice decode. */
        double r_dec_best = 0.0;
        for (int b = 0; b < N_BATCHES; b++) {
            volatile uint8_t sink = 0;
            double t0 = now_ns();
            for (int i = 0; i < iters; i++) {
                rice_decode(rice_enc, rsz, dec, BYTES * 8);
                sink ^= dec[0] ^ dec[BYTES - 1];
            }
            double t1 = now_ns();
            (void)sink;
            double mb = (double)BYTES * (double)iters / (t1 - t0) * 1e3;
            if (mb > r_dec_best) r_dec_best = mb;
        }

        /* ---------- FSE x=8 ---------- */
        int x = 8;
        size_t fsz = 0;
        double f_enc_best = 0.0, f_dec_best = 0.0;
        double f_ratio = 0.0;
        const char *f_note = "";

        build_tables_for_p(p_emp);
        if (!g_ct) {
            f_note = "(no FSE table for p<0.5)";
        } else {
            fsz = fse_encode_x(x, src, BYTES, fse_enc, sizeof(fse_enc), g_ct);
            if (fsz == 0) {
                f_note = "(FSE encode failed)";
            } else {
                memset(dec, 0xAA, BYTES);
                size_t out = fse_decode_x8(fse_enc, fsz, dec, BYTES, g_dt);
                if (out != BYTES || memcmp(src, dec, BYTES) != 0) {
                    f_note = "(FSE roundtrip mismatch)";
                    fsz = 0;
                }
            }
        }
        if (fsz > 0) {
            f_ratio = (double)(fsz * 8) / (double)(BYTES * 8);
            for (int b = 0; b < N_BATCHES; b++) {
                volatile size_t sink = 0;
                double t0 = now_ns();
                for (int i = 0; i < iters; i++)
                    sink ^= fse_encode_x(x, src, BYTES, fse_enc,
                                          sizeof(fse_enc), g_ct);
                double t1 = now_ns();
                (void)sink;
                double mb = (double)BYTES * (double)iters / (t1 - t0) * 1e3;
                if (mb > f_enc_best) f_enc_best = mb;
            }
            for (int b = 0; b < N_BATCHES; b++) {
                volatile uint8_t sink = 0;
                double t0 = now_ns();
                for (int i = 0; i < iters; i++) {
                    fse_decode_x8(fse_enc, fsz, dec, BYTES, g_dt);
                    sink ^= dec[0] ^ dec[BYTES - 1];
                }
                double t1 = now_ns();
                (void)sink;
                double mb = (double)BYTES * (double)iters / (t1 - t0) * 1e3;
                if (mb > f_dec_best) f_dec_best = mb;
            }
        }

        /* ---------- EWAH (Lemire's Enhanced Word-Aligned Hybrid) ---------- */
#ifdef PIVCO_HAS_EWAH
        size_t esz = 0;
        double e_enc_best = 0.0, e_dec_best = 0.0;
        double e_ratio = 0.0;

        esz = ewah_encode(src, BYTES * 8, maj, ewah_enc, sizeof(ewah_enc));
        if (esz > 0) {
            memset(dec, 0xAA, BYTES);
            if (ewah_decode(ewah_enc, esz, dec, BYTES * 8, maj) != 0
                || memcmp(src, dec, BYTES) != 0) {
                esz = 0;
            }
        }
        if (esz > 0) {
            e_ratio = (double)(esz * 8) / (double)(BYTES * 8);
            for (int b = 0; b < N_BATCHES; b++) {
                volatile size_t sink = 0;
                double t0 = now_ns();
                for (int i = 0; i < iters; i++)
                    sink ^= ewah_encode(src, BYTES * 8, maj,
                                         ewah_enc, sizeof(ewah_enc));
                double t1 = now_ns();
                (void)sink;
                double mb = (double)BYTES * (double)iters / (t1 - t0) * 1e3;
                if (mb > e_enc_best) e_enc_best = mb;
            }
            for (int b = 0; b < N_BATCHES; b++) {
                volatile uint8_t sink = 0;
                double t0 = now_ns();
                for (int i = 0; i < iters; i++) {
                    ewah_decode(ewah_enc, esz, dec, BYTES * 8, maj);
                    sink ^= dec[0] ^ dec[BYTES - 1];
                }
                double t1 = now_ns();
                (void)sink;
                double mb = (double)BYTES * (double)iters / (t1 - t0) * 1e3;
                if (mb > e_dec_best) e_dec_best = mb;
            }
        }
#endif

        printf("%5.3f %4d %7.4f | %3d %5zu %5.3f %7.1f %7.1f | ",
               p_emp, g_tid, H,
               k, rsz, r_ratio, r_enc_best, r_dec_best);
        if (fsz > 0) {
            printf("%3d %5zu %5.3f %7.1f %7.1f",
                   x, fsz, f_ratio, f_enc_best, f_dec_best);
        } else {
            printf("%3s %5s %5s %7s %7s",
                   "-", "-", "-", "-", "-");
        }
#ifdef PIVCO_HAS_EWAH
        if (esz > 0) {
            printf(" | %5zu %5.3f %7.1f %7.1f",
                   esz, e_ratio, e_enc_best, e_dec_best);
        } else {
            printf(" | %5s %5s %7s %7s", "-", "-", "-", "-");
        }
#endif
        printf("\n");
        fflush(stdout);
        (void)f_note;

        free_tables();
    }
    return 0;
}
