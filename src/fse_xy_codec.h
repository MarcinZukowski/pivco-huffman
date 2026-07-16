/* fse_xy_codec.h -- byte-alphabet multi-cursor (x) x unroll (y) FSE
 * encode/decode, factored out of bench_fse_xy_micro.c so the tuning
 * bench can reuse the exact shapes.  Include AFTER:
 *     #define FSE_STATIC_LINKING_ONLY
 *     #include "fse.h"
 *     #include "bitstream.h"
 * Tables (FSE_CTable/FSE_DTable) are built by the caller from the
 * target distribution -- this file is alphabet-agnostic.
 */
#ifndef PIVCO_FSE_XY_CODEC_H
#define PIVCO_FSE_XY_CODEC_H
#include <stdint.h>
#include <stddef.h>

static size_t encode_x(int x, const uint8_t *src, size_t n,
                       void *dst, size_t dst_cap,
                       const FSE_CTable *ct)
{
    if (x < 2 || x > 16) return 0;
    if (n < (size_t)x) return 0;   /* need one symbol per cursor */

    BIT_CStream_t bitC;
    if (FSE_isError(BIT_initCStream(&bitC, dst, dst_cap))) return 0;

    FSE_CState_t st[16];
    /* Any-length form (the old n % x == 0 restriction was a bench-era
     * shortcut that cost every unaligned bitmap the wide path).  The
     * decoder assigns position p to cursor p % x round-robin, including
     * its partial final round, so the encoder walks positions in exact
     * reverse decode order with the SAME mapping: the highest x
     * positions are each cursor's last-decoded symbol and are absorbed
     * into that cursor's initial state (writes no bits); every earlier
     * position is a real encode.  For n % x == 0 this degenerates to
     * the classic per-round order bit-for-bit, so previously-valid
     * streams are unchanged. */
    size_t i = n;
    for (int j = 0; j < x; j++) {
        --i;
        FSE_initCState2(&st[i % (size_t)x], ct, src[i]);
    }

    /* Flush after at most 4 symbols: at tableLog 12 each FSE_encodeSymbol
     * adds up to 12 bits, and a flush leaves up to 7 bits in the 64-bit
     * container, so 7 + 4*12 = 55 <= 64 is safe but 7 + 5*12 = 67 would
     * overflow and silently drop bits (corrupts high-nbBits runs).  This
     * matches FSE's own 4-symbols-between-flushes bound; the decoder's
     * reload cadence is independent (bits are a continuous stream). */
    /* i == n - x here.  Encode the remaining positions [0, i) in strictly
     * decreasing order, cursor = pos % x -- the same operation sequence the
     * flat loop produces, so the emitted bitstream is bit-identical (flush
     * cadence doesn't change the bits).  Structured for speed: peel the
     * (i % x) positions above the nearest x-aligned boundary with the
     * general form, then run fully-unrolled x-wide rounds where the cursor
     * IS the loop counter (no per-symbol modulo, inner loop unrolls at the
     * constant x) -- this is the old aligned fast path, restored. */
    int pk = 0;
    size_t peel = i % (size_t)x;              /* == n % x; 0 for aligned n */
    for (size_t p = 0; p < peel; p++) {
        --i;
        FSE_encodeSymbol(&bitC, &st[i % (size_t)x], src[i]);
        if (++pk == 4 && i > 0) { BIT_flushBitsFast(&bitC); pk = 0; }
    }
    if (pk > 0) BIT_flushBitsFast(&bitC);   /* clean the container before the rounds */

    /* i is now a multiple of x: the rest is the original aligned fast path,
     * unchanged from before the any-length work -- fully-unrolled x-wide
     * rounds (cursor == loop counter, round-local flush counter). */
    while (i > 0) {
        int pushed = 0;
        for (int k = x - 1; k >= 0; k--) {
            FSE_encodeSymbol(&bitC, &st[k], src[--i]);
            pushed++;
            if (pushed == 4 && i > 0) {
                BIT_flushBitsFast(&bitC);
                pushed = 0;
            }
        }
        if (pushed > 0) BIT_flushBitsFast(&bitC);
    }

    for (int k = x - 1; k >= 0; k--) {
        FSE_flushCState(&bitC, &st[k]);
    }
    return BIT_closeCStream(&bitC);
}


/* ============================================================
 *  Per-(x, y) decoder template.
 *
 *  Termination correctness is the tricky bit.  My first attempt
 *  used a fixed iteration count derived from dst_expected; that
 *  over-reads when the bitstream is "tight" (no slack between
 *  bits consumed and bits emitted, common at high skew), and
 *  garbage bits push state to invalid table indices → segfault.
 *
 *  Mirror FSE's reference instead:
 *
 *    1. Main fast loop runs while reload says `unfinished` AND
 *       there's room for x*y output bytes.  BODY decodes x*y
 *       symbols and reloads as needed by its cadence.
 *
 *    2. Tail: per round (x decodes), reload-check BETWEEN
 *       cursors.  When reload returns overflow, the remaining
 *       cursors of the current round decode one symbol each
 *       (post-overflow — they read 0 bits because the symbol is
 *       determined by the current state, and the now-overflowed
 *       bit reader supplies 0 bits to the state transition; the
 *       state ends up garbage but we don't use it again).
 *
 *    3. After the tail breaks, dst_expected % x bytes may still
 *       be left; decode them.
 *
 *  This is what `FSE_decompress_usingDTable_generic` does in
 *  ext/fse/lib/fse_decompress.c.
 *
 *  BODY is a sequence of per-cursor decode macros (D{X}RND for
 *  one round of X decodes) repeated Y times, with reloads
 *  inserted between rounds for x*tableLog > 64.
 * ============================================================ */

#define MK_DECODE_FN(NAME, X, Y, BODY) \
static size_t NAME(const void *src, size_t src_len, \
                    uint8_t *dst, size_t dst_expected, \
                    const FSE_DTable *dt) \
{ \
    BIT_DStream_t bitD; \
    if (FSE_isError(BIT_initDStream(&bitD, src, src_len))) return 0; \
    FSE_DState_t s[16]; \
    for (int k = 0; k < (X); k++) FSE_initDState(&s[k], &bitD, dt); \
    uint8_t *op = dst; \
    uint8_t * const olim = dst + dst_expected; \
    /* Main fast loop. */ \
    while ((BIT_reloadDStream(&bitD) == BIT_DStream_unfinished) \
            & (op + (X) * (Y) <= olim)) { \
        BODY; \
        op += (X) * (Y); \
    } \
    /* Tail: FSE-reference pattern. */ \
    while (op + (X) <= olim) { \
        int overflowed = 0; \
        for (int k = 0; k < (X); k++) { \
            *op++ = FSE_decodeSymbol(&s[k], &bitD); \
            if (BIT_reloadDStream(&bitD) == BIT_DStream_overflow) { \
                for (int kk = k + 1; kk < (X) && op < olim; kk++) \
                    *op++ = FSE_decodeSymbol(&s[kk], &bitD); \
                overflowed = 1; \
                break; \
            } \
        } \
        if (overflowed) break; \
    } \
    /* Partial final round (only fires if dst_expected isn't a \
     * multiple of x; bench sizes are aligned so this is unused \
     * in practice). */ \
    for (int k = 0; k < (X) && op < olim; k++) \
        *op++ = FSE_decodeSymbol(&s[k], &bitD); \
    return op - dst; \
}


/* Per-round decode macros (X decodes, with mid-round reloads
 * inserted for X * PIVCO_FSE_TABLE_LOG > 64 bits).  At our
 * tableLog=12, 5 decodes = 60 bits fits one container; 6+ decodes
 * needs a reload mid-round. */

#define D2RND(base) \
    op[(base)+0] = FSE_decodeSymbolFast(&s[0], &bitD); \
    op[(base)+1] = FSE_decodeSymbolFast(&s[1], &bitD);
#define D4RND(base) \
    op[(base)+0] = FSE_decodeSymbolFast(&s[0], &bitD); \
    op[(base)+1] = FSE_decodeSymbolFast(&s[1], &bitD); \
    op[(base)+2] = FSE_decodeSymbolFast(&s[2], &bitD); \
    op[(base)+3] = FSE_decodeSymbolFast(&s[3], &bitD);
#define D6RND(base) \
    op[(base)+0] = FSE_decodeSymbolFast(&s[0], &bitD); \
    op[(base)+1] = FSE_decodeSymbolFast(&s[1], &bitD); \
    op[(base)+2] = FSE_decodeSymbolFast(&s[2], &bitD); \
    op[(base)+3] = FSE_decodeSymbolFast(&s[3], &bitD); \
    op[(base)+4] = FSE_decodeSymbolFast(&s[4], &bitD); \
    BIT_reloadDStream(&bitD); \
    op[(base)+5] = FSE_decodeSymbolFast(&s[5], &bitD);
#define D8RND(base) \
    op[(base)+0] = FSE_decodeSymbolFast(&s[0], &bitD); \
    op[(base)+1] = FSE_decodeSymbolFast(&s[1], &bitD); \
    op[(base)+2] = FSE_decodeSymbolFast(&s[2], &bitD); \
    op[(base)+3] = FSE_decodeSymbolFast(&s[3], &bitD); \
    BIT_reloadDStream(&bitD); \
    op[(base)+4] = FSE_decodeSymbolFast(&s[4], &bitD); \
    op[(base)+5] = FSE_decodeSymbolFast(&s[5], &bitD); \
    op[(base)+6] = FSE_decodeSymbolFast(&s[6], &bitD); \
    op[(base)+7] = FSE_decodeSymbolFast(&s[7], &bitD);

/* x = 2 family.  At y=1 the main BODY is 2 decodes; at y=2 it's
 * 4 (matches FSE's reference shipping decoder shape exactly); at
 * y=4 it's 8, with one mid-body reload. */
MK_DECODE_FN(decode_x2_y1, 2, 1, D2RND(0))
MK_DECODE_FN(decode_x2_y2, 2, 2,
    D2RND(0)
    D2RND(2))
MK_DECODE_FN(decode_x2_y4, 2, 4,
    D2RND(0)
    D2RND(2)
    BIT_reloadDStream(&bitD);
    D2RND(4)
    D2RND(6))

/* x = 4 family. */
MK_DECODE_FN(decode_x4_y1, 4, 1, D4RND(0))
MK_DECODE_FN(decode_x4_y2, 4, 2,
    D4RND(0)
    BIT_reloadDStream(&bitD);
    D4RND(4))
MK_DECODE_FN(decode_x4_y4, 4, 4,
    D4RND(0)
    BIT_reloadDStream(&bitD);
    D4RND(4)
    BIT_reloadDStream(&bitD);
    D4RND(8)
    BIT_reloadDStream(&bitD);
    D4RND(12))

/* x = 6 family.  D6RND inserts its own mid-round reload after
 * the 5th decode, so no extra reload between rounds. */
MK_DECODE_FN(decode_x6_y1, 6, 1, D6RND(0))
MK_DECODE_FN(decode_x6_y2, 6, 2,
    D6RND(0)
    BIT_reloadDStream(&bitD);
    D6RND(6))
MK_DECODE_FN(decode_x6_y4, 6, 4,
    D6RND(0)
    BIT_reloadDStream(&bitD);
    D6RND(6)
    BIT_reloadDStream(&bitD);
    D6RND(12)
    BIT_reloadDStream(&bitD);
    D6RND(18))

/* x = 8 family.  D8RND inserts a reload after the 4th decode. */
MK_DECODE_FN(decode_x8_y1, 8, 1, D8RND(0))
MK_DECODE_FN(decode_x8_y2, 8, 2,
    D8RND(0)
    BIT_reloadDStream(&bitD);
    D8RND(8))
MK_DECODE_FN(decode_x8_y4, 8, 4,
    D8RND(0)
    BIT_reloadDStream(&bitD);
    D8RND(8)
    BIT_reloadDStream(&bitD);
    D8RND(16)
    BIT_reloadDStream(&bitD);
    D8RND(24))

/* x = 10 / 12 / 16 families.  Per-round macros insert internal
 * reloads to keep ≤ 5 decodes between reload calls (5 × tableLog
 * = 60 bits fits one 64-bit container). */
#define D10RND(base) \
    op[(base)+0] = FSE_decodeSymbolFast(&s[0], &bitD); \
    op[(base)+1] = FSE_decodeSymbolFast(&s[1], &bitD); \
    op[(base)+2] = FSE_decodeSymbolFast(&s[2], &bitD); \
    op[(base)+3] = FSE_decodeSymbolFast(&s[3], &bitD); \
    op[(base)+4] = FSE_decodeSymbolFast(&s[4], &bitD); \
    BIT_reloadDStream(&bitD); \
    op[(base)+5] = FSE_decodeSymbolFast(&s[5], &bitD); \
    op[(base)+6] = FSE_decodeSymbolFast(&s[6], &bitD); \
    op[(base)+7] = FSE_decodeSymbolFast(&s[7], &bitD); \
    op[(base)+8] = FSE_decodeSymbolFast(&s[8], &bitD); \
    op[(base)+9] = FSE_decodeSymbolFast(&s[9], &bitD);

#define D12RND(base) \
    op[(base)+0] = FSE_decodeSymbolFast(&s[0], &bitD); \
    op[(base)+1] = FSE_decodeSymbolFast(&s[1], &bitD); \
    op[(base)+2] = FSE_decodeSymbolFast(&s[2], &bitD); \
    op[(base)+3] = FSE_decodeSymbolFast(&s[3], &bitD); \
    op[(base)+4] = FSE_decodeSymbolFast(&s[4], &bitD); \
    BIT_reloadDStream(&bitD); \
    op[(base)+5] = FSE_decodeSymbolFast(&s[5], &bitD); \
    op[(base)+6] = FSE_decodeSymbolFast(&s[6], &bitD); \
    op[(base)+7] = FSE_decodeSymbolFast(&s[7], &bitD); \
    op[(base)+8] = FSE_decodeSymbolFast(&s[8], &bitD); \
    op[(base)+9] = FSE_decodeSymbolFast(&s[9], &bitD); \
    BIT_reloadDStream(&bitD); \
    op[(base)+10] = FSE_decodeSymbolFast(&s[10], &bitD); \
    op[(base)+11] = FSE_decodeSymbolFast(&s[11], &bitD);

#define D16RND(base) \
    op[(base)+0]  = FSE_decodeSymbolFast(&s[0],  &bitD); \
    op[(base)+1]  = FSE_decodeSymbolFast(&s[1],  &bitD); \
    op[(base)+2]  = FSE_decodeSymbolFast(&s[2],  &bitD); \
    op[(base)+3]  = FSE_decodeSymbolFast(&s[3],  &bitD); \
    op[(base)+4]  = FSE_decodeSymbolFast(&s[4],  &bitD); \
    BIT_reloadDStream(&bitD); \
    op[(base)+5]  = FSE_decodeSymbolFast(&s[5],  &bitD); \
    op[(base)+6]  = FSE_decodeSymbolFast(&s[6],  &bitD); \
    op[(base)+7]  = FSE_decodeSymbolFast(&s[7],  &bitD); \
    op[(base)+8]  = FSE_decodeSymbolFast(&s[8],  &bitD); \
    op[(base)+9]  = FSE_decodeSymbolFast(&s[9],  &bitD); \
    BIT_reloadDStream(&bitD); \
    op[(base)+10] = FSE_decodeSymbolFast(&s[10], &bitD); \
    op[(base)+11] = FSE_decodeSymbolFast(&s[11], &bitD); \
    op[(base)+12] = FSE_decodeSymbolFast(&s[12], &bitD); \
    op[(base)+13] = FSE_decodeSymbolFast(&s[13], &bitD); \
    op[(base)+14] = FSE_decodeSymbolFast(&s[14], &bitD); \
    BIT_reloadDStream(&bitD); \
    op[(base)+15] = FSE_decodeSymbolFast(&s[15], &bitD);

MK_DECODE_FN(decode_x10_y1, 10, 1, D10RND(0))
MK_DECODE_FN(decode_x10_y2, 10, 2,
    D10RND(0)
    BIT_reloadDStream(&bitD);
    D10RND(10))
MK_DECODE_FN(decode_x10_y4, 10, 4,
    D10RND(0)
    BIT_reloadDStream(&bitD);
    D10RND(10)
    BIT_reloadDStream(&bitD);
    D10RND(20)
    BIT_reloadDStream(&bitD);
    D10RND(30))

MK_DECODE_FN(decode_x12_y1, 12, 1, D12RND(0))
MK_DECODE_FN(decode_x12_y2, 12, 2,
    D12RND(0)
    BIT_reloadDStream(&bitD);
    D12RND(12))
MK_DECODE_FN(decode_x12_y4, 12, 4,
    D12RND(0)
    BIT_reloadDStream(&bitD);
    D12RND(12)
    BIT_reloadDStream(&bitD);
    D12RND(24)
    BIT_reloadDStream(&bitD);
    D12RND(36))

MK_DECODE_FN(decode_x16_y1, 16, 1, D16RND(0))
MK_DECODE_FN(decode_x16_y2, 16, 2,
    D16RND(0)
    BIT_reloadDStream(&bitD);
    D16RND(16))
MK_DECODE_FN(decode_x16_y4, 16, 4,
    D16RND(0)
    BIT_reloadDStream(&bitD);
    D16RND(16)
    BIT_reloadDStream(&bitD);
    D16RND(32)
    BIT_reloadDStream(&bitD);
    D16RND(48))


/* ============================================================
 *  Bench driver.
 * ============================================================ */

typedef size_t (*decode_fn_t)(const void *, size_t,
                                uint8_t *, size_t,
                                const FSE_DTable *);

typedef struct { int x, y; const char *name; decode_fn_t fn; } cfg_t;

static const cfg_t cfgs[] = {
    { 2,1,"x2y1", decode_x2_y1 }, { 2,2,"x2y2", decode_x2_y2 }, { 2,4,"x2y4", decode_x2_y4 },
    { 4,1,"x4y1", decode_x4_y1 }, { 4,2,"x4y2", decode_x4_y2 }, { 4,4,"x4y4", decode_x4_y4 },
    { 6,1,"x6y1", decode_x6_y1 }, { 6,2,"x6y2", decode_x6_y2 }, { 6,4,"x6y4", decode_x6_y4 },
    { 8,1,"x8y1", decode_x8_y1 }, { 8,2,"x8y2", decode_x8_y2 }, { 8,4,"x8y4", decode_x8_y4 },
    {10,1,"x10y1",decode_x10_y1},{10,2,"x10y2",decode_x10_y2},{10,4,"x10y4",decode_x10_y4},
    {12,1,"x12y1",decode_x12_y1},{12,2,"x12y2",decode_x12_y2},{12,4,"x12y4",decode_x12_y4},
    {16,1,"x16y1",decode_x16_y1},{16,2,"x16y2",decode_x16_y2},{16,4,"x16y4",decode_x16_y4},
};
#define N_CFGS (sizeof(cfgs)/sizeof(cfgs[0]))

#endif /* PIVCO_FSE_XY_CODEC_H */
