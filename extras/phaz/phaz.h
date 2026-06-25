/* phaz.h — single-file entropy-layer transplant for zstd ("phaz").
 *
 * pivco-Huffman-ANS + zstd: keep zstd's LZ parse, copy engine, repcodes and
 * window; replace only the *entropy* layer (FSE on sequence codes, HUF on
 * literals) with pivco-Huffman over pivoted streams. This header is the entire
 * in-tree footprint of the fork — it is #included into two zstd TUs so it can
 * reuse their static internals (SeqStore_t / seq_t / ZSTD_execSequence). The
 * accompanying phaz.patch adds only the two #includes + one guarded call.
 *
 *   compress side:  #define PHAZ_COMPRESS_SIDE  before  #include "phaz.h"
 *                   (in lib/compress/zstd_compress.c)
 *   decode side:    #define PHAZ_DECODE_SIDE    before  #include "phaz.h"
 *                   (at end of lib/decompress/zstd_decompress_block.c)
 *
 * The dump (compress) and decode sides are symmetric: this file defines BOTH,
 * so a single patched libzstd carries the capture hook + ZSTD_phazDecode.
 */

/* -------- shared: baseline tables + LSB-first bit IO (both TUs) -------- */
#ifndef PHAZ_SHARED_H
#define PHAZ_SHARED_H

#if defined(__GNUC__)
#  define PHAZ_UNUSED __attribute__((unused))
#else
#  define PHAZ_UNUSED
#endif

/* Copied verbatim from zstd internals (LL_base/LL_bits/ML_base/ML_bits).
 * Offsets need no table: ofCode == highbit32(offBase), base == 1<<ofCode. */
static const unsigned PHAZ_LL_base[36] PHAZ_UNUSED = {
        0,    1,    2,     3,     4,     5,     6,      7,
        8,    9,   10,    11,    12,    13,    14,     15,
       16,   18,   20,    22,    24,    28,    32,     40,
       48,   64, 0x80, 0x100, 0x200, 0x400, 0x800, 0x1000,
       0x2000, 0x4000, 0x8000, 0x10000 };
static const unsigned char PHAZ_LL_bits[36] PHAZ_UNUSED = {
        0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0,
        1, 1, 1, 1, 2, 2, 3, 3,
        4, 6, 7, 8, 9,10,11,12,
       13,14,15,16 };
static const unsigned PHAZ_ML_base[53] PHAZ_UNUSED = {
        3,  4,  5,    6,     7,     8,     9,    10,
       11, 12, 13,   14,    15,    16,    17,    18,
       19, 20, 21,   22,    23,    24,    25,    26,
       27, 28, 29,   30,    31,    32,    33,    34,
       35, 37, 39,   41,    43,    47,    51,    59,
       67, 83, 99, 0x83, 0x103, 0x203, 0x403, 0x803,
       0x1003, 0x2003, 0x4003, 0x8003, 0x10003 };
static const unsigned char PHAZ_ML_bits[53] PHAZ_UNUSED = {
        0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0,
        1, 1, 1, 1, 2, 2, 3, 3,
        4, 4, 5, 7, 8, 9,10,11,
       12,13,14,15,16 };

static int PHAZ_UNUSED phaz_highbit32(unsigned v) {   /* v > 0 */
    return 31 - __builtin_clz(v);
}
#endif /* PHAZ_SHARED_H */


/* ============================== COMPRESS ============================== */
#ifdef PHAZ_COMPRESS_SIDE
#ifndef PHAZ_COMPRESS_DEFINED
#define PHAZ_COMPRESS_DEFINED

/* dump control + captured pivoted streams (read by tools/phaz.c).
 * One byte of ll/ml/of code per sequence; extra bits packed LSB-first into xb;
 * literals concatenated per block; per-block (nbSeq, regen-length) recorded. */
int g_phaz_dump = 0;
unsigned char *g_phaz_llc, *g_phaz_mlc, *g_phaz_ofc, *g_phaz_lit, *g_phaz_xb;
unsigned long long g_phaz_xbpos;            /* bit cursor into g_phaz_xb */
unsigned *g_phaz_blk_ns, *g_phaz_blk_tl;    /* per block: nbSeq, regen length */
size_t g_phaz_nblk, g_phaz_nseq, g_phaz_lits;
unsigned long long g_phaz_extrabits;

static void phaz_putbits(unsigned long long v, int n) {
    unsigned long long pos = g_phaz_xbpos;
    int i;
    if (n <= 0) return;
    for (i = 0; i < n; i++)
        if ((v >> i) & 1ULL) g_phaz_xb[(pos + i) >> 3] |= (unsigned char)(1u << ((pos + i) & 7));
    g_phaz_xbpos += (unsigned long long)n;
    g_phaz_extrabits += (unsigned long long)n;
}

/* Hook: fires once per block in ZSTD_entropyCompressSeqStore_internal, with the
 * block's literals as parameters. Captures codes + extra bits + literals. */
static void phaz_capture(const SeqStore_t* ss, const void* literals, size_t litSize) {
    size_t const nbSeq = (size_t)(ss->sequences - ss->sequencesStart);
    const SeqDef* const seqs = ss->sequencesStart;
    const BYTE *llC, *mlC, *ofC;
    unsigned long long sumML = 0;
    size_t u;

    (void)ZSTD_seqToCodes(ss);   /* (re)fill code tables; idempotent */
    llC = ss->llCode; mlC = ss->mlCode; ofC = ss->ofCode;

    for (u = 0; u < nbSeq; u++) {
        ZSTD_SequenceLength const sl = ZSTD_getSequenceLength(ss, &seqs[u]);
        unsigned const llCode = llC[u], mlCode = mlC[u], ofCode = ofC[u];
        unsigned const offBase = seqs[u].offBase;
        size_t const i = g_phaz_nseq + u;

        g_phaz_llc[i] = (unsigned char)llCode;
        g_phaz_mlc[i] = (unsigned char)mlCode;
        g_phaz_ofc[i] = (unsigned char)ofCode;

        /* litLength: full 32-bit when MaxLL (the only code that can carry the
         * +0x10000 long-length flag), else residual at the field's width. */
        if (llCode == MaxLL) phaz_putbits(sl.litLength, 32);
        else                 phaz_putbits((unsigned long long)sl.litLength - PHAZ_LL_base[llCode], PHAZ_LL_bits[llCode]);
        if (mlCode == MaxML) phaz_putbits(sl.matchLength, 32);
        else                 phaz_putbits((unsigned long long)sl.matchLength - PHAZ_ML_base[mlCode], PHAZ_ML_bits[mlCode]);
        /* offBase = (1<<ofCode) + extra, extra is ofCode bits wide */
        phaz_putbits((unsigned long long)offBase - ((unsigned long long)1 << ofCode), (int)ofCode);

        sumML += sl.matchLength;
    }
    g_phaz_nseq += nbSeq;

    ZSTD_memcpy(g_phaz_lit + g_phaz_lits, literals, litSize);
    g_phaz_lits += litSize;

    g_phaz_blk_ns[g_phaz_nblk] = (unsigned)nbSeq;
    g_phaz_blk_tl[g_phaz_nblk] = (unsigned)(litSize + sumML);   /* regen length = litSize + Sum(matchLen) */
    g_phaz_nblk++;
}

#endif /* PHAZ_COMPRESS_DEFINED */
#endif /* PHAZ_COMPRESS_SIDE */


/* =============================== DECODE =============================== */
#ifdef PHAZ_DECODE_SIDE
#ifndef PHAZ_DECODE_DEFINED
#define PHAZ_DECODE_DEFINED

/* Stateless LSB-first bit reader over the packed extra-bits stream: derive the
 * byte offset straight from the bit cursor, load 8 bytes, shift, mask. Branchless
 * (no accumulator/refill), which is what makes reconstruct cheap. Each call reads
 * 8 bytes at (pos>>3); callers must pad xb with >=8 bytes tail slack. n in [0,32]. */
typedef struct { const unsigned char* xb; unsigned long long pos; } phaz_br;
static void phaz_br_init(phaz_br* r, const unsigned char* xb) { r->xb = xb; r->pos = 0; }
static unsigned long long phaz_br_get(phaz_br* r, int n) {
    unsigned long long chunk, p = r->pos;
    ZSTD_memcpy(&chunk, r->xb + (p >> 3), 8);
    r->pos = p + (unsigned)n;
    return (chunk >> (p & 7)) & ((n >= 64) ? ~0ULL : (((unsigned long long)1 << n) - 1));
}

/* ZSTD_updateRep inlined (ZSTD_REP_NUM==3); real offset is rep[0] afterwards. */
static void phaz_updateRep(U32 rep[3], U32 offBase, U32 ll0) {
    if (offBase > 3) {                 /* full offset */
        rep[2] = rep[1]; rep[1] = rep[0]; rep[0] = offBase - 3;
    } else {                           /* repcode */
        U32 const repCode = offBase - 1 + ll0;
        if (repCode > 0) {
            U32 const cur = (repCode == 3) ? (rep[0] - 1) : rep[repCode];
            rep[2] = (repCode >= 2) ? rep[1] : rep[2];
            rep[1] = rep[0];
            rep[0] = cur;
        }                              /* repCode==0: no change */
    }
}

/* Reconstruct sequences from the pivoted streams and run zstd's own
 * ZSTD_execSequence (repcodes/overlap/window all handled by zstd). Single
 * frame, whole-buffer output, no dictionary. Returns bytes written.
 * Exported (used by tools/phaz.c); prototype silences -Wmissing-prototypes. */
size_t ZSTD_phazDecode(void*, size_t, const unsigned char*, const unsigned char*,
        const unsigned char*, const unsigned char*, const unsigned char*, size_t,
        const unsigned*, const unsigned*, size_t);
size_t ZSTD_phazDecode(void* dst, size_t dstCap,
        const unsigned char* llc, const unsigned char* mlc, const unsigned char* ofc,
        const unsigned char* xb, const unsigned char* lit, size_t litSize,
        const unsigned* blkNs, const unsigned* blkTl, size_t nblk) {
    BYTE* const ostart = (BYTE*)dst;
    BYTE* const oend = ostart + dstCap;
    BYTE* op = ostart;
    const BYTE* litPtr = lit;
    const BYTE* const litEnd = lit + litSize;
    const BYTE* const prefixStart = ostart;
    const BYTE* const vBase = ostart;
    const BYTE* const dictEnd = ostart;
    U32 rep[3];
    phaz_br br;
    size_t b, si = 0;

    rep[0] = repStartValue[0]; rep[1] = repStartValue[1]; rep[2] = repStartValue[2];  /* {1,4,8} */
    phaz_br_init(&br, xb);

    for (b = 0; b < nblk; b++) {
        BYTE* const blockStart = op;
        unsigned const ns = blkNs[b];
        unsigned k;
        for (k = 0; k < ns; k++, si++) {
            unsigned const llCode = llc[si], mlCode = mlc[si], ofCode = ofc[si];
            seq_t seq;
            U32 offBase, ll0;
            size_t oneSeqSize;

            {   int const llBits = PHAZ_LL_bits[llCode], mlBits = PHAZ_ML_bits[mlCode];
                seq.litLength   = (llCode == MaxLL) ? (size_t)phaz_br_get(&br, 32)
                                  : PHAZ_LL_base[llCode] + (llBits ? (size_t)phaz_br_get(&br, llBits) : 0);
                seq.matchLength = (mlCode == MaxML) ? (size_t)phaz_br_get(&br, 32)
                                  : PHAZ_ML_base[mlCode] + (mlBits ? (size_t)phaz_br_get(&br, mlBits) : 0);
            }
            offBase = ((U32)1 << ofCode) + (ofCode ? (U32)phaz_br_get(&br, (int)ofCode) : 0);
            ll0 = (seq.litLength == 0);
            phaz_updateRep(rep, offBase, ll0);
            seq.offset = rep[0];

            oneSeqSize = ZSTD_execSequence(op, oend, seq, &litPtr, litEnd, prefixStart, vBase, dictEnd);
            if (ZSTD_isError(oneSeqSize)) return oneSeqSize;
            op += oneSeqSize;
        }
        /* trailing literals: regen length minus what the sequences produced */
        {   size_t const produced = (size_t)(op - blockStart);
            size_t const trailing = (size_t)blkTl[b] - produced;
            ZSTD_memcpy(op, litPtr, trailing);
            op += trailing; litPtr += trailing;
        }
    }
    return (size_t)(op - ostart);
}

#endif /* PHAZ_DECODE_DEFINED */
#endif /* PHAZ_DECODE_SIDE */
