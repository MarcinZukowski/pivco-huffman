/* ztale: zstd frame anatomy.  Splits a .zst file into literals-section
 * vs sequences-section bytes and histograms the table modes, by
 * parsing the block structure (RFC 8878).
 *
 *   ztale FILE.zst
 *
 * Prints: ZT blocks B lit_raw lit_rle lit_huf lit_treeless (payload
 * bytes incl. their headers+tables) seq_bytes blkhdr fhdr nseq_total
 * modes: per-stream counts of predef/rle/fse/repeat (ll,of,ml packed).
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <sys/stat.h>

static uint32_t rd24(const uint8_t *p) { return p[0] | (p[1] << 8) | ((uint32_t)p[2] << 16); }

int main(int argc, char **argv)
{
    if (argc != 2) { fprintf(stderr, "usage: ztale FILE.zst\n"); return 1; }
    struct stat st;
    if (stat(argv[1], &st)) { perror("stat"); return 1; }
    size_t n = (size_t)st.st_size;
    uint8_t *b = malloc(n);
    FILE *f = fopen(argv[1], "rb");
    if (!f || fread(b, 1, n, f) != n) { perror("read"); return 1; }
    fclose(f);

    size_t p = 0;
    if (n < 4 || rd24(b) != 0x2FB528 || b[3] != 0xFD) { fprintf(stderr, "not zstd\n"); return 1; }
    p = 4;
    uint8_t fhd = b[p++];
    int fcs_field = fhd >> 6, single = (fhd >> 5) & 1, dictid = fhd & 3, cksum = (fhd >> 2) & 1;
    if (!single) p += 1;                       /* window descriptor */
    p += (dictid == 3 ? 4 : dictid);
    static const int fcs_len[4] = {0, 2, 4, 8};
    int fl = fcs_len[fcs_field];
    if (fcs_field == 0 && single) fl = 1;
    p += fl;
    size_t fhdr = p;

    uint64_t blocks = 0, blkhdr = 0, seq_bytes = 0, nseq_total = 0;
    uint64_t lit_bytes[4] = {0,0,0,0};         /* raw, rle, huf, treeless */
    uint64_t modes[3][4]; memset(modes, 0, sizeof(modes));
    int last = 0;
    while (!last && p + 3 <= n) {
        uint32_t bh = rd24(b + p); p += 3; blkhdr += 3;
        last = bh & 1;
        int btype = (bh >> 1) & 3;
        uint32_t bsize = bh >> 3;
        if (btype == 0) { p += bsize; blocks++; continue; }          /* raw block */
        if (btype == 1) { p += 1; blocks++; continue; }              /* RLE block */
        if (btype != 2) { fprintf(stderr, "reserved block\n"); return 1; }
        blocks++;
        size_t bstart = p;
        /* ---- literals section ---- */
        uint8_t h0 = b[p];
        int ltype = h0 & 3, sf = (h0 >> 2) & 3;
        uint64_t lsec = 0;
        if (ltype <= 1) {                       /* raw / RLE literals */
            uint32_t rs;
            if (sf == 0 || sf == 2) { rs = h0 >> 3; p += 1; lsec = 1; }
            else if (sf == 1) { rs = (h0 >> 4) | ((uint32_t)b[p+1] << 4); p += 2; lsec = 2; }
            else { rs = (h0 >> 4) | ((uint32_t)b[p+1] << 4) | ((uint32_t)b[p+2] << 12); p += 3; lsec = 3; }
            if (ltype == 0) { p += rs; lsec += rs; }
            else           { p += 1;  lsec += 1; }
        } else {                                /* compressed / treeless */
            uint32_t cs;
            /* Sizes are bit-packed: header = 3/4/5 B for sf 0,1 / 2 / 3.
             * Layout (LE bitstream after the 4 flag bits):
             *   sf 0,1: 10+10 bits in 3 B; sf 2: 14+14 in 4 B; sf 3: 18+18 in 5 B. */
            uint64_t hv = 0; int hb = (sf <= 1) ? 3 : (sf == 2 ? 4 : 5);
            for (int i = 0; i < hb; i++) hv |= (uint64_t)b[p + i] << (8 * i);
            hv >>= 4;
            int sbits = (sf <= 1) ? 10 : (sf == 2 ? 14 : 18);
            uint32_t rs = (uint32_t)(hv & ((1u << sbits) - 1));
            cs = (uint32_t)((hv >> sbits) & ((1u << sbits) - 1));
            (void)rs;
            p += hb;
            p += cs;
            lsec = (uint64_t)hb + cs;
        }
        lit_bytes[ltype] += lsec;
        /* ---- sequences section ---- */
        size_t seq_start = p;
        uint32_t nseq;
        uint8_t s0 = b[p];
        if (s0 < 128)      { nseq = s0; p += 1; }
        else if (s0 < 255) { nseq = ((uint32_t)(s0 - 128) << 8) + b[p+1]; p += 2; }
        else               { nseq = (uint32_t)b[p+1] + ((uint32_t)b[p+2] << 8) + 0x7F00; p += 3; }
        nseq_total += nseq;
        if (nseq > 0) {
            uint8_t mc = b[p];
            modes[0][(mc >> 6) & 3]++;          /* ll */
            modes[1][(mc >> 4) & 3]++;          /* of */
            modes[2][(mc >> 2) & 3]++;          /* ml */
        }
        seq_bytes += (uint64_t)(bsize - (seq_start - bstart));
        p = bstart + bsize;
    }
    if (cksum) p += 4;
    printf("ZT blocks %llu fhdr %zu blkhdr %llu lit_raw %llu lit_rle %llu"
           " lit_huf %llu lit_tless %llu seq %llu nseq %llu"
           " llm %llu %llu %llu %llu ofm %llu %llu %llu %llu mlm %llu %llu %llu %llu total %zu\n",
           (unsigned long long)blocks, fhdr, (unsigned long long)blkhdr,
           (unsigned long long)lit_bytes[0], (unsigned long long)lit_bytes[1],
           (unsigned long long)lit_bytes[2], (unsigned long long)lit_bytes[3],
           (unsigned long long)seq_bytes, (unsigned long long)nseq_total,
           (unsigned long long)modes[0][0], (unsigned long long)modes[0][1],
           (unsigned long long)modes[0][2], (unsigned long long)modes[0][3],
           (unsigned long long)modes[1][0], (unsigned long long)modes[1][1],
           (unsigned long long)modes[1][2], (unsigned long long)modes[1][3],
           (unsigned long long)modes[2][0], (unsigned long long)modes[2][1],
           (unsigned long long)modes[2][2], (unsigned long long)modes[2][3], n);
    return 0;
}
