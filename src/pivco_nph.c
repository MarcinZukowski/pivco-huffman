/* Nested PH: the code-length record and the inner block's configuration
 * (see pivco_nph.h).  The inner block itself is coded by the codec. */

#include "pivco_nph.h"

#include <string.h>

/* prim_merge_cst_vec of the backend pivco_decode dispatches to (see
 * PIVCO_BACKEND_FN), defined in pivco_huffman_codec.c.  right_buf is
 * read up to 16 B past the values the merge consumes. */
void PIVCO_BACKEND_FN(pivco_merge_cst_vec)(const uint8_t *bm, int K,
                                           uint8_t left_sym,
                                           const uint8_t *right_buf,
                                           uint8_t *out);

size_t pivco_nph_write_lens(const uint8_t lens[PIVCO_MAX_SYMBOLS], uint8_t *out)
{
    int k = 0;
    for (int s = 0; s < PIVCO_MAX_SYMBOLS; s++)
        k += lens[s] != 0;
    size_t presence_bytes = PIVCO_MAX_SYMBOLS / 8 + (size_t)(k + 1) / 2;
    if (presence_bytes >= PIVCO_MAX_SYMBOLS / 2) {
        out[0] = PIVCO_NPH_LENS_FULL;
        for (int i = 0; i < PIVCO_MAX_SYMBOLS / 2; i++)
            out[1 + i] = (uint8_t)(lens[2 * i] | (lens[2 * i + 1] << 4));
        return PIVCO_NPH_LENS_MAX_BYTES;
    }
    out[0] = PIVCO_NPH_LENS_PRESENCE;
    uint8_t *presence = out + 1;
    uint8_t *nib = presence + PIVCO_MAX_SYMBOLS / 8;
    memset(out + 1, 0, presence_bytes);
    int j = 0;
    for (int s = 0; s < PIVCO_MAX_SYMBOLS; s++) {
        if (!lens[s]) continue;
        presence[s >> 3] |= (uint8_t)(1u << (s & 7));
        nib[j >> 1] |= (uint8_t)(lens[s] << ((j & 1) * 4));
        j++;
    }
    return 1 + presence_bytes;
}

size_t pivco_nph_read_lens(const uint8_t *in, size_t avail,
                           uint8_t lens[PIVCO_MAX_SYMBOLS])
{
    if (avail < 1) return 0;
    /* Every length is checked against 1..PIVCO_MAX_CODE_LEN (0 too, in
     * the full form, where it means absent) by OR-ing a flag per
     * length, and testing it once at the end. */
    unsigned bad = 0;
    if (in[0] == PIVCO_NPH_LENS_FULL) {
        if (avail < PIVCO_NPH_LENS_MAX_BYTES) return 0;
        for (int i = 0; i < PIVCO_MAX_SYMBOLS / 2; i++) {
            uint8_t lo = in[1 + i] & 0xF;
            uint8_t hi = in[1 + i] >> 4;
            bad |= (lo > PIVCO_MAX_CODE_LEN) | (hi > PIVCO_MAX_CODE_LEN);
            lens[2 * i]     = lo;
            lens[2 * i + 1] = hi;
        }
        return bad ? 0 : PIVCO_NPH_LENS_MAX_BYTES;
    }
    if (in[0] != PIVCO_NPH_LENS_PRESENCE) return 0;
    if (avail < 1 + PIVCO_MAX_SYMBOLS / 8) return 0;
    const uint8_t *presence = in + 1;
    const uint8_t *nib = presence + PIVCO_MAX_SYMBOLS / 8;
    uint64_t words[PIVCO_MAX_SYMBOLS / 64];
    memcpy(words, presence, sizeof(words));
    int k = 0;
    for (int w = 0; w < PIVCO_MAX_SYMBOLS / 64; w++)
        k += __builtin_popcountll(words[w]);
    size_t len = 1 + PIVCO_MAX_SYMBOLS / 8 + (size_t)(k + 1) / 2;
    if (avail < len) return 0;
    /* Unpack the k nibbles to bytes and check them; unsigned l - 1
     * wraps for l == 0, so one compare checks both ends of the range.
     * Then expand: the j-th present symbol takes the j-th length, the
     * rest 0 -- a merge of the constant 0 with vals under the presence
     * bitmap.  vals has the merge's 16 B of read slack. */
    uint8_t vals[PIVCO_MAX_SYMBOLS + 16];
    for (int i = 0; i < (k + 1) / 2; i++) {
        vals[2 * i]     = nib[i] & 0xF;
        vals[2 * i + 1] = nib[i] >> 4;
    }
    for (int j = 0; j < k; j++)
        bad |= (unsigned)(vals[j] - 1) >= (unsigned)PIVCO_MAX_CODE_LEN;
    PIVCO_BACKEND_FN(pivco_merge_cst_vec)(presence, PIVCO_MAX_SYMBOLS, 0, vals, lens);
    return bad ? 0 : len;
}

void pivco_nph_inner_cfg(pivco_cfg_t *cfg, int flat_layout)
{
    *cfg = pivco_cfg_default;
    cfg->fse_enabled = 0;
    cfg->krt = PIVCO_NPH_KRT;
    cfg->flat_layout = (pivco_flat_layout_t)flat_layout;
}
