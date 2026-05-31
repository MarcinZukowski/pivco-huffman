/* pivco_huffman_neon_tables.c — runtime construction of shared NEON tables.
 *
 * Storage + init functions extracted from the legacy NEON .c files so
 * both the legacy code and the new codec.c-compiled-as-NEON object
 * library link against one copy.  See pivco_huffman_neon_tables.h for
 * the contract. */

#include "pivco_huffman_neon_tables.h"

/* ---------- Encoder partition: compress_tab[256][32] + compress_popcnt[256]
 *
 * For each 8-bit partition mask:
 *   - bytes  0..15 → vqtbl1q indices packing the bit=1 (right) uint16
 *     lanes to the front of the destination register
 *   - bytes 16..31 → indices for the complement (bit=0 → left)
 * Both halves are loaded with a single `ldp q0, q1` (32 bytes,
 * contiguous), one cache-line access instead of two scattered lookups.
 * Lanes past the popcount are filled with 0xFF so vqtbl1q writes
 * arbitrary garbage past `n_{right,left}` — the caller bounds writes
 * via compress_popcnt[mask].
 */
uint8_t compress_tab[256][32]   __attribute__((aligned(32)));
uint8_t compress_popcnt[256]    __attribute__((aligned(64)));
int     compress_table_ready    = 0;

void init_compress_table(void)
{
    if (compress_table_ready) return;
    for (int mask = 0; mask < 256; mask++) {
        /* Right (bit=1): pack selected to front. */
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
            compress_tab[mask][j] = 0xFF;

        /* Left (bit=0): pack complement to front. */
        int out_l = 0;
        for (int i = 0; i < 8; i++) {
            if (!(mask & (1 << i))) {
                compress_tab[mask][16 + out_l * 2]     = (uint8_t)(i * 2);
                compress_tab[mask][16 + out_l * 2 + 1] = (uint8_t)(i * 2 + 1);
                out_l++;
            }
        }
        for (int j = out_l * 2; j < 16; j++)
            compress_tab[mask][16 + j] = 0xFF;
    }
    compress_table_ready = 1;
}

/* ---------- BU merge V4: expand_tab + expand_tab_pre + expand_popcnt
 *
 * `expand_tab[m][k]` is the lane index (0..15) for output position k
 * of an 8-element merge controlled by mask byte m.  Values 0..7 select
 * from the left input; 8..15 select from the right input.
 *
 * `expand_tab_pre[nr0][m1][k]` pre-bakes the iter-1 vqtbl2 indices
 * after iter-0 consumed `nr0` right bytes and `(8 - nr0)` left bytes
 * from a 16-byte L_full and 16-byte R_full source pair.  Without the
 * pre-bake, iter-1's shuf depends on iter-0's nr0 via 4 vector ALU
 * ops on the critical path; the table folds those into one indexed
 * load.  Table size: 9 × 256 × 8 = 18 432 bytes; L1d-resident on
 * every target.
 *
 * Layout per (nr0, m1, k):
 *   L-lane idx = expand_tab[m1][k] + (8 - nr0)   ∈ [(8-nr0)..15]
 *   R-lane idx = expand_tab[m1][k] + 8 + nr0     ∈ [(16+nr0)..(23+nr0)]
 *
 * `expand_popcnt[m]` = popcount(m); the count of right bytes the
 * caller consumes for mask byte m.
 */
uint8_t expand_tab    [256][8]      __attribute__((aligned(32)));
uint8_t expand_tab_pre[9][256][8]   __attribute__((aligned(64)));
uint8_t expand_popcnt [256]         __attribute__((aligned(64)));
int     expand_table_ready          = 0;

void init_expand_table(void)
{
    if (expand_table_ready) return;
    for (int m = 0; m < 256; m++) {
        int n_zeros = 0, n_ones = 0;
        for (int k = 0; k < 8; k++) {
            if (m & (1 << k)) {
                expand_tab[m][k] = (uint8_t)(8 + n_ones);
                n_ones++;
            } else {
                expand_tab[m][k] = (uint8_t)n_zeros;
                n_zeros++;
            }
        }
        expand_popcnt[m] = (uint8_t)n_ones;
    }
    /* Pre-adjusted (nr0, m1) shuf table — see header doc. */
    for (int nr0 = 0; nr0 <= 8; nr0++) {
        for (int m = 0; m < 256; m++) {
            for (int k = 0; k < 8; k++) {
                uint8_t raw = expand_tab[m][k];
                expand_tab_pre[nr0][m][k] =
                    (raw < 8) ? (uint8_t)(raw + (8 - nr0))   /* L-lane */
                              : (uint8_t)(raw + 8 + nr0);    /* R-lane */
            }
        }
    }
    expand_table_ready = 1;
}
