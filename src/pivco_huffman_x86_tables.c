/* pivco_huffman_x86_tables.c — runtime construction of shared x86 tables.
 *
 * Storage + init functions extracted from the legacy x86 .c files so
 * both the legacy code (until step 4 cutover) and the codec.c-compiled-
 * as-x86 object library link against one copy.  See header for the
 * contract. */

#include "pivco_huffman_x86_tables.h"

/* ---------- Encoder partition: compress_tab[256][32] + compress_popcnt[256]
 *
 * For each 8-bit partition mask:
 *   - bytes  0..15 → pshufb indices packing the bit=1 (right) uint16
 *     lanes to the front of the destination register
 *   - bytes 16..31 → indices for the complement (bit=0 → left)
 * Both halves loaded as two aligned 16-byte _mm_load_si128 from
 * contiguous memory.  Lanes past the popcount are set to 0x80 (pshufb
 * zero-fill sentinel) so they write zeros instead of arbitrary data.
 */
uint8_t compress_tab[256][32]   __attribute__((aligned(32)));
uint8_t compress_popcnt[256]    __attribute__((aligned(64)));
int     compress_table_ready    = 0;

void init_compress_table_x86(void)
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
            compress_tab[mask][j] = 0x80;

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
            compress_tab[mask][16 + j] = 0x80;
    }
    compress_table_ready = 1;
}

/* ---------- BU merge: expand_tab[256][8] + expand_popcnt[256]
 *
 * expand_tab[m][k] = lane index (0..15) for output position k of an
 * 8-element merge controlled by mask byte m.  Values 0..7 select from
 * L, 8..15 select from R.  Used as pshufb indices over
 * _mm_unpacklo_epi64(L8_lo, R8_lo) for the 8-byte merge body.
 */
uint8_t expand_tab[256][8]      __attribute__((aligned(32)));
uint8_t expand_popcnt[256]      __attribute__((aligned(64)));
int     expand_table_ready      = 0;

void init_expand_table_x86(void)
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
    expand_table_ready = 1;
}
