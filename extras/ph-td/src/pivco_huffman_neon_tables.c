/* pivco_huffman_neon_tables.c — TD-slice subset: encoder partition
 * compress table only.  The BU tree_merge expand tables are stripped
 * since BU isn't compiled into this slice.
 */

#include "pivco_huffman_neon_tables.h"

/* ---------- Encoder partition: compress_tab[256][32] + compress_popcnt[256]
 *
 * Per 8-element partition: code_vec → mask byte → compress_tab[mask]
 * gives a 32-byte shuffle table (16 right-lanes + 16 left-lanes) for
 * two vqtbl1q_u8 instructions on the 16-byte input (8 uint16 codes).
 * compress_popcnt[mask] gives the count of right-going lanes for the
 * cursor advance.
 */
uint8_t compress_tab[256][32]   __attribute__((aligned(32)));
uint8_t compress_popcnt[256]    __attribute__((aligned(64)));
int     compress_table_ready    = 0;

void init_compress_table(void)
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
            compress_tab[mask][j] = 0xFF;

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
