/* pivco_prims_table.c — one backend's pivco_prims_t.  Compiled once per
 * backend and per codec variant, with the same defines as that codec. */

#include "pivco_huffman.h"
#include "pivco_huffman_common.h"
#include "pivco_huffman_primitives.h"
#include "pivco_prims.h"

#define PIVCO_PRIMS_STR_(x) #x
#define PIVCO_PRIMS_STR(x)  PIVCO_PRIMS_STR_(x)

const pivco_prims_t PIVCO_BACKEND_FN(pivco_prims) = {
    .name                = PIVCO_PRIMS_STR(PIVCO_BACKEND_SUFFIX),
    .codec_init          = prim_codec_init,
    .enc_init            = prim_enc_init,
    .enc_partition_full  = prim_enc_partition_full,
    .enc_partition_right = prim_enc_partition_right,
    .enc_partition_none  = prim_enc_partition_none,
    .enc_pack_dN         = prim_enc_pack_dN,
    .enc_pack_dN_natural = prim_enc_pack_dN_natural,
    .popcount_K_right    = prim_popcount_K_right,
    .merge_cst_cst       = prim_merge_cst_cst,
    .merge_cst_vec       = prim_merge_cst_vec,
    .merge_vec_vec       = prim_merge_vec_vec,
    .merge_flat          = prim_merge_flat,
    .merge_flat_natural  = prim_merge_flat_natural,
    .histogram_chunk     = prim_histogram_chunk,
};
