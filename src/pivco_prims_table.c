/* pivco_prims_table.c — exports one table (prims/tables.h) for the prims
 * codec.  Compiled once per backend and per codec variant, with the same
 * defines as that codec. */

#include "pivco_huffman.h"
#include "pivco_huffman_common.h"
#include "pivco_huffman_primitives.h"
#include "pivco_prims.h"

_Static_assert(PIVCO_PRIM_MERGE_OVERREAD <= PIVCO_PRIMS_MERGE_OVERREAD,
               "the prims codec's buffers must cover this table's merges");

const pivco_prims_t *const PIVCO_BACKEND_FN(pivco_prims) = &prims_tab;
