/* pivco_prims_table.c — exports this compile's tables (prims/tables.h)
 * for the prims codec: the one its codec inlines, plus any defined only
 * for the prims codec.  Compiled once per backend and per codec variant,
 * with the same defines as that codec. */

/* Makes prims/tables.h define every table of this compile, not only
 * prims_tab.  The extra tables (prims_tab_<name>) are run only by the
 * prims codec; the compile-time codec TUs leave this unset, so they
 * don't define static tables they never use. */
#define PIVCO_PRIMS_ALL_TABLES 1

#include "pivco_huffman.h"
#include "pivco_huffman_common.h"
#include "pivco_huffman_primitives.h"
#include "pivco_prims.h"

_Static_assert(PIVCO_PRIM_MERGE_OVERREAD <= PIVCO_PRIMS_MERGE_OVERREAD,
               "the prims codec's buffers must cover this table's merges");

const pivco_prims_t *const PIVCO_BACKEND_FN(pivco_prims) = &prims_tab;

#if defined(PIVCO_BACKEND_AVX512)
const pivco_prims_t *const pivco_prims_avx512_zen45 = &prims_tab_avx512_zen45;
#endif
