/* phtd_shim.c -- exports the namespaced TD table size so consumers can
 * allocate a table without depending on ph-td's (forked, host-varying)
 * struct layout.  Compiled with -include phtd_names.h, so the type below
 * resolves to phtd_table_t. */
#include "pivco_huffman.h"
#include <stddef.h>

size_t phtd_table_size(void) { return sizeof(pivco_huffman_table_t); }
