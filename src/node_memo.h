/* Node memo: lets shaking price a code without re-coding the nodes it
 * has priced before.
 *
 * Shaking prices every candidate code on the same blocks of one segment.
 * The blocks share the table within a segment, so a node is in all of
 * them or in none, and its record (bitmap or flat region, as the FSE attempt
 * leaves it) depends only on the block's input, the node's symbol sets
 * and, through the static table's commit gate, its depth.  Shaking keeps
 * each node's record bytes summed over the blocks, and tells the codec
 * per candidate which records it already knows.
 *
 * While a memo is attached (pivco_scratch_t.memo), for sched record rec:
 *   known[rec] set    the codec emits nothing for the record.  When bit
 *                     `block` of coded[rec] is set it still marks the
 *                     node FSE-coded in the block's marker prefix, whose
 *                     length depends on which nodes are coded.
 *   known[rec] clear  the codec encodes the record as usual, adds its
 *                     bytes to bytes[rec], and sets bit `block` of
 *                     coded[rec] when the node is FSE-coded.
 * An encode with a memo attached is not decodable: its length leaves out
 * the known records.  Nested blocks (dnPH, snPH) encode on the child
 * encoder, which has no memo. */

#ifndef PIVCO_NODE_MEMO_H
#define PIVCO_NODE_MEMO_H

#include "pivco_huffman.h"

#include <stdint.h>

#define PIVCO_MEMO_MAX_BLOCKS 64   /* bits in coded[] */

typedef struct pivco_node_memo {
    int      block;                           /* the block being encoded */
    uint8_t  known[PIVCO_MAX_SYMBOLS - 1];
    uint32_t bytes[PIVCO_MAX_SYMBOLS - 1];
    uint64_t coded[PIVCO_MAX_SYMBOLS - 1];    /* bit b: FSE-coded in block b */
} pivco_node_memo_t;

#endif
