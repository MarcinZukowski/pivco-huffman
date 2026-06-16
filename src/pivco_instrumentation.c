#include <stdio.h>
#include <stdint.h>
#include "pivco_huffman.h"

static uint64_t node_size_hist[PIVCO_WIRE_MAX_N + 1];
static uint64_t total_elements_in_nodes[PIVCO_WIRE_MAX_N + 1];

void pivco_instrument_node_size(int n) {
    if (n >= 0 && n <= PIVCO_WIRE_MAX_N) {
        node_size_hist[n]++;
        total_elements_in_nodes[n] += n;
    }
}

void pivco_dump_node_size_hist(void) {
    printf("\n--- PIVCO Node Size Histogram ---\n");
    printf("%10s %10s %10s %10s\n", "n", "calls", "elements", "elements%");
    uint64_t total_calls = 0;
    uint64_t total_elems = 0;
    for (int i = 0; i <= PIVCO_WIRE_MAX_N; i++) {
        total_calls += node_size_hist[i];
        total_elems += total_elements_in_nodes[i];
    }
    if (total_elems == 0) return;

    for (int i = 0; i <= PIVCO_WIRE_MAX_N; i++) {
        if (node_size_hist[i] > 0) {
            printf("%10d %10llu %10llu %10.2f%%\n", 
                i, node_size_hist[i], total_elements_in_nodes[i],
                (double)total_elements_in_nodes[i] * 100.0 / total_elems);
        }
    }
    printf("Total calls: %llu, Total elements: %llu\n", total_calls, total_elems);
}
