#include "pivco_huffman.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

/* Forward declarations from test_roundtrip.c */
int test_roundtrip_all(void);

int main(void)
{
    printf("=== PIVCO-Huffman Tests ===\n\n");

    int failures = 0;
    failures += test_roundtrip_all();

    printf("\n=== %s (%d failure%s) ===\n",
           failures == 0 ? "ALL PASSED" : "FAILURES",
           failures, failures == 1 ? "" : "s");

    return failures == 0 ? 0 : 1;
}
