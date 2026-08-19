/* Freestanding stub for the wasm32 tree_viz build (no libc).
 * malloc is a bump allocator in ../shim.c (the library makes one
 * malloc/free pair per joint call). */
#ifndef TVW_STDLIB_H
#define TVW_STDLIB_H
#include <stddef.h>
void *malloc(size_t n);
void  free(void *p);
#endif
