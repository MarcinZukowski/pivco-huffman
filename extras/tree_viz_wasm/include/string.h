/* Freestanding stub for the wasm32 tree_viz build (no libc).
 * Implementations live in ../shim.c. */
#ifndef TVW_STRING_H
#define TVW_STRING_H
#include <stddef.h>
void *memcpy(void *dst, const void *src, size_t n);
void *memmove(void *dst, const void *src, size_t n);
void *memset(void *dst, int c, size_t n);
#endif
