/* Freestanding runtime for the wasm32 tree_viz build: the handful of
 * libc symbols joint_lengths.c + huffman_table.c need, plus the two
 * library externs normally provided by pivco_huffman.c.  Compiled with
 * -fno-builtin so the mem functions don't lower into calls to
 * themselves. */
#include <stddef.h>
#include <stdint.h>
#include "pivco_huffman.h"

void *memcpy(void *dst, const void *src, size_t n)
{
    uint8_t *d = (uint8_t *)dst;
    const uint8_t *s = (const uint8_t *)src;
    for (size_t i = 0; i < n; i++) d[i] = s[i];
    return dst;
}

void *memmove(void *dst, const void *src, size_t n)
{
    uint8_t *d = (uint8_t *)dst;
    const uint8_t *s = (const uint8_t *)src;
    if (d < s) { for (size_t i = 0; i < n; i++) d[i] = s[i]; }
    else       { for (size_t i = n; i-- > 0; ) d[i] = s[i]; }
    return dst;
}

void *memset(void *dst, int c, size_t n)
{
    uint8_t *d = (uint8_t *)dst;
    for (size_t i = 0; i < n; i++) d[i] = (uint8_t)c;
    return dst;
}

/* Bump allocator.  The library does one malloc/free pair per joint
 * call; the arena resets on every viz entry point (viz.c), so
 * fragmentation can't accumulate. */
static uint8_t g_heap[16u << 20];
static size_t  g_heap_off;

void tvw_heap_reset(void) { g_heap_off = 0; }

void *malloc(size_t n)
{
    size_t off = (g_heap_off + 15) & ~(size_t)15;
    if (n > sizeof(g_heap) - off) return 0;
    g_heap_off = off + n;
    return g_heap + off;
}

void free(void *p) { (void)p; }

/* Library externs (normally in pivco_huffman.c, which drags in the
 * whole codec -- not wanted here). */
const pivco_cfg_t pivco_cfg_default = {
    .tree_mode   = PIVCO_TREE_MODE_OPTIMIZED,
    .effort      = PIVCO_EFFORT_PLAIN,
    .fse_enabled = 1,
    .flat_layout = PIVCO_FLAT_VERTICAL,
};

__attribute__((noreturn))
void pivco_check_fail(const char *expr, const char *file, int line)
{
    (void)expr; (void)file; (void)line;
    __builtin_trap();   /* surfaces as RuntimeError: unreachable in JS */
}
