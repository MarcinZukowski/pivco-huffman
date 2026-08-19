/* Freestanding stub for the wasm32 tree_viz build (no libc).
 * Both stay undefined imports (-ffreestanding implies -fno-builtin,
 * so even ceil doesn't lower to the f64.ceil instruction) and are
 * satisfied by Math.log2 / Math.ceil at instantiation; Math.ceil is
 * IEEE-exact, so only log2 carries any last-ulp fidelity caveat. */
#ifndef TVW_MATH_H
#define TVW_MATH_H
#define INFINITY (__builtin_inff())
double log2(double x);
double ceil(double x);
#endif
