#ifndef PIVCO_CHECK_H
#define PIVCO_CHECK_H

/* ---------- Internal invariant checks ----------
 *
 * Policy: a violated internal invariant CRASHES, in every build type.
 * It suggests corruption that might not be recoverable, and continuing
 * risks silently wrong output.  Never use assert() directly.
 *
 *   PIVCO_CHECK(cond)        always on, Release included.  The failure
 *                            path is one out-of-line noreturn call, so a
 *                            check costs a single predictable test+branch.
 *   PIVCO_CHECK_DEBUG(cond)  compiled out under NDEBUG -- reserve for
 *                            checks too hot to keep in Release.
 */

__attribute__((noreturn))
void pivco_check_fail(const char *expr, const char *file, int line);

#define PIVCO_CHECK(cond)                                                \
    do {                                                                 \
        if (__builtin_expect(!(cond), 0))                                \
            pivco_check_fail(#cond, __FILE__, __LINE__);                 \
    } while (0)

#ifdef NDEBUG
#define PIVCO_CHECK_DEBUG(cond) ((void)0)
#else
#define PIVCO_CHECK_DEBUG(cond) PIVCO_CHECK(cond)
#endif

#endif /* PIVCO_CHECK_H */
