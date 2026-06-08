// Investigate why early measurements with lemire/counters reported ~2.5x
// higher ns/call than a direct clock_gettime() timer for the same
// partition_8 kernel.  TL;DR: the framework is fine; the gap comes from
// (a) dead-store elimination when test buffers are static arrays, and
// (b) std::vector::data() reloaded every iteration when the lambda
// captures the vector by REFERENCE.
//
// Variant matrix (all use the same compiled partition_8 kernel):
//
//   D1 direct       static arrays, sink^=p8() inside loop (baseline)
//   D2 direct-acc   static arrays, acc+=p8() inside loop, sink=acc after
//   D3 direct-vec   std::vector .data() inside loop, sink^=
//   L1 lemire       std::vector, captured by [&]  - 4 extra ldr/iter
//   L2 lemire-ptr   pointers cached but lambda still [&] - same as L1
//   L3 lemire-xor   like L2, sink^= instead of acc+=
//   L4 lemire-stat  static arrays - dead-store elim deletes the kernel
//   L5 lemire-stat  static arrays + asm("":::"memory") barrier
//   L6 lemire-val   pointers captured BY VALUE [srcp, Lp, Rp, mp]
//                   -- this is the recommended pattern
//
// Indicative numbers on Apple M4 (1024 calls per fn-call):
//   D1/D2 static, kernel survived ~ 0.46 ns/call
//   D3    vector ~ 0.85
//   L1-L3 vec captured by &  ~ 0.86
//   L4-L5 static + barrier   ~ 0.40
//   L6    vec captured by =  ~ 0.57
//
// Compiling at -O3, the inner loops differ:
//   L1 (capture-by-ref vector):
//     ldr x10, [x23]            <-- reload src.data()      *
//     ldr x12, [x19]            <-- reload m.data()         * 4 extra
//     ldr x13, [x22]            <-- reload R.data()         * loads/iter
//     ldr x14, [x21]            <-- reload L.data()        *
//     ldr q0, [x10, x11]
//     ldp q1, q2, [...]
//     tbl/tbl/str/str/...
//   L6 (capture-by-value):
//     ldr q0, [x_const_src, x11]
//     ldp q1, q2, [...]
//     tbl/tbl/str/str/...
//
// Per-iter difference: ~4 cycles, multiplied by ~1B iterations gives the
// ~30% delta.  This matters for any tight kernel benchmarked against
// std::vector buffers; the compiler conservatively treats the lambda
// body as potentially aliasing the vector's _M_start, so it reloads.

#include "counters/bench.h"
#include <arm_neon.h>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <random>
#include <ctime>

extern "C" {
extern uint8_t compress_tab[256][32];
extern uint8_t compress_popcnt[256];
void init_compress_table(void);
}

#define N_GROUPS 1024

static inline int p8(const uint16_t *src, uint8_t mask,
                      uint16_t *L, uint16_t *R) {
    uint8x16_t data = vld1q_u8((const uint8_t *)src);
    const uint8_t *tab = compress_tab[mask];
    uint8x16_t shuf_r = vld1q_u8(tab);
    uint8x16_t shuf_l = vld1q_u8(tab + 16);
    vst1q_u8((uint8_t *)R, vqtbl1q_u8(data, shuf_r));
    vst1q_u8((uint8_t *)L, vqtbl1q_u8(data, shuf_l));
    return compress_popcnt[mask];
}

// Test data: static arrays (addresses known at link time)
alignas(64) static uint16_t s_src[N_GROUPS * 8 + 16];
alignas(64) static uint16_t s_L[N_GROUPS * 8 + 16];
alignas(64) static uint16_t s_R[N_GROUPS * 8 + 16];
alignas(64) static uint8_t  s_m[N_GROUPS];

static double now_ns(void) {
    struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1e9 + ts.tv_nsec;
}

// ---------- Direct timer variants ----------

static double direct_xor_static(int reps) {
    int sink = 0;
    double t0 = now_ns();
    for (int r = 0; r < reps; r++)
        for (int i = 0; i < N_GROUPS; i++)
            sink ^= p8(s_src + i*8, s_m[i], s_L + i*8, s_R + i*8);
    double t = now_ns() - t0;
    asm volatile("" : : "r"(sink) : "memory");
    return t / ((double)reps * N_GROUPS);
}

static double direct_acc_static(int reps) {
    volatile int sink = 0;
    double t0 = now_ns();
    for (int r = 0; r < reps; r++) {
        int acc = 0;
        for (int i = 0; i < N_GROUPS; i++)
            acc += p8(s_src + i*8, s_m[i], s_L + i*8, s_R + i*8);
        sink = acc;
    }
    double t = now_ns() - t0;
    (void)sink;
    return t / ((double)reps * N_GROUPS);
}

static double direct_xor_vec(int reps,
                              const std::vector<uint16_t> &src,
                              std::vector<uint16_t> &L,
                              std::vector<uint16_t> &R,
                              const std::vector<uint8_t> &m) {
    int sink = 0;
    double t0 = now_ns();
    for (int r = 0; r < reps; r++)
        for (int i = 0; i < N_GROUPS; i++)
            sink ^= p8(src.data() + i*8, m[i], L.data() + i*8, R.data() + i*8);
    double t = now_ns() - t0;
    asm volatile("" : : "r"(sink) : "memory");
    return t / ((double)reps * N_GROUPS);
}

// ---------- Lemire variants ----------

static counters::event_aggregate lemire_baseline(
        const std::vector<uint16_t> &src,
        std::vector<uint16_t> &L,
        std::vector<uint16_t> &R,
        const std::vector<uint8_t> &m) {
    volatile int sink = 0;
    auto agg = counters::bench([&]() {
        int acc = 0;
        for (int i = 0; i < N_GROUPS; i++) {
            acc += p8(src.data() + i*8, m[i], L.data() + i*8, R.data() + i*8);
        }
        sink = acc;
    });
    (void)sink;
    return agg;
}

static counters::event_aggregate lemire_cached_ptr(
        const std::vector<uint16_t> &src,
        std::vector<uint16_t> &L,
        std::vector<uint16_t> &R,
        const std::vector<uint8_t> &m) {
    const uint16_t *srcp = src.data();
    uint16_t *Lp = L.data();
    uint16_t *Rp = R.data();
    const uint8_t *mp = m.data();
    volatile int sink = 0;
    auto agg = counters::bench([&]() {
        int acc = 0;
        for (int i = 0; i < N_GROUPS; i++) {
            acc += p8(srcp + i*8, mp[i], Lp + i*8, Rp + i*8);
        }
        sink = acc;
    });
    (void)sink;
    return agg;
}

static counters::event_aggregate lemire_xor_ptr(
        const std::vector<uint16_t> &src,
        std::vector<uint16_t> &L,
        std::vector<uint16_t> &R,
        const std::vector<uint8_t> &m) {
    const uint16_t *srcp = src.data();
    uint16_t *Lp = L.data();
    uint16_t *Rp = R.data();
    const uint8_t *mp = m.data();
    volatile int sink = 0;
    auto agg = counters::bench([&]() {
        int acc = 0;
        for (int i = 0; i < N_GROUPS; i++) {
            acc ^= p8(srcp + i*8, mp[i], Lp + i*8, Rp + i*8);
        }
        sink = acc;
    });
    (void)sink;
    return agg;
}

static counters::event_aggregate lemire_static_arrays() {
    volatile int sink = 0;
    auto agg = counters::bench([&]() {
        int acc = 0;
        for (int i = 0; i < N_GROUPS; i++) {
            acc += p8(s_src + i*8, s_m[i], s_L + i*8, s_R + i*8);
        }
        sink = acc;
    });
    (void)sink;
    return agg;
}

// Capture the buffer pointers by VALUE so they're loop-invariant inside
// the lambda body.  With [&] (the default), the lambda holds REFERENCES
// to local pointer variables, and the optimizer can't prove the lambda
// body doesn't invalidate them — so it reloads them every iteration.
// With [=]-style captures the pointers become loop-invariant copies.
static counters::event_aggregate lemire_capture_by_value(
        const std::vector<uint16_t> &src,
        std::vector<uint16_t> &L,
        std::vector<uint16_t> &R,
        const std::vector<uint8_t> &m) {
    const uint16_t *srcp = src.data();
    uint16_t *Lp = L.data();
    uint16_t *Rp = R.data();
    const uint8_t *mp = m.data();
    volatile int sink = 0;
    auto agg = counters::bench([srcp, Lp, Rp, mp, &sink]() {
        int acc = 0;
        for (int i = 0; i < N_GROUPS; i++) {
            acc += p8(srcp + i*8, mp[i], Lp + i*8, Rp + i*8);
        }
        sink = acc;
    });
    (void)sink;
    return agg;
}

// Same as lemire_static_arrays but asks the compiler to consider L/R as
// "observed" via an inline-asm memory clobber.  This defeats the dead-
// store elimination that flattens lemire_static_arrays' partition_8
// kernel into nothing but compress_popcnt[].
static counters::event_aggregate lemire_static_with_barrier() {
    volatile int sink = 0;
    auto agg = counters::bench([&]() {
        int acc = 0;
        for (int i = 0; i < N_GROUPS; i++) {
            acc += p8(s_src + i*8, s_m[i], s_L + i*8, s_R + i*8);
            // Force the compiler to retain stores to s_L and s_R.
            asm volatile("" : : "r"(s_L), "r"(s_R) : "memory");
        }
        sink = acc;
    });
    (void)sink;
    return agg;
}

int main() {
    init_compress_table();

    // Init both static and vector versions identically.
    std::mt19937 rng(0xCAFEBABE);
    for (int i = 0; i < N_GROUPS * 8; i++) s_src[i] = (uint16_t)rng();
    for (int i = 0; i < N_GROUPS;    i++) s_m[i]   = (uint8_t)rng();

    std::vector<uint16_t> v_src(s_src, s_src + N_GROUPS*8 + 16);
    std::vector<uint16_t> v_L(N_GROUPS*8 + 16);
    std::vector<uint16_t> v_R(N_GROUPS*8 + 16);
    std::vector<uint8_t>  v_m(s_m, s_m + N_GROUPS);

    std::printf("partition_8 microbench harness comparison (n_groups=%d)\n", N_GROUPS);
    std::printf("%-30s %10s\n", "variant", "ns/call");
    std::printf("%-30s %10s\n", "------", "-------");

    // Warm up with both static and vector data.
    for (int i = 0; i < 1000; i++) {
        for (int j = 0; j < N_GROUPS; j++) p8(s_src + j*8, s_m[j], s_L + j*8, s_R + j*8);
        for (int j = 0; j < N_GROUPS; j++)
            p8(v_src.data() + j*8, v_m[j], v_L.data() + j*8, v_R.data() + j*8);
    }

    constexpr int REPS = 100000;
    std::printf("%-30s %10.3f\n", "D1 direct xor static",       direct_xor_static(REPS));
    std::printf("%-30s %10.3f\n", "D2 direct acc static",       direct_acc_static(REPS));
    std::printf("%-30s %10.3f\n", "D3 direct xor vector",       direct_xor_vec(REPS, v_src, v_L, v_R, v_m));

    auto print_lemire = [](const char *name, counters::event_aggregate agg) {
        double ns_call_best = agg.fastest_elapsed_ns() / (double)N_GROUPS;
        double ns_call_avg  = agg.elapsed_ns() / (double)N_GROUPS;
        std::printf("%-30s %10.3f  (avg %.3f, inner_count=%zu)\n",
                    name, ns_call_best, ns_call_avg, (size_t)agg.inner_count);
    };

    print_lemire("L1 lemire baseline (vec)",   lemire_baseline(v_src, v_L, v_R, v_m));
    print_lemire("L2 lemire cached ptr (vec)", lemire_cached_ptr(v_src, v_L, v_R, v_m));
    print_lemire("L3 lemire xor cached ptr",   lemire_xor_ptr(v_src, v_L, v_R, v_m));
    print_lemire("L4 lemire static arrays",    lemire_static_arrays());
    print_lemire("L5 lemire static + barrier",  lemire_static_with_barrier());
    print_lemire("L6 lemire by-value capture",  lemire_capture_by_value(v_src, v_L, v_R, v_m));

    return 0;
}
