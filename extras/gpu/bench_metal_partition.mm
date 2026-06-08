// bench_metal_partition — Apple-GPU steady-state throughput measurement
// for the ph tree-walk partition primitive.
//
// Output:
//   - Apple GPU device name.
//   - Command-buffer dispatch latency (μs) — measured with an empty kernel.
//   - Partition kernel throughput at 10 MB, 1 MB, 100 KB.  Reports min
//     of 5 batches × 100 iters; pipelines all commits in one queue and
//     waits only on the last command buffer per batch (steady-state).
//   - Verification: reads back the output of one run and checks the
//     left-then-right partition invariant per 32-byte block.

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

static double now_seconds() {
    using clock = std::chrono::high_resolution_clock;
    return std::chrono::duration<double>(clock::now().time_since_epoch()).count();
}

// Reference C implementation of partition_step for verification.
static void cpu_reference_partition(const uint8_t *in, uint8_t *out, size_t n,
                                     const uint32_t *bitmap_u32) {
    for (size_t base = 0; base + 32 <= n; base += 32) {
        uint8_t left[32], right[32];
        int nl = 0, nr = 0;
        for (int i = 0; i < 32; i++) {
            uint8_t b = in[base + i];
            bool go_right = ((bitmap_u32[b >> 5] >> (b & 31)) & 1u) != 0u;
            if (go_right) right[nr++] = b;
            else          left[nl++]  = b;
        }
        memcpy(out + base, left, nl);
        memcpy(out + base + nl, right, nr);
    }
}

// Reference C implementation of tree_merge_step (group-local cursors,
// matches the GPU kernel's per-SIMD-group reset behavior).
static void cpu_reference_merge(const uint8_t *bm, const uint8_t *left,
                                 const uint8_t *right, uint8_t *out, size_t n) {
    for (size_t base = 0; base + 32 <= n; base += 32) {
        size_t lc = 0, rc = 0;
        for (int i = 0; i < 32; i++) {
            size_t pos = base + i;
            bool go_right = ((bm[pos >> 3] >> (pos & 7)) & 1u) != 0u;
            out[pos] = go_right ? right[base + rc++]
                                 : left [base + lc++];
        }
    }
}

static double bench_kernel(id<MTLCommandQueue> q,
                           id<MTLComputePipelineState> ps,
                           id<MTLBuffer> in_buf,
                           id<MTLBuffer> bitmap_buf,
                           id<MTLBuffer> out_buf,
                           id<MTLBuffer> n_buf,
                           size_t n_bytes,
                           int iters) {
    @autoreleasepool {
        id<MTLCommandBuffer> last = nil;
        double t0 = now_seconds();
        for (int i = 0; i < iters; i++) {
            id<MTLCommandBuffer> cmd = [q commandBuffer];
            id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
            [enc setComputePipelineState:ps];
            [enc setBuffer:in_buf     offset:0 atIndex:0];
            [enc setBuffer:bitmap_buf offset:0 atIndex:1];
            [enc setBuffer:out_buf    offset:0 atIndex:2];
            [enc setBuffer:n_buf      offset:0 atIndex:3];
            [enc dispatchThreads:MTLSizeMake(n_bytes, 1, 1)
                threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
            [enc endEncoding];
            [cmd commit];
            last = cmd;
        }
        [last waitUntilCompleted];
        double dt = now_seconds() - t0;
        return (double)n_bytes * (double)iters / dt / 1e9;  // GB/s of input
    }
}

// 4-buffer variant for tree_merge_step.
static double bench_kernel_merge(id<MTLCommandQueue> q,
                                  id<MTLComputePipelineState> ps,
                                  id<MTLBuffer> bm_buf,
                                  id<MTLBuffer> left_buf,
                                  id<MTLBuffer> right_buf,
                                  id<MTLBuffer> out_buf,
                                  id<MTLBuffer> n_buf,
                                  size_t n_bytes,
                                  int iters) {
    @autoreleasepool {
        id<MTLCommandBuffer> last = nil;
        double t0 = now_seconds();
        for (int i = 0; i < iters; i++) {
            id<MTLCommandBuffer> cmd = [q commandBuffer];
            id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
            [enc setComputePipelineState:ps];
            [enc setBuffer:bm_buf    offset:0 atIndex:0];
            [enc setBuffer:left_buf  offset:0 atIndex:1];
            [enc setBuffer:right_buf offset:0 atIndex:2];
            [enc setBuffer:out_buf   offset:0 atIndex:3];
            [enc setBuffer:n_buf     offset:0 atIndex:4];
            [enc dispatchThreads:MTLSizeMake(n_bytes, 1, 1)
                threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
            [enc endEncoding];
            [cmd commit];
            last = cmd;
        }
        [last waitUntilCompleted];
        double dt = now_seconds() - t0;
        return (double)n_bytes * (double)iters / dt / 1e9;
    }
}

static double bench_dispatch_latency(id<MTLCommandQueue> q,
                                      id<MTLComputePipelineState> ps_empty,
                                      int iters) {
    @autoreleasepool {
        double t0 = now_seconds();
        for (int i = 0; i < iters; i++) {
            id<MTLCommandBuffer> cmd = [q commandBuffer];
            id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
            [enc setComputePipelineState:ps_empty];
            [enc dispatchThreads:MTLSizeMake(1, 1, 1)
                threadsPerThreadgroup:MTLSizeMake(1, 1, 1)];
            [enc endEncoding];
            [cmd commit];
            [cmd waitUntilCompleted];
        }
        return (now_seconds() - t0) * 1e6 / iters;  // μs / dispatch
    }
}

int main(int argc, const char **argv) {
    (void)argc; (void)argv;
    @autoreleasepool {
        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        if (!device) {
            fprintf(stderr, "no Metal device found\n");
            return 1;
        }
        printf("device: %s\n", [[device name] UTF8String]);

        // Load the shader from extras/gpu/bench_metal_partition.metal
        // (relative to CWD; bench is run from project root).
        NSString *shader_path = @"extras/gpu/bench_metal_partition.metal";
        NSError *err = nil;
        NSString *shader_src = [NSString stringWithContentsOfFile:shader_path
                                                          encoding:NSUTF8StringEncoding
                                                             error:&err];
        if (!shader_src) {
            fprintf(stderr, "cannot read %s: %s\n",
                    [shader_path UTF8String],
                    err ? [[err localizedDescription] UTF8String] : "?");
            return 1;
        }

        MTLCompileOptions *opts = [[MTLCompileOptions alloc] init];
        id<MTLLibrary> library = [device newLibraryWithSource:shader_src
                                                       options:opts
                                                         error:&err];
        if (!library) {
            fprintf(stderr, "shader compile failed: %s\n",
                    err ? [[err localizedDescription] UTF8String] : "?");
            return 1;
        }

        id<MTLFunction> fn_partition = [library newFunctionWithName:@"partition_step"];
        id<MTLFunction> fn_merge     = [library newFunctionWithName:@"tree_merge_step"];
        id<MTLFunction> fn_empty     = [library newFunctionWithName:@"empty_kernel"];
        if (!fn_partition || !fn_merge || !fn_empty) {
            fprintf(stderr, "kernel not found\n");
            return 1;
        }

        id<MTLComputePipelineState> ps_partition =
            [device newComputePipelineStateWithFunction:fn_partition error:&err];
        if (!ps_partition) {
            fprintf(stderr, "pipeline (partition) failed: %s\n",
                    err ? [[err localizedDescription] UTF8String] : "?");
            return 1;
        }
        id<MTLComputePipelineState> ps_merge =
            [device newComputePipelineStateWithFunction:fn_merge error:&err];
        if (!ps_merge) {
            fprintf(stderr, "pipeline (merge) failed: %s\n",
                    err ? [[err localizedDescription] UTF8String] : "?");
            return 1;
        }
        id<MTLComputePipelineState> ps_empty =
            [device newComputePipelineStateWithFunction:fn_empty error:&err];
        if (!ps_empty) {
            fprintf(stderr, "pipeline (empty) failed: %s\n",
                    err ? [[err localizedDescription] UTF8String] : "?");
            return 1;
        }

        id<MTLCommandQueue> q = [device newCommandQueue];

        // Buffers — sized for 100 MB max, shared (unified) so CPU can verify.
        const size_t N_MAX = 100ull * 1024 * 1024;
        id<MTLBuffer> in_buf     = [device newBufferWithLength:N_MAX
                                                        options:MTLResourceStorageModeShared];
        id<MTLBuffer> out_buf    = [device newBufferWithLength:N_MAX
                                                        options:MTLResourceStorageModeShared];
        id<MTLBuffer> bitmap_buf = [device newBufferWithLength:32
                                                        options:MTLResourceStorageModeShared];
        id<MTLBuffer> n_buf      = [device newBufferWithLength:sizeof(uint32_t)
                                                        options:MTLResourceStorageModeShared];

        // Fill input with sequential bytes; bitmap routes value >= 128 right
        // (i.e., low 4 u32 words of bitmap = 0, high 4 = all-ones).
        uint8_t *in_ptr = (uint8_t *)[in_buf contents];
        for (size_t i = 0; i < N_MAX; i++) in_ptr[i] = (uint8_t)(i & 0xff);

        uint32_t *bitmap_ptr = (uint32_t *)[bitmap_buf contents];
        for (int i = 0; i < 4; i++) bitmap_ptr[i] = 0u;
        for (int i = 4; i < 8; i++) bitmap_ptr[i] = 0xFFFFFFFFu;

        // --- Verification at N = 10 MB ---
        const size_t N_VERIFY = N_MAX;
        *((uint32_t *)[n_buf contents]) = (uint32_t)N_VERIFY;
        memset([out_buf contents], 0, N_VERIFY);
        {
            id<MTLCommandBuffer> cmd = [q commandBuffer];
            id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
            [enc setComputePipelineState:ps_partition];
            [enc setBuffer:in_buf     offset:0 atIndex:0];
            [enc setBuffer:bitmap_buf offset:0 atIndex:1];
            [enc setBuffer:out_buf    offset:0 atIndex:2];
            [enc setBuffer:n_buf      offset:0 atIndex:3];
            [enc dispatchThreads:MTLSizeMake(N_VERIFY, 1, 1)
                threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
            [enc endEncoding];
            [cmd commit];
            [cmd waitUntilCompleted];
        }
        std::vector<uint8_t> ref(N_VERIFY);
        cpu_reference_partition(in_ptr, ref.data(), N_VERIFY, bitmap_ptr);
        const uint8_t *gpu_out = (const uint8_t *)[out_buf contents];
        size_t mism = 0, first_mism = (size_t)-1;
        for (size_t i = 0; i < N_VERIFY; i++) {
            if (gpu_out[i] != ref[i]) {
                if (first_mism == (size_t)-1) first_mism = i;
                mism++;
            }
        }
        if (mism != 0) {
            fprintf(stderr,
                    "  partition verification FAIL: %zu byte mismatches (first at %zu: gpu=0x%02x ref=0x%02x)\n",
                    mism, first_mism,
                    gpu_out[first_mism], ref[first_mism]);
            return 1;
        }
        printf("partition verification: OK (%.1f MB)\n", (double)N_VERIFY / 1e6);

        // === tree_merge_step: allocate, verify, bench ===
        // Reuse N_MAX-sized buffers for bm/left/right/out_merge.  We allocate
        // dedicated buffers so all 4 inputs are independent (no aliasing).
        id<MTLBuffer> bm_buf    = [device newBufferWithLength:N_MAX
                                                       options:MTLResourceStorageModeShared];
        id<MTLBuffer> left_buf  = [device newBufferWithLength:N_MAX
                                                       options:MTLResourceStorageModeShared];
        id<MTLBuffer> right_buf = [device newBufferWithLength:N_MAX
                                                       options:MTLResourceStorageModeShared];
        id<MTLBuffer> out_merge = [device newBufferWithLength:N_MAX
                                                       options:MTLResourceStorageModeShared];

        uint8_t *bm_ptr    = (uint8_t *)[bm_buf contents];
        uint8_t *left_ptr  = (uint8_t *)[left_buf contents];
        uint8_t *right_ptr = (uint8_t *)[right_buf contents];

        // Fill bitmap with a deterministic ~50/50 pattern.  Avoid 0xAA/0x55
        // (would only test alternating bits); use a PRNG-like fill.
        uint32_t rng = 0x12345678u;
        for (size_t i = 0; i < N_MAX; i++) {
            rng = rng * 1664525u + 1013904223u;
            bm_ptr[i] = (uint8_t)(rng >> 24);
        }
        // Left bytes = 0x10 | (group_id_low_4_bits) ; right bytes = 0x80 |
        // — gives distinct, verifiable per-group content.
        for (size_t i = 0; i < N_MAX; i++) {
            left_ptr[i]  = (uint8_t)(0x10u | (i & 0x0fu));
            right_ptr[i] = (uint8_t)(0x80u | (i & 0x0fu));
        }

        // Verify merge at N_VERIFY.
        *((uint32_t *)[n_buf contents]) = (uint32_t)N_VERIFY;
        memset([out_merge contents], 0, N_VERIFY);
        {
            id<MTLCommandBuffer> cmd = [q commandBuffer];
            id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
            [enc setComputePipelineState:ps_merge];
            [enc setBuffer:bm_buf    offset:0 atIndex:0];
            [enc setBuffer:left_buf  offset:0 atIndex:1];
            [enc setBuffer:right_buf offset:0 atIndex:2];
            [enc setBuffer:out_merge offset:0 atIndex:3];
            [enc setBuffer:n_buf     offset:0 atIndex:4];
            [enc dispatchThreads:MTLSizeMake(N_VERIFY, 1, 1)
                threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
            [enc endEncoding];
            [cmd commit];
            [cmd waitUntilCompleted];
        }
        std::vector<uint8_t> ref_merge(N_VERIFY);
        cpu_reference_merge(bm_ptr, left_ptr, right_ptr, ref_merge.data(), N_VERIFY);
        const uint8_t *gpu_merge_out = (const uint8_t *)[out_merge contents];
        mism = 0;
        first_mism = (size_t)-1;
        for (size_t i = 0; i < N_VERIFY; i++) {
            if (gpu_merge_out[i] != ref_merge[i]) {
                if (first_mism == (size_t)-1) first_mism = i;
                mism++;
            }
        }
        if (mism != 0) {
            fprintf(stderr,
                    "  merge verification FAIL: %zu byte mismatches (first at %zu: gpu=0x%02x ref=0x%02x)\n",
                    mism, first_mism,
                    gpu_merge_out[first_mism], ref_merge[first_mism]);
            return 1;
        }
        printf("merge     verification: OK (%.1f MB)\n", (double)N_VERIFY / 1e6);

        // --- Dispatch latency ---
        const int n_dispatch_iters = 500;
        double us = bench_dispatch_latency(q, ps_empty, n_dispatch_iters);
        printf("dispatch latency: %.2f μs/cmd (mean of %d single-kernel waits)\n",
               us, n_dispatch_iters);

        // --- Steady-state partition throughput at multiple sizes ---
        // iters chosen so each batch processes ~1 GB regardless of N.
        const int n_batches = 5;
        const size_t sizes[] = {
            100 * 1024ull,
            1 * 1024 * 1024ull,
            10 * 1024 * 1024ull,
            100 * 1024 * 1024ull,
        };
        printf("partition throughput (min of %d batches, ~1 GB/batch):\n", n_batches);
        for (size_t s_idx = 0; s_idx < sizeof(sizes) / sizeof(sizes[0]); s_idx++) {
            size_t N = sizes[s_idx];
            int iters = (int)((1ull << 30) / N);
            if (iters < 10) iters = 10;
            *((uint32_t *)[n_buf contents]) = (uint32_t)N;
            double best = 0.0;
            for (int b = 0; b < n_batches; b++) {
                double gb = bench_kernel(q, ps_partition,
                                          in_buf, bitmap_buf, out_buf, n_buf,
                                          N, iters);
                if (gb > best) best = gb;
            }
            printf("  N = %9zu B  (%5d iters)  →  %7.2f GB/s\n", N, iters, best);
        }

        printf("merge     throughput (min of %d batches, ~1 GB/batch):\n", n_batches);
        for (size_t s_idx = 0; s_idx < sizeof(sizes) / sizeof(sizes[0]); s_idx++) {
            size_t N = sizes[s_idx];
            int iters = (int)((1ull << 30) / N);
            if (iters < 10) iters = 10;
            *((uint32_t *)[n_buf contents]) = (uint32_t)N;
            double best = 0.0;
            for (int b = 0; b < n_batches; b++) {
                double gb = bench_kernel_merge(q, ps_merge,
                                                bm_buf, left_buf, right_buf, out_merge, n_buf,
                                                N, iters);
                if (gb > best) best = gb;
            }
            printf("  N = %9zu B  (%5d iters)  →  %7.2f GB/s (output)\n", N, iters, best);
        }

        return 0;
    }
}
