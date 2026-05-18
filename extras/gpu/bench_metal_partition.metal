// bench_metal_partition.metal — Metal compute kernels for measuring the
// ph tree-walk partition primitive throughput on Apple GPU.
//
// The CPU NEON path does this per 16-byte vector via vqtbl1q_u8(compress_tab,...).
// The Metal path does it per 32-thread SIMD group via simd_ballot + popcount-scan.
//
// Kernels:
//   - partition_step:   each SIMD-group of 32 threads reads 32 bytes,
//                       partitions them by the per-symbol bitmap (which
//                       child each byte routes to), writes the 32-byte
//                       group back as [left-bytes, right-bytes].
//   - empty_kernel:     no-op, for measuring command-buffer dispatch
//                       latency in isolation.

#include <metal_stdlib>
using namespace metal;

// Partition one 32-byte block per SIMD-group.
//
// Output layout for each 32-byte block: bytes routing left come first,
// followed by bytes routing right.  This mirrors the CPU compress_tab
// shuffle output exactly.
//
// NOTE on bitmap_u32:
//   This microbench uses a 256-bit table indexed by *byte value*
//   (bit b = "value b goes right").  That's a self-contained stand-in
//   for the real ph primitive, where the routing decision comes from
//   reading the *compressed bitstream* — 1 bit per element, in element
//   order, not 1 bit per symbol value.  The throughput characteristics
//   of the primitive are the same either way (per-call constant
//   overhead vs per-element bit load); the simplification just lets
//   us generate test input without a real ph encoder.
kernel void partition_step(
    device const uchar*  in         [[buffer(0)]],
    device const uint*   bitmap_u32 [[buffer(1)]],   // 8 × u32 = 256-bit bitmap
    device uchar*        out        [[buffer(2)]],
    constant uint&       n_bytes    [[buffer(3)]],
    uint                 tid        [[thread_position_in_grid]],
    uint                 sg_lane    [[thread_index_in_simdgroup]])
{
    if (tid >= n_bytes) return;

    uchar b = in[tid];

    // Look up bitmap bit for this byte's value (0 = go left, 1 = go right).
    bool go_right = ((bitmap_u32[b >> 5] >> (b & 31)) & 1u) != 0u;
    uint go_right_int = go_right ? 1u : 0u;

    // Position within each group via per-lane prefix sums.  Equivalent to
    // popcount-of-mask-below-me but uses simd_prefix_*sum directly so we
    // skip the simd_vote→uint cast that some MSL versions don't support.
    uint right_before = simd_prefix_exclusive_sum(go_right_int);
    uint left_before  = sg_lane - right_before;

    // Total right-going lanes in this SIMD-group → broadcast via simd_sum.
    uint n_right = simd_sum(go_right_int);
    uint n_left  = 32u - n_right;

    uint slot       = go_right ? (n_left + right_before) : left_before;
    uint group_base = tid & ~31u;   // tid / 32 * 32

    out[group_base + slot] = b;
}

// No-op kernel for dispatch-latency timing.  Single thread, single threadgroup.
kernel void empty_kernel(uint tid [[thread_position_in_grid]])
{
    (void)tid;
}

// ============================================================
// tree_merge_step — BU decoder's per-internal-node primitive.
//
// Faithful port of src/pivco_huffman_primitives_neon.h:tree_merge_neon
// (the CPU NEON kernel that ph spends most of its decode time in).
//
// Per output position i:
//   bit = (bm[i >> 3] >> (i & 7)) & 1
//   out[i] = bit ? right[rc++] : left[lc++]
// where lc, rc are sequential cursors over the left/right child byte
// streams.
//
// MICROBENCH SIMPLIFICATION: per-SIMD-group independent cursors.
// Each 32-output-byte SIMD-group resets lc/rc=0 and reads from a
// group-local 32-byte slot in `left` and `right` (group g uses
// left[g*32..g*32+31] and right[g*32..g*32+31]).  This isolates the
// inner per-SIMD-group cost — the real decoder accumulates cursors
// globally across the whole K-byte node, which on GPU is a separate
// global prefix-sum pass not measured here.
//
// Per-lane work:
//   - 1 byte bitmap load (one cache-line read serves all 32 lanes —
//     each lane needs 1 bit of the 4-byte block)
//   - simd_prefix_exclusive_sum: compute my within-group cursor
//   - 1 byte gather from left[] or right[] (within-cache-line —
//     coalesces to one transaction per direction)
//   - 1 byte coalesced store to out[]
kernel void tree_merge_step(
    device const uchar*  bm      [[buffer(0)]],     // K bits packed
    device const uchar*  left    [[buffer(1)]],     // K bytes, group-local layout
    device const uchar*  right   [[buffer(2)]],     // K bytes, group-local layout
    device uchar*        out     [[buffer(3)]],     // K output bytes
    constant uint&       n_out   [[buffer(4)]],
    uint                 tid     [[thread_position_in_grid]],
    uint                 sg_lane [[thread_index_in_simdgroup]])
{
    if (tid >= n_out) return;

    // Read this lane's routing bit (one byte of bitmap covers 8 lanes).
    uchar bit_byte = bm[tid >> 3];
    bool go_right = ((bit_byte >> (tid & 7)) & 1u) != 0u;
    uint go_right_int = go_right ? 1u : 0u;

    // Per-SIMD-group cursor: my position within the right-going group
    // = count of right-going lanes strictly below me.  Left analog
    // is just (sg_lane - right_before).  This is the GPU equivalent
    // of the CPU's expand_tab[mask] indexed shuffle.
    uint right_before = simd_prefix_exclusive_sum(go_right_int);
    uint left_before  = sg_lane - right_before;

    // Group-local slots: group g uses left[g*32..g*32+31] and right[g*32..g*32+31].
    uint group_base = tid & ~31u;

    // Gather the source byte (within-cache-line, coalesces per direction).
    uchar src = go_right ? right[group_base + right_before]
                          : left [group_base + left_before];

    out[tid] = src;
}
