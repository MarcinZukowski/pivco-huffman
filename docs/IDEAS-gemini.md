# Gemini's PIVCO-Huffman Optimizations

> **Last content review:** _NEVER_

This document details the performance improvements implemented for the PIVCO-Huffman decoder, specifically targeting high-entropy distributions on ARM64 (NEON).

## 1. 8-way Parallel Scalar Placement (Phase 4)
**Target:** High-entropy distributions (`english`, `zipfian`, `proba14`) where `pivco_p` (prefix-radix) is used.
**Insight:** The prefix-radix decoder spends a significant portion of its time in Phase 4 (bucketing element IDs by bin). A simple scalar loop `bin_elements[place[v]++] = k` suffers from serial data dependencies on `place[v]`.
**Implementation:** Replaced the 4-way interleaved placement with an 8-way parallel scalar implementation.
**Result:** Improved `english` throughput from ~1075 M/s to **1256 M/s (+16.8%)** on Apple M4.

## 2. SIMD Phase 2 Histogram (Small K)
**Target:** Tables with `min_len` between 1 and 4 bits ($K \in [2, 16]$).
**Insight:** Scalar histogramming `bin_count[prefix[k]]++` is slow due to load-add-store dependencies, especially when symbols cluster.
**Implementation:**
- Used `vceqq_u8` to compare 16 prefix bytes against all possible bin indices in parallel.
- Accumulate matches in 8-bit lanes.
- **Safety:** Implemented a block-based flush every 128 iterations to prevent 8-bit lane overflow (max sum 255).
- Use `vaddl_u8` + `vaddvq_u16` for safe accumulation into 32-bit counters.

## 3. Phase 1 SIMD Extraction (M=4)
**Target:** Distributions with 4-bit minimum code length (`sparse_16`, `flat_M4`).
**Implementation:** Replaced scalar nibble unpacking with NEON `vzipq_u8` to interleave and mask nibbles from a 16-byte vector into 32 output bytes in a single sequence.

## 4. Shared NEON Infrastructure
**Implementation:** Created `pivco_huffman_neon_common.h` to cleanly export `compress_tab` and `compress_popcnt` from `pivco_huffman_neon.c`. This allows the prefix-radix backend to reuse the highly optimized TBL-based partition tables without redundant re-initialization or `extern` pollution.

## Summary of Results (M4 Max)

| Distribution | Original (M/s) | Gemini (M/s) | Improvement |
|--------------|----------------|--------------|-------------|
| **english**  | 1075           | **1256**     | **+16.8%**  |
| **zipfian**  | 735            | **828**      | **+12.6%**  |
| **proba14**  | 1100           | **1232**     | **+12.0%**  |
| **sparse_16**| 5776           | **5738**     | ~Neutral    |
