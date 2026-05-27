/* phtd_names.h -- force-included (-include) into every ph-td library TU
 * to namespace its exported symbols + table type to phtd_*, so the
 * historical top-down decoders can link alongside the main pivco_huffman
 * library (bottom-up) in one binary (e.g. bench/bench_fair.c).
 *
 * Covers exactly the global (non-static) symbols ph-td exports -- see
 * `nm libph_td.a`.  Internal/static symbols keep file-local linkage and
 * need no renaming.  The struct type is renamed too so consumers can use
 * a distinct phtd_table_t (ph-td's struct layout differs from main's). */
#ifndef PHTD_NAMES_H
#define PHTD_NAMES_H

#define pivco_huffman_table_t                     phtd_table_t
#define pivco_huffman_build_table                 phtd_build_table
#define pivco_huffman_build_table_naive           phtd_build_table_naive
#define pivco_huffman_build_table_from_code_lens  phtd_build_table_from_code_lens
#define pivco_huffman_encode                      phtd_encode
#define pivco_huffman_decode                      phtd_decode
#define pivco_huffman_encode_scalar_opt           phtd_encode_scalar_opt
#define pivco_huffman_decode_scalar_opt           phtd_decode_scalar_opt
#define pivco_huffman_encode_naive                phtd_encode_naive
#define pivco_huffman_decode_naive                phtd_decode_naive
#define pivco_huffman_encode_neon                 phtd_encode_neon
#define pivco_huffman_decode_neon                 phtd_decode_neon
#define pivco_huffman_decode_naive_simd_neon      phtd_decode_naive_simd_neon
#define pivco_huffman_decode_bu_neon              phtd_decode_bu_neon
#define pivco_huffman_encode_avx512               phtd_encode_avx512
#define pivco_huffman_decode_avx512               phtd_decode_avx512
#define pivco_huffman_decode_naive_simd_avx512    phtd_decode_naive_simd_avx512
#define pivco_huffman_flat_decode_direct_avx512_  phtd_flat_decode_direct_avx512_
#define pivco_huffman_encode_x86                  phtd_encode_x86
#define pivco_huffman_decode_x86                  phtd_decode_x86
#define pivco_huffman_decode_bu_x86               phtd_decode_bu_x86
#define pivco_huffman_set_fse_enabled             phtd_set_fse_enabled
#define pivco_huffman_get_fse_enabled             phtd_get_fse_enabled
#define pivco_huffman_get_impl                    phtd_get_impl
#define pivco_huffman_set_impl                    phtd_set_impl
#define pivco_huffman_flat_decode_direct_neon_    phtd_flat_decode_direct_neon_
#define init_compress_table                       phtd_init_compress_table
#define compress_tab                              phtd_compress_tab
#define compress_popcnt                           phtd_compress_popcnt
#define compress_table_ready                      phtd_compress_table_ready
#define expand_tab                                phtd_expand_tab
#define expand_popcnt                             phtd_expand_popcnt
#define expand_tab_pre                            phtd_expand_tab_pre
#define expand_table_ready                        phtd_expand_table_ready
#define init_expand_table                         phtd_init_expand_table
#define pivco_prof_dump                           phtd_prof_dump
#define pivco_prof_name                           phtd_prof_name
#define pivco_prof_reset                          phtd_prof_reset
#define pivco_prof_probe_tick_freq                phtd_prof_probe_tick_freq

#endif /* PHTD_NAMES_H */
