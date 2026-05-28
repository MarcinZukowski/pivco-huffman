/* bench_ewah_wrapper — C-callable shim around Daniel Lemire's EWAH
 * library (https://github.com/lemire/EWAHBoolArray, vendored at
 * ext/ewah).  Used by extras/bench/bench_golomb.c to add an EWAH column
 * to the Rice vs FSE microbench.
 *
 * Two entry points, both matching the signatures of the Rice
 * encoder/decoder in bench_golomb.c:
 *
 *   ewah_encode(bm, n_bits, majority_bit, out, out_cap) -> bytes
 *   ewah_decode(enc, enc_len, bm_out, n_bits)           -> 0 on ok
 *
 * The minority-bit selection mirrors Rice's: we set the minority
 * bits in EWAH (which is RLE-tuned for the rare "set" case), and
 * the decoder fills the destination with the majority bit then
 * overlays the EWAH iterator's set-bit positions.
 *
 * Uses uword=uint64_t which matches EWAH's "modern hardware"
 * default and is the better choice on 64-bit ARM/x86.  Saves the
 * payload size (no "save size in bits" prefix) because we already
 * know n_bits at decode time from the bench harness.
 */

#include "ewah/ewah.h"

#include <cstring>
#include <cstdint>
#include <cstddef>

using ewah::EWAHBoolArray;

extern "C" {

size_t ewah_encode(const uint8_t *bm, size_t n_bits,
                    int majority_bit,
                    uint8_t *out, size_t out_cap)
{
    int minority_bit = majority_bit ^ 1;
    EWAHBoolArray<uint64_t> arr;
    /* Set minority-bit positions in increasing order — EWAH's set()
     * contract requires monotonic non-decreasing index. */
    for (size_t i = 0; i < n_bits; i++) {
        int b = (bm[i >> 3] >> (i & 7)) & 1;
        if (b == minority_bit) arr.set(i);
    }
    /* Pad to n_bits so the final word boundary is correct. */
    arr.padWithZeroes(n_bits);

    size_t sz = arr.sizeOnDisk(/*savesizeinbits=*/false);
    if (sz > out_cap) return 0;
    size_t written = arr.write(reinterpret_cast<char *>(out),
                                out_cap, /*savesizeinbits=*/false);
    return written;
}

int ewah_decode(const uint8_t *enc, size_t enc_len,
                 uint8_t *bm_out, size_t n_bits,
                 int majority_bit)
{
    int minority_bit = majority_bit ^ 1;
    /* Prefill destination with the majority bit. */
    std::memset(bm_out, majority_bit ? 0xFF : 0x00, (n_bits + 7) >> 3);
    size_t tail = n_bits & 7;
    if (tail && majority_bit) {
        size_t last = (n_bits + 7) >> 3;
        if (last > 0)
            bm_out[last - 1] &= static_cast<uint8_t>((1U << tail) - 1);
    }

    EWAHBoolArray<uint64_t> arr;
    size_t read_bytes = arr.read(
        reinterpret_cast<const char *>(enc),
        enc_len, /*savesizeinbits=*/false);
    if (read_bytes == 0) return -1;

    /* Iterate set bits (= minority positions) and patch the
     * prefilled destination. */
    for (auto it = arr.begin(); it != arr.end(); ++it) {
        size_t pos = *it;
        if (pos >= n_bits) return -2;
        if (minority_bit)
            bm_out[pos >> 3] |=  static_cast<uint8_t>(1U << (pos & 7));
        else
            bm_out[pos >> 3] &= ~static_cast<uint8_t>(1U << (pos & 7));
    }
    return 0;
}

}  /* extern "C" */
