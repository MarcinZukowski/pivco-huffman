# PIVCO-Huffman NEON Kernel Walkthroughs

Step-by-step traces of the SIMD micro-kernels that consume most of
PIVCO-Huffman's decode time on Apple M4 (per the
[xctrace profile](README.md#profiling)).  Each kernel section shows:

- the source code (verbatim from
  [`src/pivco_huffman_neon.c`](src/pivco_huffman_neon.c) /
  [`src/pivco_huffman_neon_flat.h`](src/pivco_huffman_neon_flat.h)),
- a worked example with concrete byte values,
- the contents of every NEON register at each step.

Register contents are shown as `[a, b, c, ...]` with values in hex.
"`--`" marks a don't-care lane (written but never read).

The five kernels covered:

| Kernel                 | Profile share | Why it's interesting                                                |
|------------------------|--------------:|---------------------------------------------------------------------|
| `partition_8`          |        37.9 % | The 2-way partition core — TBL + popcount-driven split              |
| `flat_d2_unpack`       |         3.9 % | Simplest D-bit unpacker — 4 bytes → 16 codes via 1 TBL + 1 shift    |
| `flat_d3_unpack`       |         4.1 % | First cross-byte case — works in `uint16` lanes                     |
| `flat_d4_unpack`       |       (~3 %)  | Clean even-D case — 8 bytes → 16 codes via dup + half-byte shift    |
| `scatter_both_leaves`  |         9.9 % | Both-leaves stage fusion — bit-test + blend, no unpack step         |

D=5 and D=6 unpacks follow the same `uint16`-lane pattern as D=3 with
bigger byte counts; see notes at the end.

---

## 1. `partition_8` — 2-way SIMD partition

> Splits 8 input `uint16_t` indices into two output groups based on
> an 8-bit mask: bit=1 → `right_out`, bit=0 → `left_out`, **preserving
> original order within each side**.  This is the inner kernel of the
> recursive tree-walk.

### Source

```c
static inline int partition_8(const uint16_t *src, uint8_t mask,
                               uint16_t *left_out, uint16_t *right_out)
{
    uint8x16_t data = vld1q_u8((const uint8_t *)src);

    /* Load both shuffle patterns with one ldp (32 bytes, contiguous) */
    const uint8_t *tab = compress_tab[mask];
    uint8x16_t shuf_r = vld1q_u8(tab);       /* bytes 0-15: right */
    uint8x16_t shuf_l = vld1q_u8(tab + 16);  /* bytes 16-31: left  */

    uint8x16_t right = vqtbl1q_u8(data, shuf_r);
    uint8x16_t left  = vqtbl1q_u8(data, shuf_l);

    int n_right = compress_popcnt[mask];

    vst1q_u8((uint8_t *)right_out, right);
    vst1q_u8((uint8_t *)left_out, left);

    return n_right;
}
```

### Worked example

```
src   = [0x0010, 0x0020, 0x0030, 0x0040, 0x0050, 0x0060, 0x0070, 0x0080]
mask  = 0b01101001  (= 0x69)

  bit 0 = 1  → src[0] = 0x10  → right
  bit 1 = 0  → src[1] = 0x20  → left
  bit 2 = 0  → src[2] = 0x30  → left
  bit 3 = 1  → src[3] = 0x40  → right
  bit 4 = 0  → src[4] = 0x50  → left
  bit 5 = 1  → src[5] = 0x60  → right
  bit 6 = 1  → src[6] = 0x70  → right
  bit 7 = 0  → src[7] = 0x80  → left
```

**Expected:**
`right = [0x10, 0x40, 0x60, 0x70]`,
`left  = [0x20, 0x30, 0x50, 0x80]`,
`n_right = 4`.

### Step-by-step

**Step 1: load 8 indices as 16 bytes**

```c
uint8x16_t data = vld1q_u8((const uint8_t *)src);
```

```
data = [10,00, 20,00, 30,00, 40,00, 50,00, 60,00, 70,00, 80,00]
        \---/  \---/  \---/  \---/  \---/  \---/  \---/  \---/
       src[0] src[1] src[2] src[3] src[4] src[5] src[6] src[7]
       (each 2-byte slot = one little-endian uint16_t)
```

**Step 2: load the precomputed shuffle pair from `compress_tab[0x69]`**

`compress_tab[256][32]` holds the 32-byte shuffle pair per mask, built
once by `init_compress_table()`.  For `mask = 0x69`, popcount = 4 → 4
right bytes (= 8 bytes = 4 × uint16) and 4 left.

```c
const uint8_t *tab = compress_tab[0x69];
uint8x16_t shuf_r = vld1q_u8(tab);
uint8x16_t shuf_l = vld1q_u8(tab + 16);
```

```
shuf_r = [00,01, 06,07, 10,11, 12,13, 80,80,80,80, 80,80,80,80]
          \---/  \---/  \---/  \---/  \---- "out of range" --------/
          src[0] src[3] src[5] src[6]

shuf_l = [02,03, 04,05, 08,09, 14,15, 80,80,80,80, 80,80,80,80]
          \---/  \---/  \---/  \---/
          src[1] src[2] src[4] src[7]
```

The `0x80` bytes are sentinel "lane out of range" markers — `vqtbl1q_u8`
outputs `0` in any output lane whose index byte is ≥ 16.

**Step 3: TBL shuffle — pack right and left sides**

```c
uint8x16_t right = vqtbl1q_u8(data, shuf_r);
uint8x16_t left  = vqtbl1q_u8(data, shuf_l);
```

```
right = [10,00, 40,00, 60,00, 70,00, 00,00, 00,00, 00,00, 00,00]
       = [0x0010, 0x0040, 0x0060, 0x0070,    --,    --,    --,    --]

left  = [20,00, 30,00, 50,00, 80,00, 00,00, 00,00, 00,00, 00,00]
       = [0x0020, 0x0030, 0x0050, 0x0080,    --,    --,    --,    --]
```

**Step 4: read precomputed popcount, store both halves, return**

```c
int n_right = compress_popcnt[0x69];   /* = 4 */
vst1q_u8((uint8_t *)right_out, right);
vst1q_u8((uint8_t *)left_out, left);
return n_right;
```

The caller reads only the first `n_right` lanes from `right_out` and
`8 - n_right` from `left_out`.  The garbage values in the high lanes
are *written* but never *read*.

### What makes this fast

- **One TBL per side** instead of 8 conditional scalar copies.
- **Precomputed popcount** — no `cnt` instruction in the hot path.
- **Combined shuffle table** — `compress_tab[256][32]` lets both shuf_r
  and shuf_l be loaded with one `ldp q0, q1` (one cache-line read).
- **The bottleneck is the two `vst1q_u8` stores**, not the TBL —
  the M4's store buffer becomes the limit before the SIMD data path
  does (see README "Profiling": these two stores together are 38 % of
  total CPU time).

---

## 2. `flat_d2_unpack` — unpack 16 × 2-bit codes

> Reads 4 packed bytes (= 32 bits = 16 × 2-bit codes) and returns the
> 16 codes as a `uint8x16_t`, one byte per code, value in {0, 1, 2, 3}.

### Source

```c
/* D=2 unpack constants: each byte of input holds 4 codes; replicate
 * each input byte to 4 output lanes, then right-shift lane k by 2k
 * to align the desired 2-bit code at the low bits. */
static const uint8_t flat_d2_dup_tab[16] = {
    0,0,0,0,  1,1,1,1,  2,2,2,2,  3,3,3,3
};
static const int8_t flat_d2_shift_tab[16] = {
    0,-2,-4,-6,  0,-2,-4,-6,  0,-2,-4,-6,  0,-2,-4,-6
};

static inline uint8x16_t flat_d2_unpack(const uint8_t *bm_ptr)
{
    uint32_t packed;
    memcpy(&packed, bm_ptr, 4);
    uint8x16_t bm_lo = vreinterpretq_u8_u32(
        vsetq_lane_u32(packed, vdupq_n_u32(0), 0));
    uint8x16_t dup = vqtbl1q_u8(bm_lo, vld1q_u8(flat_d2_dup_tab));
    uint8x16_t shifted = vshlq_u8(dup, vld1q_s8(flat_d2_shift_tab));
    return vandq_u8(shifted, vdupq_n_u8(0x03));
}
```

### Worked example

```
bm[0..3] = [0xE4, 0x1B, 0xF0, 0x0F]
```

Reading each byte LSB-first (2 bits per code, code 0 at bits[1:0]):

```
0xE4 = 11 10 01 00      bm[0]: codes [0, 1, 2, 3]
0x1B = 00 01 10 11      bm[1]: codes [3, 2, 1, 0]
0xF0 = 11 11 00 00      bm[2]: codes [0, 0, 3, 3]
0x0F = 00 00 11 11      bm[3]: codes [3, 3, 0, 0]
```

**Expected:** `[0,1,2,3, 3,2,1,0, 0,0,3,3, 3,3,0,0]`.

### Step-by-step

**Step 1: load 4 bytes into the low 32 bits of a vector register**

```c
uint32_t packed;
memcpy(&packed, bm_ptr, 4);                                /* one ldr w */
uint8x16_t bm_lo = vreinterpretq_u8_u32(
    vsetq_lane_u32(packed, vdupq_n_u32(0), 0));
```

```
bm_lo = [E4, 1B, F0, 0F, 00, 00, 00, 00, 00, 00, 00, 00, 00, 00, 00, 00]
         \---- the 4 input bytes ----/  \------- zero pad -------/
```

**Step 2: replicate each input byte 4 times across 16 lanes**

```c
uint8x16_t dup = vqtbl1q_u8(bm_lo, vld1q_u8(flat_d2_dup_tab));
```

`flat_d2_dup_tab = [0,0,0,0, 1,1,1,1, 2,2,2,2, 3,3,3,3]` — TBL indices
that pull byte i to output lanes 4i..4i+3.

```
dup = [E4,E4,E4,E4, 1B,1B,1B,1B, F0,F0,F0,F0, 0F,0F,0F,0F]
```

**Step 3: per-lane right-shift to align each code at the low bits**

```c
uint8x16_t shifted = vshlq_u8(dup, vld1q_s8(flat_d2_shift_tab));
```

`vshlq_u8` with **negative** shift amounts is a per-lane *right* shift.
Each of the 4 lanes per input byte gets a different shift {0, -2, -4, -6}.

```
flat_d2_shift_tab = [0,-2,-4,-6, 0,-2,-4,-6, 0,-2,-4,-6, 0,-2,-4,-6]

Lane-by-lane for input 0xE4 = 11100100:
  0xE4 >> 0  = 11100100 = 0xE4   (low 2 bits = 00 = code 0)
  0xE4 >> 2  = 00111001 = 0x39   (low 2 bits = 01 = code 1)
  0xE4 >> 4  = 00001110 = 0x0E   (low 2 bits = 10 = code 2)
  0xE4 >> 6  = 00000011 = 0x03   (low 2 bits = 11 = code 3)

shifted = [E4,39,0E,03, 1B,06,01,00, F0,3C,0F,03, 0F,03,00,00]
```

**Step 4: mask off everything except the low 2 bits per lane**

```c
return vandq_u8(shifted, vdupq_n_u8(0x03));
```

```
result = [00,01,02,03, 03,02,01,00, 00,00,03,03, 03,03,00,00]
       = [ 0, 1, 2, 3,  3, 2, 1, 0,  0, 0, 3, 3,  3, 3, 0, 0]   ✓
```

### What makes this fast

- **One TBL replicates 4 bytes to 16 lanes** — saves 16 scalar shifts
  + register moves.
- **Per-lane variable shift via `vshlq_u8`** — NEON's signed shift
  count lets a single instruction realise any per-lane shift.  Crucial
  for D=2; without it you'd need 4 separate `vshrq_n_u8`s + blends.
- **The caller pairs the result with `vqtbl1q_u8(c2s_vec, codes)`** to
  look up symbols in 1 cycle — for D ≤ 4 the c2s table fits in one
  16-byte register.

---

## 3. `flat_d3_unpack` — unpack 8 × 3-bit codes (cross-byte)

> 3-bit codes don't divide a byte cleanly.  3 bytes = 24 bits = 8 codes,
> and **5 of those 8 codes cross a byte boundary** (code 0 starts at bit
> 0, code 1 at bit 3 — fine — but code 2 at bit 6 spans bytes 0/1, and
> so on).  Workaround: process in `uint16` lanes (each holding a 16-bit
> window with enough bits to shift out any one 3-bit code).

### Source

```c
static const uint8_t flat_d3_shuf_tab[16] = {
    /* 5 lanes of (b0, b1): for codes 0..4 (shifts 0,3,6,9,12) */
    0, 1,  0, 1,  0, 1,  0, 1,  0, 1,
    /* 3 lanes of (b1, b2): for codes 5..7 (shifts 7,10,13) */
    1, 2,  1, 2,  1, 2
};
static const int16_t flat_d3_shift_tab[8] = {
    0, -3, -6, -9, -12, -7, -10, -13
};

static inline uint8x8_t flat_d3_unpack(const uint8_t *bm_ptr)
{
    /* Load 3 bytes byte-by-byte into a vector with top bytes zero.
     * Avoid a 4-byte read so we don't run past the end of the stream. */
    uint8x16_t bm_lo = vdupq_n_u8(0);
    bm_lo = vsetq_lane_u8(bm_ptr[0], bm_lo, 0);
    bm_lo = vsetq_lane_u8(bm_ptr[1], bm_lo, 1);
    bm_lo = vsetq_lane_u8(bm_ptr[2], bm_lo, 2);
    uint8x16_t shuffled = vqtbl1q_u8(bm_lo, vld1q_u8(flat_d3_shuf_tab));
    uint16x8_t w = vreinterpretq_u16_u8(shuffled);
    uint16x8_t shifted = vshlq_u16(w, vld1q_s16(flat_d3_shift_tab));
    uint16x8_t masked = vandq_u16(shifted, vdupq_n_u16(0x07));
    return vmovn_u16(masked);
}
```

### Worked example

To make the trace easy to verify by hand, pick input bytes so the
output codes are `[1, 2, 3, 4, 5, 6, 7, 0]` — every value of a 3-bit
code, in order.

```
bm[0..2] = [0xD1, 0x58, 0x1F]
```

Working out why:

```
LSB-first 3-bit codes encode value v as v[0], v[1], v[2] in that
position order in the bit stream.  Concatenating codes 1..7,0:
  001 010 011 100 101 110 111 000     (24 bits, LSB of code 0 first)

Bit positions in the stream:
   0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 21 22 23
   1  0  0  0  1  0  1  1  0  0  1  0  1  1  1  0  1  1  1  0  0  0  0  0

Pack into bytes LSB-first (so byte k = bits 8k..8k+7, bit 8k is LSB):
   bm[0] = bits  0..7  → 1 0 0 0 1 0 1 1     hi-to-lo = 1 1 0 1 0 0 0 1 = 0xD1
   bm[1] = bits  8..15 → 0 0 1 0 1 1 1 0     hi-to-lo = 0 1 0 1 1 0 0 0 = 0x58
   bm[2] = bits 16..23 → 1 1 1 0 0 0 0 0     hi-to-lo = 0 0 0 1 1 1 1 1 = 0x1F
```

Code locations:
- Codes 0–4 live entirely in bits 0–14, i.e. inside `bm[0..1]` →
  the unpack reads them out of a uint16 window `(bm[1]<<8) | bm[0]`.
- Codes 5–7 live in bits 15–23, with code 5 straddling `bm[1]/bm[2]` →
  reads from window `(bm[2]<<8) | bm[1]`.

**Expected output:** `[1, 2, 3, 4, 5, 6, 7, 0]`.

### Step-by-step

**Step 1: byte-by-byte load (avoids overrunning the 3-byte region)**

```c
uint8x16_t bm_lo = vdupq_n_u8(0);
bm_lo = vsetq_lane_u8(bm_ptr[0], bm_lo, 0);
bm_lo = vsetq_lane_u8(bm_ptr[1], bm_lo, 1);
bm_lo = vsetq_lane_u8(bm_ptr[2], bm_lo, 2);
```

```
bm_lo = [D1, 58, 1F, 00, 00, 00, 00, 00, 00, 00, 00, 00, 00, 00, 00, 00]
```

**Step 2: TBL gather — pull `(b0,b1)` into the first 5 uint16 lanes,
`(b1,b2)` into the last 3**

```c
uint8x16_t shuffled = vqtbl1q_u8(bm_lo, vld1q_u8(flat_d3_shuf_tab));
uint16x8_t w = vreinterpretq_u16_u8(shuffled);
```

```
flat_d3_shuf_tab = [0,1,  0,1,  0,1,  0,1,  0,1,    1,2,  1,2,  1,2]
                    \--- 5 × (b0,b1) ---/         \-- 3 × (b1,b2) --/

shuffled = [D1,58, D1,58, D1,58, D1,58, D1,58,    58,1F, 58,1F, 58,1F]

As uint16 lanes (little-endian — low byte first):
   lane 0..4 = (b1<<8) | b0 = (0x58<<8) | 0xD1 = 0x58D1   (covers bits  0..15)
   lane 5..7 = (b2<<8) | b1 = (0x1F<<8) | 0x58 = 0x1F58   (covers bits  8..23)

w = [0x58D1, 0x58D1, 0x58D1, 0x58D1, 0x58D1, 0x1F58, 0x1F58, 0x1F58]
```

**Step 3: per-lane right-shift to align each code at the low bits**

```c
uint16x8_t shifted = vshlq_u16(w, vld1q_s16(flat_d3_shift_tab));
```

`vshlq_u16` with negative counts is per-lane right-shift.  The shift
table {0, −3, −6, −9, −12, −7, −10, −13} mixes references: the first
five reference bit 0 of `bm[0]`; the last three reference bit 0 of
`bm[1]` (so lane 5 wants stream bit 15 = window bit 7, hence −7 not
−15).

```
flat_d3_shift_tab = [0, -3, -6, -9, -12, -7, -10, -13]

Lane-by-lane:
  lane 0: 0x58D1 >> 0  = 0x58D1
  lane 1: 0x58D1 >> 3  = 0x0B1A
  lane 2: 0x58D1 >> 6  = 0x0163
  lane 3: 0x58D1 >> 9  = 0x002C
  lane 4: 0x58D1 >> 12 = 0x0005
  lane 5: 0x1F58 >> 7  = 0x003E
  lane 6: 0x1F58 >> 10 = 0x0007
  lane 7: 0x1F58 >> 13 = 0x0000

shifted = [0x58D1, 0x0B1A, 0x0163, 0x002C, 0x0005,
           0x003E, 0x0007, 0x0000]
```

**Step 4: mask to 3 bits, narrow uint16 → uint8**

```c
uint16x8_t masked = vandq_u16(shifted, vdupq_n_u16(0x07));
return vmovn_u16(masked);
```

```
masked = [0x0001, 0x0002, 0x0003, 0x0004, 0x0005,
          0x0006, 0x0007, 0x0000]

vmovn_u16 → uint8x8_t:
result = [01, 02, 03, 04, 05, 06, 07, 00]
       = [ 1,  2,  3,  4,  5,  6,  7,  0]   ✓
```

### What makes this fast

- **One TBL gathers byte-pairs into the right uint16 lanes.**  Without
  this, you'd need eight separate cross-byte loads + shift sequences.
- **`vshlq_u16` with per-lane shift counts** handles all 8 codes in a
  single instruction.
- **`vmovn_u16` narrows 8 × uint16 → 8 × uint8** for free (the codes
  fit in 3 bits, top byte is always zero after the mask).

The whole unpack is **5 NEON ops for 8 codes** — about 0.7 ops/code,
in the ballpark of D=2's 0.4 ops/code.  D=3 has fewer codes per unpack
(8 vs 16) which is why its profile share (4.1 %) is comparable to
D=2's (3.9 %) despite the higher per-iteration cost.

---

## 4. `flat_d4_unpack` — unpack 16 × 4-bit codes (clean)

> 4-bit codes pack 2 to a byte with no cross-byte boundary.  Cleanest of
> the unpacks: 8 input bytes → 16 output codes via dup + half-byte shift.

### Source

```c
static const uint8_t flat_d4_dup_tab[16] = {
    0,0, 1,1, 2,2, 3,3, 4,4, 5,5, 6,6, 7,7
};
static const int8_t flat_d4_shift_tab[16] = {
    0,-4, 0,-4, 0,-4, 0,-4, 0,-4, 0,-4, 0,-4, 0,-4
};

static inline uint8x16_t flat_d4_unpack(const uint8_t *bm_ptr)
{
    uint64_t packed;
    memcpy(&packed, bm_ptr, 8);
    uint8x16_t bm_lo = vreinterpretq_u8_u64(
        vsetq_lane_u64(packed, vdupq_n_u64(0), 0));
    uint8x16_t dup = vqtbl1q_u8(bm_lo, vld1q_u8(flat_d4_dup_tab));
    uint8x16_t shifted = vshlq_u8(dup, vld1q_s8(flat_d4_shift_tab));
    return vandq_u8(shifted, vdupq_n_u8(0x0F));
}
```

### Worked example

```
bm[0..7] = [0x10, 0x32, 0x54, 0x76, 0x98, 0xBA, 0xDC, 0xFE]
```

Each byte is `(hi_nibble << 4) | lo_nibble` and unpacks to `[lo, hi]`:

```
0x10 → [0, 1]    0x32 → [2, 3]    0x54 → [4, 5]    0x76 → [6, 7]
0x98 → [8, 9]    0xBA → [A, B]    0xDC → [C, D]    0xFE → [E, F]
```

**Expected:** `[0,1, 2,3, 4,5, 6,7, 8,9, A,B, C,D, E,F]`.

### Step-by-step

**Step 1: 8-byte load (one `ldr x`, then move to vector lane 0)**

```c
uint64_t packed;
memcpy(&packed, bm_ptr, 8);
uint8x16_t bm_lo = vreinterpretq_u8_u64(
    vsetq_lane_u64(packed, vdupq_n_u64(0), 0));
```

```
bm_lo = [10, 32, 54, 76, 98, BA, DC, FE, 00, 00, 00, 00, 00, 00, 00, 00]
```

**Step 2: replicate each input byte 2 times across 16 lanes**

```c
uint8x16_t dup = vqtbl1q_u8(bm_lo, vld1q_u8(flat_d4_dup_tab));
```

`flat_d4_dup_tab = [0,0, 1,1, 2,2, 3,3, 4,4, 5,5, 6,6, 7,7]`:

```
dup = [10,10, 32,32, 54,54, 76,76, 98,98, BA,BA, DC,DC, FE,FE]
```

**Step 3: per-lane right-shift {0, -4, 0, -4, ...} — even lanes keep
the byte; odd lanes shift down by 4**

```c
uint8x16_t shifted = vshlq_u8(dup, vld1q_s8(flat_d4_shift_tab));
```

```
flat_d4_shift_tab = [0,-4, 0,-4, 0,-4, 0,-4, 0,-4, 0,-4, 0,-4, 0,-4]

For 0x10:                For 0x32:                ...
  0x10 >> 0 = 0x10         0x32 >> 0 = 0x32
  0x10 >> 4 = 0x01         0x32 >> 4 = 0x03

shifted = [10,01, 32,03, 54,05, 76,07, 98,09, BA,0B, DC,0D, FE,0F]
```

**Step 4: mask off the high nibble of each lane**

```c
return vandq_u8(shifted, vdupq_n_u8(0x0F));
```

```
result = [00,01, 02,03, 04,05, 06,07, 08,09, 0A,0B, 0C,0D, 0E,0F]
       = [0,1,  2,3,  4,5,  6,7,  8,9,  A,B,  C,D,  E,F]   ✓
```

### What makes this fast

- **One TBL replicates 8 bytes to 16 lanes**, then **one shift +
  one mask** completes the unpack.  Total: 4 NEON ops for 16 codes
  = 0.25 ops/code, the cheapest of all D values.
- **Same `vqtbl1q_u8(c2s, codes)` + `vst1q_u8(out, syms)` tail** as D=2,
  giving the fastest end-to-end flat-decode rate (52 GB/s on M4 — see
  Key Compute Primitives table in the README).

---

## 5. `scatter_both_leaves` — both children are leaves

> Stage fusion: when both children of a tree node are leaves
> (sym0 if bit=0, sym1 if bit=1), skip the partition entirely.  For
> each input index, emit `sym0` or `sym1` directly based on the
> bitmap.  No unpack, no recursion.  ~10 % of decode time on
> `prose_pride`.

### Source (16-elem unrolled loop body)

```c
uint8x8_t vsym0  = vdup_n_u8(sym0);
uint8x8_t vdelta = vdup_n_u8(sym0 ^ sym1);
static const uint8_t bit_pos_tab[8] = {1,2,4,8,16,32,64,128};
uint8x8_t vbit_pos = vld1_u8(bit_pos_tab);

for (; j + 16 <= n; j += 16) {
    uint8x8_t bits0 = vtst_u8(vdup_n_u8(bm[j >> 3]), vbit_pos);
    uint8x8_t vals0 = veor_u8(vsym0, vand_u8(vdelta, bits0));
    uint8x8_t bits1 = vtst_u8(vdup_n_u8(bm[(j >> 3) + 1]), vbit_pos);
    uint8x8_t vals1 = veor_u8(vsym0, vand_u8(vdelta, bits1));
    uint16x8_t i0 = vld1q_u16(indices + j);
    uint16x8_t i1 = vld1q_u16(indices + j + 8);
    symbols[vgetq_lane_u16(i0, 0)] = vget_lane_u8(vals0, 0);
    symbols[vgetq_lane_u16(i0, 1)] = vget_lane_u8(vals0, 1);
    /* ... 14 more lane stores for i0[2..7] and i1[0..7] ... */
}
```

### Worked example (one 8-element chunk)

```
sym0 = 0x41 ('A')                   sym1 = 0x42 ('B')
bm[0] = 0b 11010010 = 0xD2          → bits:  bit 0 = 0 → 'A'
                                              bit 1 = 1 → 'B'
                                              bit 2 = 0 → 'A'
                                              bit 3 = 0 → 'A'
                                              bit 4 = 1 → 'B'
                                              bit 5 = 0 → 'A'
                                              bit 6 = 1 → 'B'
                                              bit 7 = 1 → 'B'

indices = [100, 101, 102, 103, 104, 105, 106, 107]
```

**Expected:** writes `A, B, A, A, B, A, B, B` to `symbols[100..107]`.

### Step-by-step

**Step 1: broadcast sym0, delta, and the bit-position table**

```c
uint8x8_t vsym0    = vdup_n_u8(0x41);
uint8x8_t vdelta   = vdup_n_u8(0x41 ^ 0x42);   /* = 0x03 */
uint8x8_t vbit_pos = vld1_u8(bit_pos_tab);
```

```
vsym0    = [41, 41, 41, 41, 41, 41, 41, 41]
vdelta   = [03, 03, 03, 03, 03, 03, 03, 03]
vbit_pos = [01, 02, 04, 08, 10, 20, 40, 80]
```

**Step 2: AND-test the bitmap byte against each bit position**

```c
uint8x8_t bits0 = vtst_u8(vdup_n_u8(bm[0]), vbit_pos);
```

`vtst_u8(a, b)` computes `(a & b) != 0` per lane, returning all-ones
(`0xFF`) where the bit was set, else zero.

```
broadcast  = [D2, D2, D2, D2, D2, D2, D2, D2]
vbit_pos   = [01, 02, 04, 08, 10, 20, 40, 80]
AND        = [00, 02, 00, 00, 10, 00, 40, 80]
                ^^      ^^  ^^      ^^  ^^  ^^
              false  true false  true  ...

vtst_u8    = [00, FF, 00, 00, FF, 00, FF, FF]
```

**Step 3: select sym0 or sym1 per lane via XOR-and-mask**

```c
uint8x8_t vals0 = veor_u8(vsym0, vand_u8(vdelta, bits0));
```

`vdelta = sym0 ^ sym1`. Where `bits0[k] = 0xFF`, lane k's masked
delta = `sym0 ^ sym1`, so `vsym0 ^ delta = sym1`.  Where
`bits0[k] = 0x00`, lane k's masked delta = 0, so `vsym0 ^ 0 = sym0`.

```
vand_u8(vdelta, bits0)  = [00, 03, 00, 00, 03, 00, 03, 03]
veor_u8(vsym0, ...)     = [41, 42, 41, 41, 42, 41, 42, 42]
                        =  A   B   A   A   B   A   B   B    ✓
```

**Step 4: indexed scalar stores**

```c
uint16x8_t idx = vld1q_u16(indices + j);    /* = [100, 101, ..., 107] */
symbols[vgetq_lane_u16(idx, 0)] = vget_lane_u8(vals, 0);  /* symbols[100] = A */
symbols[vgetq_lane_u16(idx, 1)] = vget_lane_u8(vals, 1);  /* symbols[101] = B */
/* ... 6 more lane-extract + scalar-store pairs ... */
```

This is the **scatter floor** that bounds many other rows in the
[Key Compute Primitives](README.md#cross-platform-primitive-costs)
table: ~0.14–0.18 ns/elem on M4, ~0.66 on Graviton 4 / Zen 3.

### What makes this fast

- **No unpack, no recursion** — 8 codes processed with 3 NEON ops
  (test + and + xor) plus 8 scalar lane stores.
- **`vtst` + `veor`-blend** is faster than a `vbsl` (bit-select) here
  because we already had `delta` precomputed.  Saves one register.
- **The scalar lane-store tail dominates the cost** at the per-element
  level — same store-port-bound pattern as `partition_8`'s output stores.

---

## D=5 / D=6 — same pattern as D=3, scaled up

The D=5 and D=6 unpacks are mechanically identical to D=3 (uint16-lane
window + per-lane shift) but operate on more bytes per iteration:

| D | Bytes read | Codes/iter | Operations | Special tail (last chunk)            |
|--:|-----------:|-----------:|-----------:|---------------------------------------|
| 3 |          3 |          8 |       ~5   | 3-byte byte-by-byte load              |
| 5 |          5 |          8 |       ~5   | 5-byte memcpy via stack buf           |
| 6 |          6 |          8 |       ~5   | 6-byte memcpy via stack buf           |

The unpack sequences are in
[`src/pivco_huffman_neon_flat.h`](src/pivco_huffman_neon_flat.h);
each is ~10 lines and reads exactly like `flat_d3_unpack` with
different shuffle / shift constants.

The downstream c2s lookup is what changes: D=2/D=3/D=4 use
`vqtbl1q_u8` (16-byte c2s table); D=5 uses `vqtbl2_u8` (32-byte = 2
registers); D=6 uses `vqtbl4_u8` (64-byte = 4 registers).  On
Graviton 4's Neoverse-V2 the multi-register TBL is markedly slower
than M4's — see the [Key Compute Primitives](README.md) D=5/D=6 rows
and IDEAS.md "Graviton 4 NEON D=5/D=6 regression" for the gating.

---

## Reading the assembly

The kernels above all compile to ~10–20 ARM64 instructions with
`-O2`.  To see the actual machine code:

```sh
otool -tvV ./build/pivco_huffman_profile_english | \
    awk '/^_decode_node_neon:/,/^_[a-zA-Z]/' | head -80
```

Or for any specific function (`partition_8`, `flat_d2_unpack`, …),
grep on the symbol name.  The xctrace profile (see
[`extras/profile_m4.sh`](extras/profile_m4.sh)) maps each retired-IP
sample back to source lines so you can see which instructions are
actually retiring under timer fires.
