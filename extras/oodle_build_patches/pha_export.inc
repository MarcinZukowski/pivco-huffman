/* ====================================================================
 * pha_export.inc  -- pivco-huffman prebuilt-table export shim for Oodle
 *
 * This file is pivco-authored.  It is spliced into the END of the
 * freshly-extracted clone's newlz_arrays_tans.cpp (inside namespace
 * oo2) by inject_export_shim.sh at build time -- it is never committed
 * into the clone and pivco-huffman copies no Oodle source.
 *
 * Oodle's public newlz_get_array_tans fuses three costs into one call:
 * header-parse + decode-table build + decode.  ph's own codec uses
 * pre-built static tables (a table is picked by a 1-byte id, never
 * rebuilt per call), so to compare ph vs Oodle tANS apples-to-apples we
 * need Oodle's decode timed WITHOUT the per-call table build.  These
 * wrappers split the fused call:
 *
 *   pha_tans_build_decoder()   -- parse header + build the decode table
 *                                 into caller memory.  Call ONCE.
 *   pha_tans_decode_prebuilt() -- decode using that table.  Call N times;
 *                                 re-primes only the cheap per-call
 *                                 bitstream state, never the table.
 *
 * The build step is 100% public API (newlz_tans.h: UnPackCounts +
 * Decoder_Size/Init/Fill).  The decode step mirrors the per-call state
 * setup + kernel dispatch from newlz_get_array_tans (it must, since
 * Oodle exposes no decode-with-prebuilt-table entry point).  Because
 * this lives in the same translation unit it can see KrakenTansState,
 * the static tansx2_* dispatchers, and the GetBits_LSB / Refill macros.
 *
 * Serialization: the decode table is a flat newlz_tans_Decoder_Size()
 * block holding exactly one base-relative self-pointer.  To persist it,
 * store the bytes; to restore at a new address, memcpy then call
 * newlz_tans_Decoder_Init() again (re-derives the pointer; leaves the
 * table data untouched).  No wrapper needed -- that is pure public API.
 * ==================================================================== */

extern "C" {

/* Memory (bytes) the caller must provide for a decoder of this L_bits. */
long pha_tans_decoder_size(int L_bits)
{
    return (long) newlz_tans_Decoder_Size(L_bits);
}

/* Parse the tANS header in [comp, comp+comp_len) and build the decode
 * table into decmem (must be >= pha_tans_decoder_size(L_bits)).
 * On success returns 0 and sets:
 *   *L_bits_out      : table L_bits (also needed by decode)
 *   *payload_off_out : byte offset in comp where the tANS payload starts
 *   *payload_len_out : payload length in bytes
 * Returns <0 on parse/build failure (incl. the rare <8-byte payload
 * case, which the bench never hits and we deliberately don't handle). */
int pha_tans_build_decoder(const unsigned char *comp, long comp_len,
                           void *decmem, long decmemsize,
                           int *L_bits_out, long *payload_off_out,
                           long *payload_len_out)
{
    const U8 *comp_ptr = comp;
    const U8 *comp_end = comp + comp_len;
    if (comp_len < 8) return -1;

    rrVarBits_Temps();
    int L_bits;
    newlz_tans_UnpackedCounts counts;

    rrVarBits header_vb;
    rrVarBits_GetOpen(header_vb.m, comp_ptr, comp_end);

    RR_VARBITSTYPE flag = rrVarBits_Get1(header_vb.m);
    if (flag != 0) return -1;

    L_bits = (int) rrVarBits_Get_C(header_vb.m, NEWLZ_TANS_L_BITS_BITS);
    L_bits += NEWLZ_TANS_L_BITS_MIN;
    if (L_bits > NEWLZ_TANS_L_BITS_MAX) return -1;

    if (!newlz_tans_UnPackCounts(&header_vb, L_bits, &counts))
        return -1;

    comp_ptr = rrVarBits_GetEndPtr(header_vb.m);
    if (comp_ptr > comp_end) return -1;
    long payload_len = (long) rrPtrDiff(comp_end - comp_ptr);
    if (payload_len < 8) return -1;   /* skip <8B landing-pad path */

    if ((long) decmemsize < (long) newlz_tans_Decoder_Size(L_bits)) return -1;
    newlz_tans_Decoder *dec = newlz_tans_Decoder_Init(decmem, decmemsize, L_bits);
    if (!dec) return -1;
    newlz_tans_Decoder_Fill(dec, &counts);

    *L_bits_out      = L_bits;
    *payload_off_out = (long) rrPtrDiff(comp_ptr - comp);
    *payload_len_out = payload_len;
    return 0;
}

/* Decode using a table previously built by pha_tans_build_decoder (and
 * possibly memcpy'd + re-Init'd elsewhere).  decmem is the same block
 * passed to the build call.  Returns 1 on success, 0 on failure.
 * Mirrors the per-call setup of newlz_get_array_tans (state priming +
 * kernel dispatch); does NOT rebuild the table. */
int pha_tans_decode_prebuilt(void *decmem, int L_bits,
                             const unsigned char *payload, long payload_len,
                             unsigned char *to, long to_len)
{
    newlz_tans_Decoder *dec = (newlz_tans_Decoder *) decmem;
    const U8 *comp_ptr = payload;
    const U8 *comp_end = payload + payload_len;
    if (payload_len < 8 || to_len < 5) return 0;

    const U8 *lvb1_ptr; UINTr lvb1_bits; int lvb1_bitcnt;
    const U8 *lvb2_ptr; UINTr lvb2_bits; int lvb2_bitcnt;

    lvb1_ptr = comp_ptr;
    lvb1_bits = RR_GETR_LE(lvb1_ptr);
    lvb1_bitcnt = RAD_REGBITS;
    lvb1_ptr += RAD_REGBYTES;

    lvb2_ptr = comp_end - RAD_REGBYTES;
    lvb2_bits = RR_GETR_BE(lvb2_ptr);
    lvb2_bitcnt = RAD_REGBITS;
    lvb2_ptr -= RAD_REGBYTES;

    UINTr x1, x2, x3, x4, x5;
    GetBits_LSB(x1, lvb1, L_bits);
    GetBits_LSB(x2, lvb2, L_bits);
    GetBits_LSB(x3, lvb1, L_bits);
    GetBits_LSB(x4, lvb2, L_bits);
#ifndef __RAD64REGS__
    Refill32_LSB_LE_Forward(lvb1);
#endif
    GetBits_LSB(x5, lvb1, L_bits);

    SINTa dec_len = (SINTa) to_len - 5;

    KrakenTansState s;
    s.decodeptr = to;
    s.decodeend = to + dec_len;
    s.bitp[0] = lvb1_ptr - (lvb1_bitcnt >> 3);
    s.bitp[1] = lvb2_ptr + RAD_REGBYTES + (lvb2_bitcnt >> 3);
    s.bits[0] = (U32) lvb1_bits;
    s.bits[1] = (U32) lvb2_bits;
    s.bitc[0] = lvb1_bitcnt & 7;
    s.bitc[1] = lvb2_bitcnt & 7;
    s.table = dec->decode_table_U8;
    s.tans_state[0] = (U32) x1;
    s.tans_state[1] = (U32) x2;
    s.tans_state[2] = (U32) x3;
    s.tans_state[3] = (U32) x4;
    s.tans_state[4] = (U32) x5;

    bool ok;
#if defined(NEWLZ_JAGUAR_TANS_ASM)
    ok = tansx2_x64_jag_asm(&s);
#elif defined(NEWLZ_X64GENERIC_TANS_ASM)
    if (rrCPUx86_feature_present(RRX86_CPU_BMI2))
    {
        if (rrCPUx86_feature_present(RRX86_CPU_RAPTOR_LAKE))
            ok = tansx2_x64_bmi2_rpl_asm(&s);
        else
            ok = tansx2_x64_bmi2_asm(&s);
    }
    else
        ok = tansx2_x64_asm(&s);
#elif defined(NEWLZ_ARM64_TANS_ASM)
    ok = tansx2_arm64_asm(&s);
#elif defined(__RAD64REGS__)
    ok = tansx2_64(&s);
#else
    ok = tansx2_32(&s);
#endif
    return ok ? 1 : 0;
}

}  /* extern "C" */
