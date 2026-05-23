# Fair-bench megagrid — decode MB/s (prebuilt; (op)=opaque-only engine)
# 1MB, best 5x10, table-G=128KB.  SHA a463ecf, 2026-05-22.
# hosts: m4=Apple M4 | c8i=Xeon GraniteRapids | c8a=EPYC Zen5 | c8g=Graviton4 | c6a=EPYC Zen3
# note: oodle present on m4 only; AVX-512 TD n/a on c6a (Zen3, no HW); ph BLK=4096 on x86, 8192 on arm.

## proba80
engine             m4       c8i       c8a       c8g       c6a
ph              17833     41271     27599      8778      9038
pha              4250      3138      5353      2613      2839
td_naive         1269       353       405       349       326
td_scl_opt       1677       838       835       655       814
td_nv_simd       3686      2759      3556      1935       n/a
td_simdopt       7005      8758      9601      4380       n/a
huf0             2885      1996      3499      1963      1714
huf0_stk     2868(op)  1944(op)  3340(op)  1939(op)  1710(op)
fse_stk           685       547       747       489       542
fse_x8y1          n/a       n/a       n/a       n/a       n/a
oo_huff      3216(op)         -         -         -         -
oo_tans      2703(op)         -         -         -         -

## prose_pride
engine             m4       c8i       c8a       c8g       c6a
ph               5178      6910      8753      2484      1670
pha              5135      6863      8755      2473      1675
td_naive          439        53        61        60        54
td_scl_opt        601        74        84        85        76
td_nv_simd       1794      1379      1517       764       n/a
td_simdopt       2556      2633      2685      1175       n/a
huf0             2721      1886      3256      1867      1573
huf0_stk     2549(op)  1791(op)  3044(op)  1783(op)  1513(op)
fse_stk           868       531      1046       582       696
fse_x8y1         2090      1397      1938      1258      1074
oo_huff      3178(op)         -         -         -         -
oo_tans      2469(op)         -         -         -         -
