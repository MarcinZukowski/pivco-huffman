/* Isolate FSE_buildDTable cost at tableLog 7 vs 10 (16-symbol alphabet). */
#define FSE_STATIC_LINKING_ONLY
#include "fse.h"
#include <stdio.h>
#include <time.h>
#include <string.h>
static double now(void){ struct timespec t; clock_gettime(CLOCK_MONOTONIC,&t);
    return t.tv_sec + t.tv_nsec*1e-9; }
int main(void){
    /* skewed 16-symbol histogram, id-51-flavoured */
    unsigned cnt[16] = {6000,1200,900,700,500,400,300,250,200,150,120,100,80,60,40,20};
    for (int L = 6; L <= 12; L += 1) {
        if (L==9||L==11) continue;
        short norm[16];
        if (FSE_isError(FSE_normalizeCount(norm, L, cnt, 11020, 15))) return 1;
        FSE_DTable dt[FSE_DTABLE_SIZE_U32(12)];
        int N = 200000;
        /* warm */
        for (int i = 0; i < 1000; i++) FSE_buildDTable(dt, norm, 15, L);
        double best = 1e30;
        for (int r = 0; r < 5; r++) {
            double t = now();
            for (int i = 0; i < N; i++) FSE_buildDTable(dt, norm, 15, L);
            double dt2 = (now() - t) / N;
            if (dt2 < best) best = dt2;
        }
        printf("L=%2d  build %7.0f ns  (table %4d slots, %5d B)\n",
               L, best*1e9, 1<<L, (1<<L)*4);
    }
    return 0;
}
