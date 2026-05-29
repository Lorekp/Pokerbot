/**
 * Offline HU showdown equity CLI (OMPEval when vendored, else compile with USE_NUMBA_STUB).
 *
 * Usage:
 *   enumerate_hu_equity h0 h1 v0 v1
 *   enumerate_hu_equity --eval7 c0 c1 c2 c3 c4 c5 c6
 */
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#if defined(USE_OMPEVAL) && USE_OMPEVAL
#include "omp/Hand.h"
#include "omp/HandEvaluator.h"
#endif

static int eval7_category(int c0, int c1, int c2, int c3, int c4, int c5, int c6) {
#if defined(USE_OMPEVAL) && USE_OMPEVAL
    omp::HandEvaluator evaluator;
    omp::Hand hand = omp::Hand::empty();
    hand += omp::Hand(c0);
    hand += omp::Hand(c1);
    hand += omp::Hand(c2);
    hand += omp::Hand(c3);
    hand += omp::Hand(c4);
    hand += omp::Hand(c5);
    hand += omp::Hand(c6);
    return static_cast<int>(evaluator.evaluate(hand));
#else
    (void)c0;
    (void)c1;
    (void)c2;
    (void)c3;
    (void)c4;
    (void)c5;
    (void)c6;
    std::fprintf(stderr, "Rebuild with USE_OMPEVAL=1 and vendored OMPEval\n");
    std::exit(2);
#endif
}

static double hu_equity(int h0, int h1, int v0, int v1) {
    bool dead[52] = {};
    dead[h0] = dead[h1] = dead[v0] = dead[v1] = true;
    int deck[48];
    int nd = 0;
    for (int c = 0; c < 52; ++c) {
        if (!dead[c]) deck[nd++] = c;
    }
    double wins = 0.0, total = 0.0;
    for (int i0 = 0; i0 < nd - 4; ++i0) {
        int c0 = deck[i0];
        for (int i1 = i0 + 1; i1 < nd - 3; ++i1) {
            int c1 = deck[i1];
            for (int i2 = i1 + 1; i2 < nd - 2; ++i2) {
                int c2 = deck[i2];
                for (int i3 = i2 + 1; i3 < nd - 1; ++i3) {
                    int c3 = deck[i3];
                    for (int i4 = i3 + 1; i4 < nd; ++i4) {
                        int c4 = deck[i4];
                        int hr = eval7_category(h0, h1, c0, c1, c2, c3, c4);
                        int vr = eval7_category(v0, v1, c0, c1, c2, c3, c4);
                        if (hr < vr) wins += 1.0;
                        else if (hr == vr) wins += 0.5;
                        total += 1.0;
                    }
                }
            }
        }
    }
    return total > 0 ? wins / total : 0.5;
}

int main(int argc, char** argv) {
    if (argc < 2) {
        std::fprintf(stderr, "usage: %s h0 h1 v0 v1 | --eval7 c0..c6\n", argv[0]);
        return 1;
    }
    if (std::string(argv[1]) == "--eval7") {
        if (argc != 9) {
            std::fprintf(stderr, "--eval7 requires 7 card ids\n");
            return 1;
        }
        std::vector<int> c(7);
        for (int i = 0; i < 7; ++i) c[i] = std::atoi(argv[2 + i]);
        std::printf("%d\n", eval7_category(c[0], c[1], c[2], c[3], c[4], c[5], c[6]));
        return 0;
    }
    if (argc != 5) {
        std::fprintf(stderr, "usage: %s h0 h1 v0 v1\n", argv[0]);
        return 1;
    }
    int h0 = std::atoi(argv[1]);
    int h1 = std::atoi(argv[2]);
    int v0 = std::atoi(argv[3]);
    int v1 = std::atoi(argv[4]);
    std::printf("%.17g\n", hu_equity(h0, h1, v0, v1));
    return 0;
}
