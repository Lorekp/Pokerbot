#include "hu_score2_core.hpp"

#include "omp/Hand.h"

int card_id(int rank, int suit) { return rank * 4 + suit; }

std::array<int, 4> unpack_key(uint64_t key) {
    std::array<int, 4> cards{};
    for (int i = 0; i < 4; ++i) {
        const uint16_t part = static_cast<uint16_t>((key >> (16 * i)) & 0xFFFFu);
        const int rank = (part >> 2) & 0xF;
        const int suit = part & 0x3;
        cards[static_cast<size_t>(i)] = card_id(rank, suit);
    }
    return cards;
}

std::tuple<int, int, int, int> representative_from_key(uint64_t key) {
    const auto c = unpack_key(key);
    return {c[0], c[1], c[2], c[3]};
}

uint32_t score_from_omp_ranks(uint16_t hr, uint16_t vr) {
    if (hr > vr) {
        return 2u;
    }
    if (hr == vr) {
        return 1u;
    }
    return 0u;
}

uint16_t eval7_with(omp::HandEvaluator& evaluator, int c0, int c1, int c2, int c3,
                    int c4, int c5, int c6) {
    omp::Hand hand = omp::Hand::empty() + omp::Hand(c0) + omp::Hand(c1) + omp::Hand(c2) +
                     omp::Hand(c3) + omp::Hand(c4) + omp::Hand(c5) + omp::Hand(c6);
    return evaluator.evaluate(hand);
}

uint16_t eval7(int c0, int c1, int c2, int c3, int c4, int c5, int c6) {
    omp::HandEvaluator evaluator;
    return eval7_with(evaluator, c0, c1, c2, c3, c4, c5, c6);
}

uint32_t enumerate_score2(int h0, int h1, int v0, int v1) {
    bool dead[52] = {};
    dead[h0] = dead[h1] = dead[v0] = dead[v1] = true;
    int deck[48];
    int nd = 0;
    for (int c = 0; c < 52; ++c) {
        if (!dead[c]) {
            deck[nd++] = c;
        }
    }

    omp::HandEvaluator evaluator;
    uint32_t score2 = 0;
    for (int i0 = 0; i0 < nd - 4; ++i0) {
        for (int i1 = i0 + 1; i1 < nd - 3; ++i1) {
            for (int i2 = i1 + 1; i2 < nd - 2; ++i2) {
                for (int i3 = i2 + 1; i3 < nd - 1; ++i3) {
                    for (int i4 = i3 + 1; i4 < nd; ++i4) {
                        const int c0 = deck[i0];
                        const int c1 = deck[i1];
                        const int c2 = deck[i2];
                        const int c3 = deck[i3];
                        const int c4 = deck[i4];
                        const uint16_t hr =
                            eval7_with(evaluator, h0, h1, c0, c1, c2, c3, c4);
                        const uint16_t vr =
                            eval7_with(evaluator, v0, v1, c0, c1, c2, c3, c4);
                        score2 += score_from_omp_ranks(hr, vr);
                    }
                }
            }
        }
    }
    return score2;
}

uint32_t enumerate_score2_one(uint64_t key) {
    const auto c = unpack_key(key);
    return enumerate_score2(c[0], c[1], c[2], c[3]);
}

bool verify_sentinel() {
    return enumerate_score2_one(SENTINEL_KEY_AA_AKO) == SENTINEL_SCORE2_AA_AKO;
}

uint32_t native_outcome(uint16_t hr, uint16_t vr) {
    return score_from_omp_ranks(hr, vr);
}
