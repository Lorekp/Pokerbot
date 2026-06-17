#pragma once

#include <array>
#include <cstdint>

namespace trio {

inline uint16_t rank_beats(uint16_t rank, uint16_t best) { return rank > best; }

inline void pick_nway_winners(const uint16_t* ranks, int n, uint16_t* best_out, int* nwin_out) {
    uint16_t best = ranks[0];
    int nwin = 1;
    for (int i = 1; i < n; ++i) {
        const uint16_t r = ranks[i];
        if (rank_beats(r, best)) {
            best = r;
            nwin = 1;
        } else if (r == best) {
            nwin += 1;
        }
    }
    *best_out = best;
    *nwin_out = nwin;
}

inline void seat_shares_from_ranks(const uint16_t* ranks, int n, double* shares_out) {
    uint16_t best = 0;
    int nwin = 0;
    pick_nway_winners(ranks, n, &best, &nwin);
    const double share = 1.0 / static_cast<double>(nwin);
    for (int i = 0; i < n; ++i) {
        shares_out[i] = (ranks[i] == best) ? share : 0.0;
    }
}

inline void pair_equity_counters(uint16_t hr_btn, uint16_t hr_sb, uint16_t hr_bb,
                                 double* btn_bb_w, double* btn_sb_w, double* sb_bb_w) {
    if (hr_btn > hr_bb) {
        *btn_bb_w += 1.0;
    } else if (hr_btn == hr_bb) {
        *btn_bb_w += 0.5;
    }
    if (hr_btn > hr_sb) {
        *btn_sb_w += 1.0;
    } else if (hr_btn == hr_sb) {
        *btn_sb_w += 0.5;
    }
    if (hr_sb > hr_bb) {
        *sb_bb_w += 1.0;
    } else if (hr_sb == hr_bb) {
        *sb_bb_w += 0.5;
    }
}

}  // namespace trio
