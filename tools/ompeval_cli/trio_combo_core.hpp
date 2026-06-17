#pragma once

#include <array>
#include <cstdint>

struct TrioComboMetrics {
    double shares[3];
    double pair_eq[3];  // btn_bb, btn_sb, sb_bb
};

struct TrioAdaptiveResult {
    double shares[3];
    double pair_eq[3];
    int boards_used;
    int mode;  // 0=degenerate, 1=exact_enum, 2=mc, 3=mc_cap, 4=exact_enum_cap
};

void exact_six_card_combo_metrics(const int hole[6], TrioComboMetrics& out);

void mc_nway_shares(const int hole[6], int n_players, int n_boards, uint32_t seed, double* shares_out,
                    double* pair_eq_out = nullptr);

/** Batched, OpenMP-parallel n-way MC pot shares for many independent showdowns.
 *
 * Reuses the SAME OMPEval evaluator + Fisher-Yates board sampler as ``mc_nway_shares`` (so the
 * output is statistically identical to calling that per showdown; only the per-showdown RNG seed
 * and the parallel execution differ from a serial Python loop). Designed for the 4-max marginal
 * precompute hot path, where one chunk has thousands of distinct (alive-hole-set) showdowns.
 *
 *   holes_flat : concatenated hole-card ids for all showdowns (2 ids per player).
 *   sizes      : n_show entries, ``sizes[s]`` = number of players in showdown s (2..4).
 *   seeds      : n_show entries, RNG seed per showdown (caller supplies a stable per-showdown seed).
 *   n_boards   : MC boards per showdown.
 *   shares_out : n_show * max_players doubles, row-major; row s holds ``sizes[s]`` shares, the
 *                remaining ``max_players - sizes[s]`` entries are left 0.
 *   max_players: stride of shares_out (>= max(sizes)).
 *   threads    : OpenMP thread count (0 = OpenMP default / all cores).
 */
void mc_nway_shares_batch(const int* holes_flat, const int* sizes, const uint32_t* seeds, int n_show,
                          int n_boards, double* shares_out, int max_players, int threads);

TrioAdaptiveResult eval_combo_adaptive(const int hole[6], uint32_t seed, int max_boards, int min_boards,
                                       double stderr_gate);

int count_combo_triples(int row, int j, int k);

void exact_pair_eq_for_triple(int row, int j, int k, double* pair_eq_out, int* combo_count_out);

/** Exact combo-averaged 3-way shares only (full enumeration, no MC). */
void exact_shares_for_triple(int row, int j, int k, double* shares_out, int* combo_count_out);

void avg_metrics_for_triple(int row, int j, int k, int mc_boards, int exact_combo_max, int min_boards_per_combo,
                            double stderr_gate, double* shares_out, double* pair_eq_out, int* combo_count_out,
                            const float* pair_eq_lookup = nullptr, int pair_eq_lookup_stride = 0);

bool verify_trio_sentinel();

struct TrioRowProfile {
    int64_t exact_triple_count;
    int64_t mc_triple_count;
    double exact_triple_ns;
    double mc_triple_ns;
    double mc_hidden_pair_exact_ns;
    int64_t mc_board_evals;
};

void reset_trio_row_profile();
TrioRowProfile get_trio_row_profile();
