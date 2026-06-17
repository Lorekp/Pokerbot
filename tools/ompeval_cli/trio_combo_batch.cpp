#include "trio_combo_batch.hpp"

#include <omp.h>

#include "trio_combo_core.hpp"

void build_trio_row_chunk(int row, int j_start, int j_end, int mc_boards, int exact_combo_max, int min_boards_per_combo,
                          double stderr_gate, float* shares_out, float* pair_eq_out, std::int32_t* combo_counts_out,
                          int n_buckets, int threads, const float* pair_eq_lookup, int pair_eq_lookup_stride) {
    if (threads > 0) {
        omp_set_num_threads(threads);
    }
    const int width = j_end - j_start;
    const int n_tasks = width * n_buckets;
#pragma omp parallel for schedule(dynamic, 4)
    for (int t = 0; t < n_tasks; ++t) {
        const int jj = t / n_buckets;
        const int k = t % n_buckets;
        const int j = j_start + jj;
        double shares[3];
        double pair_eq[3];
        int combo_count = 0;
        avg_metrics_for_triple(row, j, k, mc_boards, exact_combo_max, min_boards_per_combo, stderr_gate, shares,
                               pair_eq, &combo_count, pair_eq_lookup, pair_eq_lookup_stride);
        const std::size_t base =
            (static_cast<std::size_t>(jj) * static_cast<std::size_t>(n_buckets) + static_cast<std::size_t>(k)) * 3u;
        for (int s = 0; s < 3; ++s) {
            shares_out[base + static_cast<std::size_t>(s)] = static_cast<float>(shares[s]);
            pair_eq_out[base + static_cast<std::size_t>(s)] = static_cast<float>(pair_eq[s]);
        }
        combo_counts_out[static_cast<std::size_t>(jj) * static_cast<std::size_t>(n_buckets) +
                         static_cast<std::size_t>(k)] = static_cast<std::int32_t>(combo_count);
    }
}

void batch_exact_shares_for_triples(const int* rows, const int* js, const int* ks, int n_triples, float* shares_out,
                                    std::int32_t* combo_counts_out, int threads) {
    if (threads > 0) {
        omp_set_num_threads(threads);
    }
#pragma omp parallel for schedule(dynamic, 16)
    for (int t = 0; t < n_triples; ++t) {
        double sh[3];
        int combo_count = 0;
        exact_shares_for_triple(rows[t], js[t], ks[t], sh, &combo_count);
        const std::size_t base = static_cast<std::size_t>(t) * 3u;
        for (int s = 0; s < 3; ++s) {
            shares_out[base + static_cast<std::size_t>(s)] = static_cast<float>(sh[s]);
        }
        combo_counts_out[t] = static_cast<std::int32_t>(combo_count);
    }
}
