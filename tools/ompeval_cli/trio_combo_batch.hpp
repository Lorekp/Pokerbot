#pragma once

#include <cstdint>

void build_trio_row_chunk(int row, int j_start, int j_end, int mc_boards, int exact_combo_max, int min_boards_per_combo,
                          double stderr_gate, float* shares_out, float* pair_eq_out, std::int32_t* combo_counts_out,
                          int n_buckets, int threads, const float* pair_eq_lookup = nullptr,
                          int pair_eq_lookup_stride = 0);

/** OpenMP batch exact 3-way share oracle for sparse (row,j,k) validation samples. */
void batch_exact_shares_for_triples(const int* rows, const int* js, const int* ks, int n_triples, float* shares_out,
                                    std::int32_t* combo_counts_out, int threads);
