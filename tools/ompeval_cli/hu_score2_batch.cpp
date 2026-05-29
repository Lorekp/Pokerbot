#include "hu_score2_batch.hpp"

#include "hu_score2_core.hpp"

#include <omp.h>

void build_scores(const uint64_t* keys, uint32_t* out, std::size_t n_keys, int threads) {
    if (threads > 0) {
        omp_set_num_threads(threads);
    }

#pragma omp parallel for schedule(static)
    for (std::ptrdiff_t i = 0; i < static_cast<std::ptrdiff_t>(n_keys); ++i) {
        out[static_cast<std::size_t>(i)] = enumerate_score2_one(keys[static_cast<std::size_t>(i)]);
    }
}
