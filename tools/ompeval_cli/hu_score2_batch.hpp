#pragma once

#include <cstddef>
#include <cstdint>

void build_scores(const uint64_t* keys, uint32_t* out, std::size_t n_keys, int threads);
