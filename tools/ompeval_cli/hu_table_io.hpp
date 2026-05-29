#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

struct KeyFile {
    std::vector<uint64_t> keys;
};

// Minimal 1-D little-endian uint64 NumPy (.npy) reader for frozen fixtures only.
KeyFile load_keys_file(const std::string& path);

void write_scores_raw(const std::string& path, const uint32_t* scores, std::size_t n);
