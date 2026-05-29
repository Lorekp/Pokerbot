/**
 * Phase 3 one-shot HU showdown score2 builder CLI.
 *
 * Reads frozen canonical keys (.npy or raw uint64) and writes raw uint32 score2.
 */
#include "hu_score2_batch.hpp"
#include "hu_score2_core.hpp"
#include "hu_table_io.hpp"

#include <chrono>
#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

namespace {

struct Options {
    std::string keys_path;
    std::string out_path;
    int threads = 0;
    int chunk_size = 1000;
    std::string format = "raw";
};

void print_usage(const char* argv0) {
    std::cerr
        << "Usage: " << argv0
        << " --keys PATH --out PATH [--threads N] [--chunk-size N] [--format raw]\n"
        << "  --keys PATH       1-D uint64 .npy fixture or raw little-endian uint64[]\n"
        << "  --out PATH        raw uint32 score2 output (one score per key, key order)\n"
        << "  --threads N       OpenMP threads (0 = default)\n"
        << "  --chunk-size N    progress log interval (default 1000)\n"
        << "  --format raw      only supported format in Phase 3\n";
}

bool parse_args(int argc, char** argv, Options& opts) {
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        auto need_value = [&](const char* name) -> std::string {
            if (i + 1 >= argc) {
                throw std::runtime_error(std::string("missing value for ") + name);
            }
            return argv[++i];
        };
        if (arg == "--keys") {
            opts.keys_path = need_value("--keys");
        } else if (arg == "--out") {
            opts.out_path = need_value("--out");
        } else if (arg == "--threads") {
            opts.threads = std::stoi(need_value("--threads"));
        } else if (arg == "--chunk-size") {
            opts.chunk_size = std::stoi(need_value("--chunk-size"));
        } else if (arg == "--format") {
            opts.format = need_value("--format");
        } else if (arg == "--help" || arg == "-h") {
            print_usage(argv[0]);
            return false;
        } else {
            throw std::runtime_error("unknown argument: " + arg);
        }
    }
    if (opts.keys_path.empty() || opts.out_path.empty()) {
        print_usage(argv[0]);
        return false;
    }
    if (opts.format != "raw") {
        throw std::runtime_error("unsupported --format (Phase 3 supports raw only)");
    }
    return true;
}

}  // namespace

int main(int argc, char** argv) {
    Options opts{};
    try {
        if (!parse_args(argc, argv, opts)) {
            return opts.keys_path.empty() ? 1 : 0;
        }
    } catch (const std::exception& ex) {
        std::cerr << "argument error: " << ex.what() << "\n";
        print_usage(argv[0]);
        return 1;
    }

    if (!verify_sentinel()) {
        std::cerr << "sentinel self-test failed; refusing to build\n";
        return 2;
    }

    try {
        const KeyFile keys = load_keys_file(opts.keys_path);
        if (keys.keys.empty()) {
            std::cerr << "no keys loaded from " << opts.keys_path << "\n";
            return 3;
        }

        std::vector<uint32_t> scores(keys.keys.size());
        const auto t0 = std::chrono::steady_clock::now();
        build_scores(keys.keys.data(), scores.data(), keys.keys.size(), opts.threads);
        const auto t1 = std::chrono::steady_clock::now();
        write_scores_raw(opts.out_path, scores.data(), scores.size());

        const double wall_s =
            std::chrono::duration<double>(t1 - t0).count();
        const double s_per_key = wall_s / static_cast<double>(keys.keys.size());
        std::cerr << "built " << keys.keys.size() << " keys in " << wall_s << " s ("
                  << s_per_key << " s/key)\n";
        std::cerr << "wrote " << opts.out_path << "\n";
        return 0;
    } catch (const std::exception& ex) {
        std::cerr << "build failed: " << ex.what() << "\n";
        return 4;
    }
}
