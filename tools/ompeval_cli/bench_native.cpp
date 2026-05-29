#include "hu_score2_batch.hpp"
#include "hu_score2_core.hpp"
#include "hu_table_io.hpp"

#include <benchmark/benchmark.h>

#include <cstdint>
#include <string>
#include <vector>

namespace {

std::vector<uint64_t> g_keys;

void ensure_keys_loaded(const char* fixture_path) {
    if (!g_keys.empty()) {
        return;
    }
    const KeyFile file = load_keys_file(fixture_path);
    g_keys = std::move(file.keys);
}

void BM_build_scores_fixture(benchmark::State& state) {
    const char* fixture_path =
        std::getenv("HU_BENCH_KEYS") != nullptr
            ? std::getenv("HU_BENCH_KEYS")
            : "../../bench/datasets/fixtures/smoke_keys.uint64.npy";
    ensure_keys_loaded(fixture_path);
    const int threads = static_cast<int>(state.range(0));
    std::vector<uint32_t> out(g_keys.size());

    for (auto _ : state) {
        build_scores(g_keys.data(), out.data(), g_keys.size(), threads);
    }

    state.SetItemsProcessed(static_cast<int64_t>(g_keys.size()) *
                            static_cast<int64_t>(state.iterations()));
    state.counters["boards_per_s"] = benchmark::Counter(
        static_cast<double>(g_keys.size()) * 1712304.0,
        benchmark::Counter::kIsRate);
    state.counters["eval7_per_s"] = benchmark::Counter(
        static_cast<double>(g_keys.size()) * 3424608.0, benchmark::Counter::kIsRate);
}

void BM_enumerate_score2_one_gate_a(benchmark::State& state) {
    for (auto _ : state) {
        benchmark::DoNotOptimize(enumerate_score2_one(SENTINEL_KEY_AA_AKO));
    }
    state.SetItemsProcessed(static_cast<int64_t>(state.iterations()));
}

}  // namespace

BENCHMARK(BM_enumerate_score2_one_gate_a)->Unit(benchmark::kMillisecond)->Repetitions(5);

// Thread count passed via --benchmark_args=<threads>; default sweep uses wrapper script.
BENCHMARK(BM_build_scores_fixture)
    ->Arg(1)
    ->Arg(2)
    ->Arg(4)
    ->Unit(benchmark::kMillisecond)
    ->Repetitions(5);

BENCHMARK_MAIN();
