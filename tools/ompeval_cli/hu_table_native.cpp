/**
 * Phase 2/3 native HU showdown score2 oracle (pybind11 wrapper).
 *
 * Core logic lives in hu_score2_core / hu_score2_batch (shared with CLI + bench).
 * Trio 3-way combo kernels in trio_combo_core / trio_combo_batch.
 */
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <algorithm>
#include <cstdint>
#include <stdexcept>
#include <tuple>
#include <vector>

#include "hu_score2_batch.hpp"
#include "hu_score2_core.hpp"
#include "trio_combo_batch.hpp"
#include "trio_combo_core.hpp"

namespace py = pybind11;

static py::array_t<uint32_t> build_scores_py(py::array_t<uint64_t, py::array::c_style | py::array::forcecast> keys,
                                             int threads) {
    py::buffer_info key_buf = keys.request();
    if (key_buf.ndim != 1) {
        throw std::invalid_argument("keys must be a 1-D array");
    }
    const auto* key_ptr = static_cast<const uint64_t*>(key_buf.ptr);
    const std::size_t n_keys = static_cast<std::size_t>(key_buf.shape[0]);

    py::array_t<uint32_t> out(static_cast<py::ssize_t>(n_keys));
    py::buffer_info out_buf = out.request();
    auto* out_ptr = static_cast<uint32_t*>(out_buf.ptr);

    {
        py::gil_scoped_release release;
        build_scores(key_ptr, out_ptr, n_keys, threads);
    }
    return out;
}

static py::array_t<double> exact_six_card_combo_metrics_py(
    py::array_t<int, py::array::c_style | py::array::forcecast> cards) {
    py::buffer_info buf = cards.request();
    if (buf.ndim != 1 || buf.shape[0] != 6) {
        throw std::invalid_argument("cards must be length-6 int array");
    }
    const auto* ptr = static_cast<const int*>(buf.ptr);
    int hole[6];
    for (int i = 0; i < 6; ++i) {
        hole[i] = ptr[i];
    }
    TrioComboMetrics m{};
    exact_six_card_combo_metrics(hole, m);
    py::array_t<double> out(6);
    auto* o = out.mutable_data();
    o[0] = m.shares[0];
    o[1] = m.shares[1];
    o[2] = m.shares[2];
    o[3] = m.pair_eq[0];
    o[4] = m.pair_eq[1];
    o[5] = m.pair_eq[2];
    return out;
}

static py::array_t<double> mc_nway_shares_py(py::array_t<int, py::array::c_style | py::array::forcecast> cards,
                                             int n_players, int n_boards, uint32_t seed) {
    py::buffer_info buf = cards.request();
    if (buf.ndim != 1) {
        throw std::invalid_argument("cards must be 1-D int array");
    }
    const int n_cards = static_cast<int>(buf.shape[0]);
    if (n_cards < 2 * n_players) {
        throw std::invalid_argument("cards too short for n_players");
    }
    const auto* ptr = static_cast<const int*>(buf.ptr);
    int hole[6] = {};
    for (int i = 0; i < 2 * n_players && i < 6; ++i) {
        hole[i] = ptr[i];
    }
    py::array_t<double> out(n_players);
    double shares[6] = {};
    mc_nway_shares(hole, n_players, n_boards, seed, shares);
    auto* o = out.mutable_data();
    for (int i = 0; i < n_players; ++i) {
        o[i] = shares[i];
    }
    return out;
}

static py::array_t<double> mc_nway_shares_batch_py(
    py::array_t<int, py::array::c_style | py::array::forcecast> holes_flat,
    py::array_t<int, py::array::c_style | py::array::forcecast> sizes,
    py::array_t<uint32_t, py::array::c_style | py::array::forcecast> seeds, int n_boards, int threads) {
    py::buffer_info sizes_buf = sizes.request();
    py::buffer_info seeds_buf = seeds.request();
    py::buffer_info holes_buf = holes_flat.request();
    if (sizes_buf.ndim != 1 || seeds_buf.ndim != 1 || holes_buf.ndim != 1) {
        throw std::invalid_argument("holes_flat, sizes, seeds must be 1-D arrays");
    }
    const int n_show = static_cast<int>(sizes_buf.shape[0]);
    if (seeds_buf.shape[0] != n_show) {
        throw std::invalid_argument("sizes and seeds length mismatch");
    }
    const int* sizes_ptr = static_cast<const int*>(sizes_buf.ptr);
    const uint32_t* seeds_ptr = static_cast<const uint32_t*>(seeds_buf.ptr);
    const int* holes_ptr = static_cast<const int*>(holes_buf.ptr);

    int max_players = 0;
    int64_t need = 0;
    for (int s = 0; s < n_show; ++s) {
        const int sz = sizes_ptr[s];
        if (sz < 2 || sz > 6) {
            throw std::invalid_argument("each showdown size must be in [2, 6]");
        }
        if (sz > max_players) {
            max_players = sz;
        }
        need += 2 * sz;
    }
    if (need != static_cast<int64_t>(holes_buf.shape[0])) {
        throw std::invalid_argument("holes_flat length != 2 * sum(sizes)");
    }
    if (max_players == 0) {
        max_players = 1;
    }

    py::array_t<double> out({n_show, max_players});
    auto* out_ptr = out.mutable_data();
    std::fill(out_ptr, out_ptr + static_cast<int64_t>(n_show) * max_players, 0.0);
    if (n_show > 0) {
        py::gil_scoped_release release;
        mc_nway_shares_batch(holes_ptr, sizes_ptr, seeds_ptr, n_show, n_boards, out_ptr, max_players,
                             threads);
    }
    return out;
}

static py::tuple eval_combo_adaptive_py(py::array_t<int, py::array::c_style | py::array::forcecast> cards,
                                        uint32_t seed, int max_boards, int min_boards, double stderr_gate) {
    py::buffer_info buf = cards.request();
    if (buf.ndim != 1 || buf.shape[0] != 6) {
        throw std::invalid_argument("cards must be length-6 int array");
    }
    const auto* ptr = static_cast<const int*>(buf.ptr);
    int hole[6];
    for (int i = 0; i < 6; ++i) {
        hole[i] = ptr[i];
    }
    const TrioAdaptiveResult r = eval_combo_adaptive(hole, seed, max_boards, min_boards, stderr_gate);
    py::array_t<double> metrics(6);
    auto* o = metrics.mutable_data();
    o[0] = r.shares[0];
    o[1] = r.shares[1];
    o[2] = r.shares[2];
    o[3] = r.pair_eq[0];
    o[4] = r.pair_eq[1];
    o[5] = r.pair_eq[2];
    return py::make_tuple(metrics, r.boards_used, r.mode);
}

static py::tuple avg_metrics_for_triple_py(int row, int j, int k, int mc_boards, int exact_combo_max,
                                           int min_boards_per_combo, double stderr_gate,
                                           py::object pair_eq_lookup_obj) {
    double shares[3];
    double pair_eq[3];
    int combo_count = 0;
    const float* lookup_ptr = nullptr;
    int lookup_stride = 0;
    py::array_t<float, py::array::c_style | py::array::forcecast> lookup_arr;
    if (!pair_eq_lookup_obj.is_none()) {
        lookup_arr = pair_eq_lookup_obj.cast<py::array_t<float>>();
        if (lookup_arr.ndim() != 3 || lookup_arr.shape(2) != 3) {
            throw std::invalid_argument("pair_eq_lookup must have shape (n_j, n_k, 3)");
        }
        lookup_ptr = lookup_arr.data();
        lookup_stride = static_cast<int>(lookup_arr.shape(1));
    }
    avg_metrics_for_triple(row, j, k, mc_boards, exact_combo_max, min_boards_per_combo, stderr_gate, shares, pair_eq,
                           &combo_count, lookup_ptr, lookup_stride);
    return py::make_tuple(shares[0], shares[1], shares[2], pair_eq[0], pair_eq[1], pair_eq[2], combo_count);
}

static py::dict get_trio_row_profile_py() {
    const TrioRowProfile p = get_trio_row_profile();
    py::dict out;
    out["exact_triple_count"] = p.exact_triple_count;
    out["mc_triple_count"] = p.mc_triple_count;
    out["exact_triple_ns"] = p.exact_triple_ns;
    out["mc_triple_ns"] = p.mc_triple_ns;
    out["mc_hidden_pair_exact_ns"] = p.mc_hidden_pair_exact_ns;
    out["mc_board_evals"] = p.mc_board_evals;
    return out;
}

static py::tuple build_trio_row_chunk_py(int row, int j_start, int j_end, int mc_boards, int exact_combo_max,
                                         int min_boards_per_combo, double stderr_gate, int n_buckets, int threads,
                                         py::object pair_eq_lookup_obj) {
    if (j_end <= j_start) {
        throw std::invalid_argument("j_end must exceed j_start");
    }
    const int width = j_end - j_start;
    const std::size_t cells = static_cast<std::size_t>(width) * static_cast<std::size_t>(n_buckets);
    py::array_t<float> shares({width, n_buckets, 3});
    py::array_t<float> pair_eq({width, n_buckets, 3});
    py::array_t<std::int32_t> combo_counts({width, n_buckets});

    const float* lookup_ptr = nullptr;
    int lookup_stride = 0;
    py::array_t<float, py::array::c_style | py::array::forcecast> lookup_arr;
    if (!pair_eq_lookup_obj.is_none()) {
        lookup_arr = pair_eq_lookup_obj.cast<py::array_t<float>>();
        if (lookup_arr.ndim() != 3 || lookup_arr.shape(1) != n_buckets || lookup_arr.shape(2) != 3) {
            throw std::invalid_argument("pair_eq_lookup must have shape (n_j, n_buckets, 3)");
        }
        lookup_ptr = lookup_arr.data();
        lookup_stride = n_buckets;
    }

    {
        py::gil_scoped_release release;
        build_trio_row_chunk(row, j_start, j_end, mc_boards, exact_combo_max, min_boards_per_combo, stderr_gate,
                             shares.mutable_data(), pair_eq.mutable_data(), combo_counts.mutable_data(), n_buckets,
                             threads, lookup_ptr, lookup_stride);
    }
    return py::make_tuple(shares, pair_eq, combo_counts);
}

static py::tuple exact_pair_eq_for_triple_py(int row, int j, int k) {
    double pair_eq[3];
    int combo_count = 0;
    exact_pair_eq_for_triple(row, j, k, pair_eq, &combo_count);
    return py::make_tuple(pair_eq[0], pair_eq[1], pair_eq[2], combo_count);
}

static py::tuple batch_exact_shares_for_triples_py(py::array_t<int, py::array::c_style | py::array::forcecast> rows,
                                                   py::array_t<int, py::array::c_style | py::array::forcecast> js,
                                                   py::array_t<int, py::array::c_style | py::array::forcecast> ks,
                                                   int threads) {
    if (rows.ndim() != 1 || js.ndim() != 1 || ks.ndim() != 1) {
        throw std::invalid_argument("rows, js, ks must be 1-D int arrays");
    }
    const py::ssize_t n = rows.shape(0);
    if (js.shape(0) != n || ks.shape(0) != n) {
        throw std::invalid_argument("rows, js, ks length mismatch");
    }
    py::array_t<float> shares(std::vector<py::ssize_t>{n, 3});
    py::array_t<std::int32_t> combo_counts({n});
    if (n == 0) {
        return py::make_tuple(shares, combo_counts);
    }
    {
        py::gil_scoped_release release;
        batch_exact_shares_for_triples(rows.data(), js.data(), ks.data(), static_cast<int>(n), shares.mutable_data(),
                                       combo_counts.mutable_data(), threads);
    }
    return py::make_tuple(shares, combo_counts);
}

PYBIND11_MODULE(hu_table_native, m) {
    m.doc() = "HU showdown score2 + trio combo native oracle";
    m.attr("DENOM2") = DENOM2;
    m.attr("SENTINEL_KEY_AA_AKO") = SENTINEL_KEY_AA_AKO;
    m.attr("SENTINEL_SCORE2_AA_AKO") = SENTINEL_SCORE2_AA_AKO;
    m.def("representative_from_key", &representative_from_key, py::arg("key"),
          "Decode canonical uint64 key to (h0,h1,v0,v1) card ids");
    m.def("enumerate_score2", &enumerate_score2, py::arg("h0"), py::arg("h1"), py::arg("v0"), py::arg("v1"),
          "Exact score2 for one disjoint matchup");
    m.def("enumerate_score2_one", &enumerate_score2_one, py::arg("key"), "Exact score2 for one canonical key");
    m.def("build_scores", &build_scores_py, py::arg("keys"), py::arg("threads") = 0,
          "Batch score2 for canonical keys; threads=0 uses OpenMP default");
    m.def("verify_sentinel", &verify_sentinel,
          "Functional self-test: sentinel key must match bootstrapped score2");
    m.def("verify_trio_sentinel", &verify_trio_sentinel, "Trio combo kernel self-test");
    m.def("native_outcome", &native_outcome, py::arg("hr"), py::arg("vr"),
          "Map OMPEval ranks to score2 delta (+2 win, +1 tie, +0 loss for hero)");
    m.def(
        "eval7",
        [](int c0, int c1, int c2, int c3, int c4, int c5, int c6) {
            return eval7(c0, c1, c2, c3, c4, c5, c6);
        },
        py::arg("c0"), py::arg("c1"), py::arg("c2"), py::arg("c3"), py::arg("c4"), py::arg("c5"), py::arg("c6"),
        "OMPEval 7-card category (higher = better)");
    m.def("exact_six_card_combo_metrics", &exact_six_card_combo_metrics_py, py::arg("cards"),
          "Exact 3-way shares + pair eq for six hole cards");
    m.def("mc_nway_shares", &mc_nway_shares_py, py::arg("cards"), py::arg("n_players"), py::arg("n_boards"),
          py::arg("seed"), "Monte Carlo n-way pot shares");
    m.def("mc_nway_shares_batch", &mc_nway_shares_batch_py, py::arg("holes_flat"), py::arg("sizes"),
          py::arg("seeds"), py::arg("n_boards"), py::arg("threads") = 0,
          "Batched OpenMP-parallel n-way MC pot shares for many showdowns (shares_out [n_show, max_players])");
    m.def("eval_combo_adaptive", &eval_combo_adaptive_py, py::arg("cards"), py::arg("seed"), py::arg("max_boards"),
          py::arg("min_boards") = 16, py::arg("stderr_gate") = 0.001, "Adaptive MC 3-way shares (pair-eq via table)");
    m.def("exact_pair_eq_for_triple", &exact_pair_eq_for_triple_py, py::arg("row"), py::arg("j"), py::arg("k"),
          "Exact combo-averaged pair equities for one bucket triple");
    m.def("avg_metrics_for_triple", &avg_metrics_for_triple_py, py::arg("row"), py::arg("j"), py::arg("k"),
          py::arg("mc_boards"), py::arg("exact_combo_max") = 36, py::arg("min_boards_per_combo") = 16,
          py::arg("stderr_gate") = 0.001, py::arg("pair_eq_lookup") = py::none(),
          "Combo-averaged metrics for one bucket triple");
    m.def("reset_trio_row_profile", &reset_trio_row_profile, "Reset trio row path profiler counters");
    m.def("get_trio_row_profile", &get_trio_row_profile_py, "Read merged trio row path profiler counters");
    m.def("build_trio_row_chunk", &build_trio_row_chunk_py, py::arg("row"), py::arg("j_start"), py::arg("j_end"),
          py::arg("mc_boards"), py::arg("exact_combo_max") = 36, py::arg("min_boards_per_combo") = 16,
          py::arg("stderr_gate") = 0.001, py::arg("n_buckets") = 169, py::arg("threads") = 0,
          py::arg("pair_eq_lookup") = py::none(), "OpenMP row slice builder for trio tensor");
    m.def("batch_exact_shares_for_triples", &batch_exact_shares_for_triples_py, py::arg("rows"), py::arg("js"),
          py::arg("ks"), py::arg("threads") = 0,
          "OpenMP batch exact 3-way share oracle for validation samples");
}
