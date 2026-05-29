/**
 * Phase 2/3 native HU showdown score2 oracle (pybind11 wrapper).
 *
 * Core logic lives in hu_score2_core / hu_score2_batch (shared with CLI + bench).
 */
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <cstdint>
#include <stdexcept>
#include <tuple>
#include <vector>

#include "hu_score2_batch.hpp"
#include "hu_score2_core.hpp"

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

PYBIND11_MODULE(hu_table_native, m) {
    m.doc() = "HU showdown score2 native oracle (Phase 2/3)";
    m.attr("DENOM2") = DENOM2;
    m.attr("SENTINEL_KEY_AA_AKO") = SENTINEL_KEY_AA_AKO;
    m.attr("SENTINEL_SCORE2_AA_AKO") = SENTINEL_SCORE2_AA_AKO;
    m.def("representative_from_key", &representative_from_key,
          py::arg("key"), "Decode canonical uint64 key to (h0,h1,v0,v1) card ids");
    m.def("enumerate_score2", &enumerate_score2, py::arg("h0"), py::arg("h1"),
          py::arg("v0"), py::arg("v1"), "Exact score2 for one disjoint matchup");
    m.def("enumerate_score2_one", &enumerate_score2_one, py::arg("key"),
          "Exact score2 for one canonical key");
    m.def("build_scores", &build_scores_py, py::arg("keys"), py::arg("threads") = 0,
          "Batch score2 for canonical keys; threads=0 uses OpenMP default");
    m.def("verify_sentinel", &verify_sentinel,
          "Functional self-test: sentinel key must match bootstrapped score2");
    m.def("native_outcome", &native_outcome, py::arg("hr"), py::arg("vr"),
          "Map OMPEval ranks to score2 delta (+2 win, +1 tie, +0 loss for hero)");
    m.def(
        "eval7",
        [](int c0, int c1, int c2, int c3, int c4, int c5, int c6) {
            return eval7(c0, c1, c2, c3, c4, c5, c6);
        },
        py::arg("c0"), py::arg("c1"), py::arg("c2"), py::arg("c3"), py::arg("c4"),
        py::arg("c5"), py::arg("c6"), "OMPEval 7-card category (higher = better)");
}
