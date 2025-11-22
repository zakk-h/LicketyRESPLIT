#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include "cpp/lickety_resplit.cpp

namespace py = pybind11;


PYBIND11_MODULE(_core, m) {
    m.doc() = "LicketyRESPLIT C++ core bindings";

    py::class_<LicketyRESPLIT>(m, "LicketyRESPLIT")
        .def(py::init<>())

        .def(
            "fit",
            [](LicketyRESPLIT &self,
               py::array_t<uint8_t, py::array::c_style | py::array::forcecast> X,
               py::array_t<int,     py::array::c_style | py::array::forcecast> y,
               double lambda_reg,
               int depth_budget,
               double rashomon_mult,
               double multiplicative_slack,
               std::string key_mode_str,
               bool trie_cache_enabled,
               int lookahead_k) {

                py::buffer_info xinfo = X.request();
                py::buffer_info yinfo = y.request();

                if (xinfo.ndim != 2) {
                    throw std::runtime_error("X must be 2D (n_samples x n_features)");
                }

                int n_samples  = static_cast<int>(xinfo.shape[0]);
                int n_features = static_cast<int>(xinfo.shape[1]);

                auto *x_ptr = static_cast<uint8_t*>(xinfo.ptr);
                auto *y_ptr = static_cast<int*>(yinfo.ptr);

                std::vector<std::vector<bool>> X_col_major(
                    n_features,
                    std::vector<bool>(n_samples)
                );

                for (int f = 0; f < n_features; ++f) {
                    for (int i = 0; i < n_samples; ++i) {
                        uint8_t v = x_ptr[i * n_features + f];
                        X_col_major[f][i] = (v != 0);
                    }
                }

                std::vector<int> y_vec(y_ptr, y_ptr + n_samples);

                LicketyRESPLIT::KeyMode km =
                    (key_mode_str == "exact"
                         ? LicketyRESPLIT::KeyMode::EXACT
                         : LicketyRESPLIT::KeyMode::HASH64);

                self.set_key_mode(km);
                self.set_trie_cache_enabled(trie_cache_enabled);
                self.set_multiplicative_slack(multiplicative_slack);

                self.fit(X_col_major, y_vec, lambda_reg,
                         depth_budget, rashomon_mult, lookahead_k);
            },
            py::arg("X"),
            py::arg("y"),
            py::arg("lambda_reg") = 0.01,
            py::arg("depth_budget") = 5,
            py::arg("rashomon_mult") = 0.01,
            py::arg("multiplicative_slack") = 0.0,
            py::arg("key_mode") = "hash",
            py::arg("trie_cache_enabled") = false,
            py::arg("lookahead_k") = 1
        )

        .def("count_trees",
             [](LicketyRESPLIT &self) {
                 return self.result ? self.result->count_trees() : 0ULL;
             })

        .def("get_min_objective",
             [](LicketyRESPLIT &self) {
                 return self.result
                        ? self.result->min_objective
                        : std::numeric_limits<int>::max();
             })

        .def("get_root_histogram",
             [](LicketyRESPLIT &self) {
                 if (!self.result) {
                     return std::vector<std::pair<int, std::uint64_t>>{};
                 }
                 self.result->ensure_hist_built();
                 const auto &hist = self.result->hist;
                 std::vector<std::pair<int, std::uint64_t>> out;
                 out.reserve(hist.size());
                 for (const auto &e : hist) {
                     out.emplace_back(e.obj, e.cnt);
                 }
                 return out;
             });
}
