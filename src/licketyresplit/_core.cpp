#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "cpp/lickety_resplit.cpp"

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
             })

        .def(
            "get_predictions",
            [](const LicketyRESPLIT &self,
               std::uint64_t tree_index,
               py::array_t<uint8_t, py::array::c_style | py::array::forcecast> X) {
                py::buffer_info xinfo = X.request();
                if (xinfo.ndim != 2) {
                    throw std::runtime_error("X must be 2D (n_samples x n_features)");
                }

                int n_samples  = static_cast<int>(xinfo.shape[0]);
                int n_features = static_cast<int>(xinfo.shape[1]);
                auto *x_ptr = static_cast<uint8_t*>(xinfo.ptr);

                std::vector<std::vector<uint8_t>> X_row_major(
                    n_samples, std::vector<uint8_t>(n_features));
                for (int i = 0; i < n_samples; ++i) {
                    for (int f = 0; f < n_features; ++f) {
                        X_row_major[i][f] = x_ptr[i * n_features + f];
                    }
                }

                auto preds = self.get_predictions(tree_index, X_row_major);

                py::array_t<uint8_t> out(n_samples);
                auto out_info = out.request();
                auto *out_ptr = static_cast<uint8_t*>(out_info.ptr);
                std::memcpy(
                    out_ptr, preds.data(),
                    static_cast<std::size_t>(n_samples) * sizeof(uint8_t));
                return out;
            },
            py::arg("tree_index"),
            py::arg("X")
        )

        .def(
            "get_all_predictions",
            [](const LicketyRESPLIT &self,
               py::array_t<uint8_t, py::array::c_style | py::array::forcecast> X,
               bool stack) {
                py::buffer_info xinfo = X.request();
                if (xinfo.ndim != 2) {
                    throw std::runtime_error("X must be 2D (n_samples x n_features)");
                }

                int n_samples  = static_cast<int>(xinfo.shape[0]);
                int n_features = static_cast<int>(xinfo.shape[1]);
                auto *x_ptr = static_cast<uint8_t*>(xinfo.ptr);

                std::vector<std::vector<uint8_t>> X_row_major(
                    n_samples, std::vector<uint8_t>(n_features));
                for (int i = 0; i < n_samples; ++i) {
                    for (int f = 0; f < n_features; ++f) {
                        X_row_major[i][f] = x_ptr[i * n_features + f];
                    }
                }

                auto all_preds = self.get_all_predictions(X_row_major);
                std::uint64_t total = all_preds.size();

                if (!stack) {
                    py::list lst;
                    for (std::uint64_t t = 0; t < total; ++t) {
                        py::array_t<uint8_t> arr(n_samples);
                        auto info = arr.request();
                        auto *ptr = static_cast<uint8_t*>(info.ptr);
                        std::memcpy(
                            ptr,
                            all_preds[t].data(),
                            static_cast<std::size_t>(n_samples) * sizeof(uint8_t));
                        lst.append(arr);
                    }
                    return py::object(lst);
                } else {
                    py::array_t<uint8_t> out(
                        {static_cast<py::ssize_t>(total),
                         static_cast<py::ssize_t>(n_samples)});
                    auto out_info = out.request();
                    auto *out_ptr = static_cast<uint8_t*>(out_info.ptr);
                    for (std::uint64_t t = 0; t < total; ++t) {
                        std::memcpy(
                            out_ptr + t * n_samples,
                            all_preds[t].data(),
                            static_cast<std::size_t>(n_samples) * sizeof(uint8_t));
                    }
                    return py::object(out);
                }
            },
            py::arg("X"),
            py::arg("stack") = false
        )

        .def(
            "get_tree_paths",
            [](const LicketyRESPLIT &self, std::uint64_t tree_index) {
                auto result = self.get_tree_paths(tree_index);
                const auto &paths = result.first;
                const auto &preds = result.second;

                py::list py_paths;
                for (const auto &p : paths) {
                    py::list py_path;
                    for (int v : p) {
                        py_path.append(v);
                    }
                    py_paths.append(py_path);
                }

                py::array_t<int> py_preds(preds.size());
                auto info = py_preds.request();
                auto *ptr = static_cast<int*>(info.ptr);
                for (std::size_t i = 0; i < preds.size(); ++i) {
                    ptr[i] = preds[i];
                }

                return py::make_tuple(py_paths, py_preds);
            },
            py::arg("tree_index")
        );
}
