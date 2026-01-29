#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "cpp/praxis.cpp"
#include "cpp/rid.cpp"

namespace py = pybind11;

PYBIND11_MODULE(_core, m) {
    m.doc() = "PRAXIS C++ core bindings";

    py::class_<PRAXIS>(m, "PRAXIS")
        .def(py::init<>())

        .def(
            "fit",
            [](PRAXIS &self,
               py::array_t<uint8_t, py::array::c_style | py::array::forcecast> X,
               py::array_t<int,     py::array::c_style | py::array::forcecast> y,
               double lambda_reg,
               int depth_budget,
               double rashomon_mult,
               double multiplicative_slack,
               std::string key_mode_str,
               bool trie_cache_enabled,
               int lookahead_k,
               int root_budget,
               bool use_multipass,
               bool rule_list_mode,
               int oracle_style,
               bool majority_leaf_only,
               bool cache_cheap_subproblems,
               int greedy_split_mode,
               bool proxy_caching,
               int num_proxy_features,
               bool rashomon_mode
            ) {

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

                PRAXIS::KeyMode km;
                if (key_mode_str == "exact") {
                    km = PRAXIS::KeyMode::EXACT;
                } else if (key_mode_str == "literal" || key_mode_str == "lits" || key_mode_str == "lits_exact" || key_mode_str == "itemset") {
                    km = PRAXIS::KeyMode::LITS_EXACT;
                } else {
                    km = PRAXIS::KeyMode::HASH64;
                }


                self.set_key_mode(km);
                self.set_trie_cache_enabled(trie_cache_enabled);
                self.set_multiplicative_slack(multiplicative_slack);
                self.set_use_multipass(use_multipass);
                self.set_rule_list_mode(rule_list_mode);
                self.set_cache_cheap_subproblems(cache_cheap_subproblems);
                self.set_greedy_split_mode(greedy_split_mode);
                self.set_majority_leaf_only(majority_leaf_only);
                self.set_proxy_caching_enabled(proxy_caching);

                self.fit(
                    X_col_major,
                    y_vec,
                    lambda_reg,
                    depth_budget,
                    rashomon_mult,
                    lookahead_k,
                    root_budget,
                    use_multipass,
                    rule_list_mode,
                    oracle_style,
                    majority_leaf_only,
                    cache_cheap_subproblems,
                    proxy_caching,
                    num_proxy_features,
                    rashomon_mode
                );
            },
            py::arg("X"),
            py::arg("y"),
            py::arg("lambda_reg") = 0.01,
            py::arg("depth_budget") = 5,
            py::arg("rashomon_mult") = 0.01,
            py::arg("multiplicative_slack") = 0.0,
            py::arg("key_mode") = "hash",
            py::arg("trie_cache_enabled") = false,
            py::arg("lookahead_k") = 1,
            py::arg("root_budget") = -1,
            py::arg("use_multipass") = true,
            py::arg("rule_list_mode") = false,
            py::arg("oracle_style") = 0,
            py::arg("majority_leaf_only") = false,
            py::arg("cache_cheap_subproblems") = false,
            py::arg("greedy_split_mode") = 1,
            py::arg("proxy_caching") = true,
            py::arg("num_proxy_features") = 0,
            py::arg("rashomon_mode") = true
        )

        .def("count_trees",
             [](PRAXIS &self) {
                 return self.result ? self.result->count_trees() : 0ULL;
             })

        .def("get_min_objective",
             [](PRAXIS &self) {
                 return self.result
                        ? self.result->min_objective
                        : std::numeric_limits<int>::max();
             })

        .def("get_root_histogram",
             [](PRAXIS &self) {
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
            [](const PRAXIS &self,
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
            [](const PRAXIS &self,
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
            "get_tree_objective",
            [](const PRAXIS &self, std::uint64_t tree_index) {
                auto obj_pair = self.get_ith_tree_objective(tree_index);
                // obj_pair.first  = unnormalized objective (int)
                // obj_pair.second = normalized objective (double)
                return py::make_tuple(obj_pair.first, obj_pair.second);
            },
            py::arg("tree_index")
        )

        .def(
            "get_tree_paths",
            [](const PRAXIS &self, std::uint64_t tree_index) {
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
        )

        .def(
            "get_tree_frontier_scores",
            [](PRAXIS &self, std::uint64_t tree_index, int depth_budget) {
                auto vec = self.get_tree_frontier_scores(tree_index, depth_budget);
                return vec;
            },
            py::arg("tree_index"),
            py::arg("depth_budget")
        )

        .def(
            "root_lickety_objective_lookahead1",
            [](PRAXIS &self, int depth_budget) {
                return self.root_lickety_objective_lookahead1(depth_budget);
            },
            py::arg("depth_budget")
        );

    m.def(
        "rid_subtractive_model_reliance",
        [](py::array_t<uint8_t, py::array::c_style | py::array::forcecast> X,
        py::array_t<int,     py::array::c_style | py::array::forcecast> y,
        int n_boot,
        double lambda_reg,
        int depth_budget,
        double rashomon_mult,
        int lookahead_k,
        std::uint64_t seed,
        bool memory_efficient,
        py::object binning_map_obj ) {
            py::buffer_info xinfo = X.request();
            py::buffer_info yinfo = y.request();

            if (xinfo.ndim != 2) throw std::runtime_error("X must be 2D");
            if (yinfo.ndim != 1) throw std::runtime_error("y must be 1D");

            int n_samples  = (int)xinfo.shape[0];
            int n_features = (int)xinfo.shape[1];
            if ((int)yinfo.shape[0] != n_samples) throw std::runtime_error("y must match X rows");

            auto *x_ptr = static_cast<uint8_t*>(xinfo.ptr);
            auto *y_ptr = static_cast<int*>(yinfo.ptr);

            // build row-major X
            std::vector<std::vector<uint8_t>> X_row_major(n_samples, std::vector<uint8_t>(n_features));
            for (int i = 0; i < n_samples; ++i) {
                std::memcpy(X_row_major[i].data(),
                            x_ptr + (std::size_t)i * (std::size_t)n_features,
                            (std::size_t)n_features * sizeof(uint8_t));
            }

            std::vector<int> y_vec(y_ptr, y_ptr + n_samples);   
            
            // binning map stuff
            int d = n_features;
            std::vector<std::vector<int>> groups;
            if (binning_map_obj.is_none()) {
                groups.resize(d);
                for (int j = 0; j < d; ++j) groups[j] = {j};
            } else {
                py::dict bm = binning_map_obj.cast<py::dict>();

                std::vector<int> keys;
                for (auto item : bm) keys.push_back(py::cast<int>(item.first));
                std::sort(keys.begin(), keys.end());

                groups.reserve(keys.size());
                for (int k : keys) {
                    py::list lst = bm[py::int_(k)].cast<py::list>();
                    std::vector<int> cols;
                    for (auto h : lst) cols.push_back(py::cast<int>(h));
                    groups.push_back(std::move(cols));
                }
            }


            RIDResult r = compute_rid_subtractive_mr_bootstrap(
                X_row_major,
                y_vec,
                n_boot,
                lambda_reg,
                depth_budget,
                rashomon_mult,
                lookahead_k,
                seed,
                memory_efficient,
                groups
            );

            py::dict out;
            out["mean_sub_mr"] = r.mean_sub_mr; // vector<double>
            out["cdf_x"] = r.cdf_x; // vector<vector<double>>
            out["cdf_p"] = r.cdf_p; // vector<vector<double>>
            return out;
        },
        py::arg("X"),
        py::arg("y"),
        py::arg("n_boot") = 10,
        py::arg("lambda_reg") = 0.01,
        py::arg("depth_budget") = 5,
        py::arg("rashomon_mult") = 0.05,
        py::arg("lookahead_k") = 1,
        py::arg("seed") = 0,
        py::arg("memory_efficient") = false,
        py::arg("binning_map") = py::none()
    );


}
