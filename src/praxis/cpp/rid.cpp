#include <algorithm>
#include <cstdint>
#include <random>
#include <unordered_map>
#include <vector>
#include <iostream>
#include <cmath>

using std::cout;

struct RIDResult {
    std::vector<double> mean_sub_mr;
    std::vector<std::vector<double>> cdf_x;
    std::vector<std::vector<double>> cdf_p;
};

static inline uint64_t popcnt64_u(uint64_t x) {
#if defined(_MSC_VER)
    return (uint64_t)__popcnt64(x);
#else
    return (uint64_t)__builtin_popcountll(x);
#endif
}

// build y==1 bitset for eval dataset (size n, in n_words words)
// static inline Packed build_y1_packed(const std::vector<int>& y, int n_words, uint64_t tail_mask) {
//     Packed y1((size_t)n_words);
//     for (int i = 0; i < (int)y.size(); ++i) {
//         if (y[i]) y1.w[(size_t)(i >> 6)] |= (1ULL << (i & 63));
//     }
//     if (n_words > 0) y1.w[(size_t)(n_words - 1)] &= tail_mask;
//     return y1;
// }

// y_bits[c] has bit i = 1 iff y[i] == c
static inline std::vector<Packed> build_yc_packed(
    const std::vector<int>& y,
    int n_classes,
    int n_words,
    uint64_t tail_mask
) {
    std::vector<Packed> y_bits;
    y_bits.reserve((size_t)n_classes);
    for (int c = 0; c < n_classes; ++c) y_bits.emplace_back((size_t)n_words);

    for (int i = 0; i < (int)y.size(); ++i) {
        const int c = y[i];
        // (optional) assert 0 <= c < n_classes
        y_bits[(size_t)c].w[(size_t)(i >> 6)] |= (1ULL << (i & 63));
    }

    if (n_words > 0) {
        for (int c = 0; c < n_classes; ++c) {
            y_bits[(size_t)c].w[(size_t)(n_words - 1)] &= tail_mask;
        }
    }
    return y_bits;
}

// count correct predictions given pred1 bitset and y1 bitset.
// pred1 bit i = 1 iff prediction==1 on row i.
// static inline int count_correct_packed(const Packed& pred1, const Packed& y1, int n_words, uint64_t tail_mask) {
//     uint64_t correct = 0;
//     for (int w = 0; w < n_words; ++w) {
//         uint64_t p = pred1.w[(size_t)w];
//         uint64_t y = y1.w[(size_t)w];

//         // correct bits = (p & y) | (~p & ~y)
//         uint64_t c = (p & y) | (~p & ~y);

//         // mask tail on last word
//         if (w == n_words - 1) c &= tail_mask;

//         correct += popcnt64_u(c);
//     }
//     return (int)correct;
// }

static inline int count_correct_packed_multi(
    const PackedPredMulti& pred,
    const std::vector<Packed>& y_bits,
    int n_words,
    uint64_t tail_mask
) {
    const int C = (int)y_bits.size();
    uint64_t correct = 0;

    for (int c = 0; c < C; ++c) {
        const auto& pw = pred.by_class[(size_t)c].w;
        const auto& yw = y_bits[(size_t)c].w;

        for (int w = 0; w < n_words; ++w) {
            uint64_t bits = pw[(size_t)w] & yw[(size_t)w];
            if (w == n_words - 1) bits &= tail_mask;
            correct += popcnt64_u(bits);
        }
    }
    return (int)correct;
}



static inline void bootstrap_indices(int n, std::mt19937_64& rng, std::vector<int>& idx) {
    std::uniform_int_distribution<int> unif(0, n - 1);
    idx.resize(n);
    for (int i = 0; i < n; ++i) idx[i] = unif(rng);
}

static inline void make_bootstrap_dataset(
    const std::vector<std::vector<uint8_t>>& X,
    const std::vector<int>& y,
    const std::vector<int>& idx,
    std::vector<std::vector<uint8_t>>& Xb,
    std::vector<int>& yb
) {
    const int n = (int)idx.size();
    const int d = (int)X[0].size();
    Xb.assign(n, std::vector<uint8_t>(d));
    yb.assign(n, 0);
    for (int i = 0; i < n; ++i) {
        const int s = idx[i];
        Xb[i] = X[s];
        yb[i] = y[s];
    }
}

static inline void rowmajor_to_colmajor_bool(
    const std::vector<std::vector<uint8_t>>& X_row,
    std::vector<std::vector<bool>>& X_col
) {
    const int n = (int)X_row.size();
    const int d = (int)X_row[0].size();
    X_col.assign(d, std::vector<bool>(n, false));
    for (int i = 0; i < n; ++i) {
        const auto& row = X_row[i];
        for (int j = 0; j < d; ++j) {
            X_col[j][i] = (row[j] != 0);
        }
    }
}

static inline void make_permutation(int n, std::mt19937_64& rng, std::vector<int>& perm) {
    perm.resize(n);
    for (int i = 0; i < n; ++i) perm[i] = i;
    std::shuffle(perm.begin(), perm.end(), rng);
}

// scramble one column of X in-place using perm, but without copying whole X.
// caller should restore original column after use.
// static inline void scramble_column_inplace(
//     std::vector<std::vector<uint8_t>>& X,
//     int col,
//     const std::vector<int>& perm,
//     std::vector<uint8_t>& saved_col
// ) {
//     const int n = (int)X.size();
//     saved_col.resize(n);
//     for (int i = 0; i < n; ++i) saved_col[i] = X[i][col];
//     for (int i = 0; i < n; ++i) X[i][col] = saved_col[perm[i]];
// }

// static inline void restore_column_inplace(
//     std::vector<std::vector<uint8_t>>& X,
//     int col,
//     const std::vector<uint8_t>& saved_col
// ) {
//     const int n = (int)X.size();
//     for (int i = 0; i < n; ++i) X[i][col] = saved_col[i];
// }

// scramble a single feature, but represented by multiple binary columns
static inline void scramble_block_inplace(
    std::vector<std::vector<uint8_t>>& X,
    const std::vector<int>& cols,
    const std::vector<int>& perm,
    std::vector<std::vector<uint8_t>>& saved_cols
) {
    const int n = (int)X.size();
    saved_cols.assign(cols.size(), std::vector<uint8_t>(n));

    // save originals
    for (size_t ci = 0; ci < cols.size(); ++ci) {
        const int col = cols[ci];
        for (int i = 0; i < n; ++i) saved_cols[ci][i] = X[i][col];
    }

    // apply same permutation to each column in the block
    for (size_t ci = 0; ci < cols.size(); ++ci) {
        const int col = cols[ci];
        for (int i = 0; i < n; ++i) X[i][col] = saved_cols[ci][perm[i]];
    }
}

static inline void restore_block_inplace(
    std::vector<std::vector<uint8_t>>& X,
    const std::vector<int>& cols,
    const std::vector<std::vector<uint8_t>>& saved_cols
) {
    const int n = (int)X.size();
    for (size_t ci = 0; ci < cols.size(); ++ci) {
        const int col = cols[ci];
        for (int i = 0; i < n; ++i) X[i][col] = saved_cols[ci][i];
    }
}

static inline int count_correct(const std::vector<uint8_t>& preds, const std::vector<int>& y) {
    const int n = (int)y.size();
    int c = 0;
    for (int i = 0; i < n; ++i) c += (preds[i] == (uint8_t)y[i]);
    return c;
}

RIDResult compute_rid_subtractive_mr_bootstrap(
    const std::vector<std::vector<uint8_t>>& X_row_major,
    const std::vector<int>& y,
    int n_bootstraps,
    double lambda,
    int depth_budget,
    double rashomon_mult,
    int lookahead_k,
    uint64_t seed,
    bool memory_efficient,
    const std::vector<std::vector<int>>& binning_map_vars = {}
) {
    const int n_full = (int)X_row_major.size();
    const int d = (int)X_row_major[0].size();

    // build var->cols mapping.
    // if no binning map is provided, assume no relationship between binary features
    std::vector<std::vector<int>> var_cols;
    if (!binning_map_vars.empty()) {
        var_cols = binning_map_vars;
    } else {
        var_cols.resize((size_t)d);
        for (int j = 0; j < d; ++j) var_cols[(size_t)j] = std::vector<int>{j};
    }

    const int V = (int)var_cols.size();

    std::mt19937_64 rng(seed);

    RIDResult out;
    out.mean_sub_mr.assign(V, 0.0);
    out.cdf_x.assign(V, {});
    out.cdf_p.assign(V, {});

    // for each feature j, we aggregate a weighted empirical distribution of delta_correct = correct_orig - correct_scrambled
    std::vector<std::unordered_map<int, double>> mass_by_delta(V); // maps feature, delta to mass

    for (int b = 0; b < n_bootstraps; ++b) {
        std::vector<int> idx;
        bootstrap_indices(n_full, rng, idx);

        std::vector<std::vector<uint8_t>> Xb;
        std::vector<int> yb;
        make_bootstrap_dataset(X_row_major, y, idx, Xb, yb);

        const int n = (int)Xb.size();
        const int n_words = (n + 63) / 64;
        const uint64_t tail_mask = (n % 64) ? ((1ULL << (n % 64)) - 1ULL) : ~0ULL;
        int y_max = 0;
        for (int i = 0; i < (int)yb.size(); ++i) y_max = std::max(y_max, yb[i]);
        const int n_classes = y_max + 1;

        const auto y_bits = build_yc_packed(yb, n_classes, n_words, tail_mask);

        // row-major -> col-major bool for training
        std::vector<std::vector<bool>> Xcol;
        rowmajor_to_colmajor_bool(Xb, Xcol);

        PRAXIS model;
        model.fit(
            Xcol,
            yb,
            lambda,
            depth_budget,
            rashomon_mult,
            lookahead_k,
            -1, true, false, 0, false, false, true, 0, true
        );

        const uint64_t T64 = model.result ? model.result->count_trees() : 0ULL;
        const int T = (int)T64;
        if (T == 0) continue;
        cout << "Finished RID bootstrap: " << (b + 1) << " / " << n_bootstraps << " with " << T << " trees\n";

        // pre-sample permutations for each feature (one scramble per feature per bootstrap)
        std::vector<std::vector<int>> perms((size_t)V);
        for (int v = 0; v < V; ++v) make_permutation(n, rng, perms[(size_t)v]);

        // reuse buffer for column/block scrambling
        std::vector<std::vector<uint8_t>> saved_cols;

        const int budget_override = (int)llround((1.0 + rashomon_mult) * (double)model.result->min_objective);
        auto orig = model.get_all_predictions_packed_trie(Xb, budget_override);
        const uint64_t Tvec = (uint64_t)orig.size();

        if (Tvec == 0) continue;
        
        // weight per tree per bootstrap
        const double wt_tree = 1.0 / ((double)n_bootstraps * (double)Tvec); // we may return more trees than we use (within new budget), so Tvec here


        std::vector<int> correct_orig((size_t)Tvec, 0);
        for (uint64_t t = 0; t < Tvec; ++t) {
            correct_orig[(size_t)t] = count_correct_packed_multi(orig[(size_t)t].pred, y_bits, n_words, tail_mask);
        }
    
        // Consider this optimization: convert the bootstrap to column major once, and scramble the columns in column major (which is probably slightly slower), and then replace get_all_predictions_packed_trie to take in column major instead of taking in row and converting to column.
        // I think precomputing column major once instead of f times is better, even if it is not ideal for the scrambling.
        
        for (int v = 0; v < V; ++v) {
            const std::vector<int>& cols = var_cols[(size_t)v];
            scramble_block_inplace(Xb, cols, perms[(size_t)v], saved_cols);

            auto scr = model.get_all_predictions_packed_trie(Xb, budget_override);
            const uint64_t Tuse = Tvec;

            for (uint64_t t = 0; t < Tuse; ++t) {
                const int correct_scr = count_correct_packed_multi(scr[(size_t)t].pred, y_bits, n_words, tail_mask);
                const int delta_correct = correct_orig[(size_t)t] - correct_scr;

                out.mean_sub_mr[v] += wt_tree * ((double)delta_correct / (double)n);
                mass_by_delta[v][delta_correct] += wt_tree;
            }

            restore_block_inplace(Xb, cols, saved_cols);
        }

        
    }

    // build weighted CDF for each feature from the mass map
    const double denom = (double)n_full;

    for (int v = 0; v < V; ++v) {
        std::vector<std::pair<int, double>> items;
        items.reserve(mass_by_delta[v].size());
        for (const auto& kv : mass_by_delta[v]) items.push_back(kv);

        std::sort(items.begin(), items.end(),
                [](const auto& a, const auto& b) { return a.first < b.first; });

        out.cdf_x[v].reserve(items.size());
        out.cdf_p[v].reserve(items.size());

        double cum = 0.0;
        for (const auto& kv : items) {
            const int delta = kv.first;
            const double w = kv.second;
            cum += w;
            out.cdf_x[v].push_back((double)delta / (double)n_full);
            out.cdf_p[v].push_back(cum);
        }
    }


    return out;
}
