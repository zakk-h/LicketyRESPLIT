#include <vector>
#include <unordered_map>
#include <algorithm>
#include <cmath>
#include <limits>
#include <cstring>
#include <cstdint>
#include <memory>
#include <string>
#include <iostream>

using namespace std;

static inline int popcnt64(uint64_t x) {
    return __builtin_popcountll(x);
}

struct Packed {
    vector<uint64_t> w; // words (64-bit each)
    Packed() = default;
    explicit Packed(size_t nwords) : w(nwords, 0ULL) {} // allocates a vector of nwords many 64-bit words, with all bits off

    inline void clear() { std::fill(w.begin(), w.end(), 0ULL); } // reset the mask to all bits off to reuse the same object without reallocating
    inline bool any() const { // checks if any bits are set to 1
        for (uint64_t t : w) if (t) return true;
        return false;
    }
    inline int count() const { // how many bits are 1 in total? for each word, queries popcount for 1 bits in that word.
        int s = 0;
        for (uint64_t t : w) s += popcnt64(t);
        return s;
    }
};

// scramble a 64-bit value
static inline uint64_t mix64(uint64_t x) {
    x += 0x9e3779b97f4a7c15ULL;
    x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL;
    x = (x ^ (x >> 27)) * 0x94d049bb133111ebULL;
    x = x ^ (x >> 31);
    return x;
}

// hash the array of words into a 64-bit hash value: many-to-one in theory, but with our expected amount of pruning, something like 54k total keys for a reasonably sized rashomon set computation, which yields something like 10^-11 probability of having a collision somewhere.
static inline uint64_t hash_mask64(const uint64_t* w, int n_words, uint64_t tail_mask) {
    uint64_t h = 0x9e3779b97f4a7c15ULL;
    for (int i = 0; i < n_words; ++i) {
        uint64_t x = w[i];
        if (i == n_words - 1) x &= tail_mask;
        uint64_t m = mix64(x);
        h ^= m + 0x9e3779b97f4a7c15ULL + (h << 6) + (h >> 2);
    }
    return h;
}

// used for exact, non-probabilistic keyks at the expense of more memory. we intern the exact bytes of a mask/bitvector and assign a small integer ID.
// first unique mask id 0, second unique mask id 1 and so on.
class MaskIdTable {
public:
    uint32_t intern(const Packed& mask, int n_words, uint64_t tail_mask) {
        const size_t bytes = (size_t)n_words * sizeof(uint64_t); // constant across the dataset, how many words needed * 64 byte length
        string key;
        key.resize(bytes);
        uint64_t* out = reinterpret_cast<uint64_t*>(&key[0]); // pointer to the start of key
        for (int i = 0; i < n_words; ++i) {
            uint64_t x = mask.w[i];
            if (i == n_words - 1) x &= tail_mask; // the last word may have padding bits, tail_mask zeroes out the unused bits.
            out[i] = x; // the byte representation of mask.w - we need to convert to use as a key in the unordered map
        }
        auto it = table.find(key); // have we seen this bitmask before?
        if (it != table.end()) return it->second; // return the previously assigned id if it points to the entry, meaning we have it already
        uint32_t id = (uint32_t)pool_size++; // use the value, then increment it
        table.emplace(std::move(key), id); // store without copying
        return id;
    }

    size_t size() const { return pool_size; }

private:
    unordered_map<string, uint32_t> table;
    size_t pool_size = 0;
};

// two structures to define the key type used in hash maps
// K2: for greedy and lickety cache (subproblem, depth)
// K3: tries (subproblem, depth, budget)
// define equality with operator== and how to hash the keys for unordered_map

struct K2 {
    uint64_t k; // hash or interned-id
    int depth;
    bool operator==(const K2& o) const { return k == o.k && depth == o.depth; } // element-wise equality
    struct Hash { // custom hash
        size_t operator()(const K2& x) const noexcept {
            size_t h = (size_t)x.k;
            size_t d = (size_t)x.depth;
            h ^= d + 0x9e3779b97f4a7c15ULL + (h<<6) + (h>>2);
            return h;
        }
    };
};

struct K3 {
    uint64_t k; // hash or interned-id
    int depth;
    int budget;
    bool operator==(const K3& o) const { return k == o.k && depth == o.depth && budget == o.budget; }
    struct Hash {
        size_t operator()(const K3& x) const noexcept {
            size_t h = (size_t)x.k;
            size_t d = (size_t)x.depth;
            size_t b = (size_t)x.budget;
            h ^= d + 0x9e3779b97f4a7c15ULL + (h<<6) + (h>>2);
            h ^= b + 0x9e3779b97f4a7c15ULL + (h<<6) + (h>>2);
            return h;
        }
    };
};

struct HistEntry {
    int obj;
    uint64_t cnt;
};
static inline bool hist_less(const HistEntry& a, const HistEntry& b){ return a.obj < b.obj; } // helper for sorting

struct TreeTrieNode; // fwd

struct LeafNode {
    int prediction; // 0/1 (kept for completeness)
    int loss;       // lamN + miscls
};

struct SplitNode {
    int feature = -1;
    shared_ptr<TreeTrieNode> left;
    shared_ptr<TreeTrieNode> right;
    uint64_t num_valid_trees = 0; // trees contributed by this split under parent's budget
};

struct TreeTrieNode {
    int budget = 0;
    int min_objective = numeric_limits<int>::max();
    vector<LeafNode> leaves; // (prediction,loss) for as many [<=2 in binary classification] if within budget
    vector<SplitNode> splits; // stores splitnodes which have the feature they split on and left and right trienodes
    vector<HistEntry> hist; // sorted ascending by obj; counts aggregated (obj, count) are elements

    uint64_t count_trees() const {
        uint64_t s = 0;
        for (const auto& e : hist) s += e.cnt;
        return s;
    }

    void add_hist(int obj, uint64_t add_cnt = 1) {
        auto it = lower_bound(hist.begin(), hist.end(), HistEntry{obj,0}, hist_less); // find the first poisition in hist where obj could be inserted without breaking sort order
        if (it != hist.end() && it->obj == obj) it->cnt += add_cnt; // if it already exists, just increment
        else hist.insert(it, HistEntry{obj, add_cnt}); // otherwise, add
        if (obj < min_objective) min_objective = obj; // keep min_objective fresh
    }

    void add_leaf(int prediction, int loss) { // assumes you call within budget
        leaves.push_back(LeafNode{prediction, loss});
        add_hist(loss, 1);
    }

    void add_split(int feat,
                   const shared_ptr<TreeTrieNode>& L,
                   const shared_ptr<TreeTrieNode>& R) {
        SplitNode s;
        s.feature = feat;
        s.left  = L;
        s.right = R;

        if (L && R) {
            int min_sum = (L->min_objective == numeric_limits<int>::max() ||
                           R->min_objective == numeric_limits<int>::max())
                          ? numeric_limits<int>::max()
                          : (L->min_objective + R->min_objective);
            if (min_sum < min_objective) min_objective = min_sum;
        }

        if ((L && !L->hist.empty()) && (R && !R->hist.empty())) {
            unordered_map<int, uint64_t> sum_counts; // make a temporary map to map obj -> count until we know how they distribute in full to then transfer to the vector-based histogram
            sum_counts.reserve(L->hist.size() * 2); // 2x is a good starting estimate

            uint64_t valid = 0;
            vector<int> R_objs; R_objs.reserve(R->hist.size()); // split (obj, cnt) into two parallel ararys, just for R due to the binary search needs in the future
            vector<uint64_t> R_cnts; R_cnts.reserve(R->hist.size());
            for (auto &e : R->hist) { R_objs.push_back(e.obj); R_cnts.push_back(e.cnt); }

            for (const auto& le : L->hist) {
                if (le.obj > budget) break; // should never happen by invariant but if we do lossy caching/more heuristics
                int rem = budget - le.obj; // R cannot exceed
                auto it_end = upper_bound(R_objs.begin(), R_objs.end(), rem); // find the first index strictly greater than rim
                int idx_end = (int)distance(R_objs.begin(), it_end); // gets the index of it_end (it_end is an iterator)
                for (int j = 0; j < idx_end; ++j) { // go until the last index that doesn't exceed rem
                    int tot = le.obj + R_objs[j];
                    uint64_t addc = le.cnt * R_cnts[j];
                    sum_counts[tot] += addc;
                    valid += addc;
                }
            }
            // we've updated our temporary sum_counts map, now we must merge it into the existing histogram
            if (!sum_counts.empty()) {
                vector<HistEntry> tmp; tmp.reserve(sum_counts.size());
                for (auto &kv : sum_counts) tmp.push_back(HistEntry{kv.first, kv.second}); // back the (obj, count) format and sorting
                sort(tmp.begin(), tmp.end(), hist_less);

                // now, we have to aggregate this into the histogram for all splits at that node
                vector<HistEntry> merged; merged.reserve(hist.size() + tmp.size());
                // simply merge two sorted lists into a new list and swap it in
                size_t i=0, j=0;
                while (i < hist.size() && j < tmp.size()) {
                    if (hist[i].obj < tmp[j].obj) merged.push_back(hist[i++]);
                    else if (tmp[j].obj < hist[i].obj) merged.push_back(tmp[j++]);
                    else { merged.push_back(HistEntry{hist[i].obj, hist[i].cnt + tmp[j].cnt}); ++i; ++j; }
                }
                while (i < hist.size()) merged.push_back(hist[i++]);
                while (j < tmp.size()) merged.push_back(tmp[j++]);
                hist.swap(merged);
            }
            s.num_valid_trees = valid;
        }

        splits.push_back(std::move(s)); // adding this split information to the trienode
    }
};

class LicketyRESPLIT {
public:
    enum class KeyMode { HASH64, EXACT };

private:
    int n_samples = 0;
    int n_features = 0;
    int n_words = 0;
    uint64_t tail_mask = ~0ULL; // to clear high bits in last word
    int lamN = 0;

    int best_objective = 0;
    int obj_bound = 0;

    vector<Packed> X_bits; // vector of Packed, each Packed is a feature column. packed is a sequence of 64-bit words where each bit corresponds to the row value for the column
    Packed Ypos; // each bit of a word is the label for the row

    KeyMode key_mode = KeyMode::HASH64; // will change later in fit
    bool trie_cache_enabled = true;
    MaskIdTable mask_ids; // used only if in exact mode

    unordered_map<K2, int, K2::Hash> greedy_cache;
    unordered_map<K2, int, K2::Hash> lickety_cache;
    unordered_map<K3, shared_ptr<TreeTrieNode>, K3::Hash> trie_cache; // if trie_cache_enabled is on

    // bitwise and of the bitvectors represented as lists of words. makes a sparser list of words
    inline void and_bits(const Packed& a, const Packed& b, Packed& out) const {
        const int m = n_words;
        for (int i = 0; i < m; ++i) out.w[i] = a.w[i] & b.w[i]; //.w[i] is a word, doing bitwise ands on words for all words
        out.w[m-1] &= tail_mask; // bitwise and out.w[m-1] & tail_mask. tail_mask is 1 iff valid, so it is setting invalid bits to 0.
    }
    // marks the samples active in a but not in b.
    inline void andnot_bits(const Packed& a, const Packed& b, Packed& out) const {
        const int m = n_words;
        for (int i = 0; i < m; ++i) out.w[i] = a.w[i] & ~b.w[i];
        out.w[m-1] &= tail_mask;
    }
    // return the number of 1 bits in the intersection a & b
    inline int popcount_and(const Packed& a, const Packed& b) const {
        int s = 0;
        for (int i = 0; i < n_words; ++i) s += popcnt64(a.w[i] & b.w[i]);
        return s;
    }

    inline uint64_t key_of_mask(const Packed& mask) {
        if (key_mode == KeyMode::HASH64) {
            return hash_mask64(mask.w.data(), n_words, tail_mask);
        } else {
            return (uint64_t)mask_ids.intern(mask, n_words, tail_mask); // cast 32->64
        }
    }

    inline int count_total(const Packed& mask) const { return mask.count(); } // number of active samples
    inline int count_pos(const Packed& mask) const { return popcount_and(mask, Ypos); } // number of active samples that are positive

    static inline double entropy(double p) {
        const double eps = 1e-12;
        p = max(eps, min(1.0 - eps, p));
        return -(p * log2(p) + (1.0 - p) * log2(1.0 - p));
    }

public:
    shared_ptr<TreeTrieNode> result;

    void set_key_mode(KeyMode m) { key_mode = m; }
    void set_trie_cache_enabled(bool on) { trie_cache_enabled = on; }

    void fit(const vector<vector<bool>>& X_col_major, const vector<int>& y,
             double lambda, int depth_budget, double rashomon_mult) {
        n_features = (int)X_col_major.size();
        n_samples  = (int)X_col_major[0].size();
        n_words = (n_samples + 63) / 64; // 64 -> 1, 65 -> 2
        tail_mask = (n_samples % 64) ? ((1ULL << (n_samples % 64)) - 1ULL) : ~0ULL; // if multiple of 64, all 1s. otherwise, n_samples % 64 1s followed by 0s.
        lamN = (int)llround(lambda * (double)n_samples);

        X_bits.assign(n_features, Packed(n_words)); // length n_features with entries of Packed, initialized to all 0s bits, we will set below.
        for (int f = 0; f < n_features; ++f) {
            auto &col = X_bits[f].w; // reference to the array of 64-bit words for that feature
            for (int i = 0; i < n_samples; ++i) {
                if (X_col_major[f][i]) col[i>>6] |= (1ULL << (i & 63)); // i>>6 integer division by 64 to answer what word are we in. then i & 63 = i % 64 to get index within word. (1ULL << (i & 63)) creates a 64-bit mask with exactly one bit = 1 at the position you need, then do bitwise or with ol[i >> 6] to set the position to 1 if it is not already set.
            }
            col[n_words-1] &= tail_mask; // bitwise and to 0 out invalid
        }

        Ypos = Packed(n_words);
        for (int i = 0; i < n_samples; ++i) {
            if (y[i]) Ypos.w[i>>6] |= (1ULL << (i & 63));
        }
        Ypos.w[n_words-1] &= tail_mask;

        Packed root(n_words);
        for (int i = 0; i < n_words-1; ++i) root.w[i] = ~0ULL; // not 0, so all 1.
        root.w[n_words-1] = tail_mask; // enforce 0s for out of scope

        best_objective = lickety_split(root, depth_budget);
        cout << "Best objective: " << best_objective
             << " (" << (double)best_objective / (double)n_samples << ")\n";

        obj_bound = (int)llround((double)best_objective * (1.0 + rashomon_mult));
        cout << "Objective bound: " << obj_bound << "\n";

        result = construct_trie(root, depth_budget, obj_bound);

        cout << "Found " << result->count_trees() << " trees\n";
        cout << "Minimum objective: " << result->min_objective << "\n";
        cout << "Cache sizes - Greedy: " << greedy_cache.size()
             << ", Lickety: " << lickety_cache.size()
             << ", Trie: " << trie_cache.size();
        if (key_mode == KeyMode::EXACT) {
            cout << ", Unique masks: " << mask_ids.size();
        }
        cout << ", Trie cache: " << (trie_cache_enabled ? "ON" : "OFF");
        cout << "\n";
    }

private:
    shared_ptr<TreeTrieNode> construct_trie(const Packed& mask, int depth, int budget) {
        const uint64_t k = key_of_mask(mask);
        K3 key{k, depth, budget};

        if (trie_cache_enabled) {
            if (auto it = trie_cache.find(key); it != trie_cache.end()) {
                return it->second; // exact lookup for simplicity
            }
        }

        auto node = make_shared<TreeTrieNode>(); // wraps in shared pointer so memory management is automatic
        node->budget = budget;

        const int n_sub = count_total(mask);
        if (n_sub == 0) {
            return node;
        }

        const int pos = count_pos(mask);

        // braces make temporary local scope so variables created will cease to persist
        {
            int mis0 = pos;
            int mis1 = n_sub - pos;
            int cost0 = lamN + mis0;
            int cost1 = lamN + mis1;
            if (cost0 <= budget) node->add_leaf(0, cost0);
            if (cost1 <= budget) node->add_leaf(1, cost1);
        }

        if (depth == 0 || budget < 2 * lamN) {
            if (trie_cache_enabled) trie_cache.emplace(key, node);
            return node;
        }

        Packed L(n_words), R(n_words);

        for (int f = 0; f < n_features; ++f) {
            and_bits(mask, X_bits[f], L);
            andnot_bits(mask, X_bits[f], R);

            if (!L.any() || !R.any()) continue;

            int lossL = lickety_split(L, depth - 1);
            int lossR = lickety_split(R, depth - 1);
            if (lossL + lossR > budget) continue;

            auto LR = multipass_long(lossL, lossR, L, R, budget, depth); // while loop until converges

            if (!LR.first || !LR.second || LR.first->hist.empty() || LR.second->hist.empty()) // safeguard, especially needed if we allow non-injective keys
                continue;

            node->add_split(f, LR.first, LR.second); // add split with left and right subtries
        }

        if (trie_cache_enabled) trie_cache.emplace(key, node);
        return node;
    }

    // returns left and righ treetrienode. the left and right mask are constants, even as you recurse on construct_trie
    pair<shared_ptr<TreeTrieNode>, shared_ptr<TreeTrieNode>>
    multipass_long(int loss_l, int loss_r,
                   const Packed& Lmask, const Packed& Rmask,
                   int budget, int depth) {
        int left_budget  = budget - loss_r;
        shared_ptr<TreeTrieNode> left_node =
            (left_budget >= 0) ? construct_trie(Lmask, depth - 1, left_budget)
                               : nullptr; // handles some potential issues with non-injective keys
        int min_left = (left_node ? left_node->min_objective : numeric_limits<int>::max());

        int right_budget = (min_left == numeric_limits<int>::max()) ? -1 : (budget - min_left);
        shared_ptr<TreeTrieNode> right_node =
            (right_budget >= 0) ? construct_trie(Rmask, depth - 1, right_budget)
                                : nullptr;
        int min_right = (right_node ? right_node->min_objective : numeric_limits<int>::max());

        while (true) {
            bool improved = false;

            int new_left_budget = (min_right == numeric_limits<int>::max()) ? -1 : (budget - min_right);
            if (new_left_budget > left_budget) {
                left_budget = new_left_budget;
                if (left_budget >= 0) {
                    left_node = construct_trie(Lmask, depth - 1, left_budget);
                    int new_min_left = left_node->min_objective;
                    if (new_min_left < min_left) min_left = new_min_left;
                }
            }

            int new_right_budget = (min_left == numeric_limits<int>::max()) ? -1 : (budget - min_left);
            if (new_right_budget > right_budget) {
                right_budget = new_right_budget;
                if (right_budget >= 0) {
                    right_node = construct_trie(Rmask, depth - 1, right_budget);
                    int new_min_right = right_node->min_objective;
                    if (new_min_right < min_right) { min_right = new_min_right; improved = true; }
                }
            }

            if (!improved) break;
        }

        return {left_node, right_node};
    }

    int train_greedy(const Packed& mask, int depth_budget) {
        const uint64_t k = key_of_mask(mask);
        K2 key{k, depth_budget};
        if (auto it = greedy_cache.find(key); it != greedy_cache.end()) return it->second;

        const int n_sub = count_total(mask);
        if (n_sub == 0) {
            return 0; // should never happen
        }

        const int pos = count_pos(mask);
        const int leaf_loss = lamN + min(pos, n_sub - pos);

        if (depth_budget <= 0 || leaf_loss <= 2 * lamN) {
            greedy_cache.emplace(key, leaf_loss);
            return leaf_loss;
        }

        // choose split via entropy gain
        int best_feat = find_best_split(mask);
        if (best_feat < 0) { // this should also never happen
            greedy_cache.emplace(key, leaf_loss);
            return leaf_loss;
        }

        Packed L(n_words), R(n_words);
        and_bits(mask, X_bits[best_feat], L);
        andnot_bits(mask, X_bits[best_feat], R);
        if (!L.any() || !R.any()) { // this should also never happen if no error is thrown with find best split
            greedy_cache.emplace(key, leaf_loss);
            return leaf_loss;
        }

        int left_obj  = train_greedy(L, depth_budget - 1);
        int right_obj = train_greedy(R, depth_budget - 1);
        int split_obj = left_obj + right_obj;

        int ans = min(leaf_loss, split_obj);
        greedy_cache.emplace(key, ans);
        return ans;
    }

    // hardcoded k=1 lookahead
    int lickety_split(const Packed& mask, int depth_budget) {
        const uint64_t k = key_of_mask(mask);
        K2 key{k, depth_budget};
        if (auto it = lickety_cache.find(key); it != lickety_cache.end()) return it->second;

        if (depth_budget <= 0) {
            int v = train_greedy(mask, 0);
            return v;
        }

        const int n_sub = count_total(mask);
        const int pos   = count_pos(mask);
        const int leaf_loss = lamN + min(pos, n_sub - pos);
        if (leaf_loss <= 2 * lamN) {
            lickety_cache.emplace(key, leaf_loss);
            return leaf_loss;
        }

        int best_feat = -1;
        int best_sum  = numeric_limits<int>::max();

        Packed L(n_words), R(n_words), bestL(n_words), bestR(n_words);

        for (int f = 0; f < n_features; ++f) {
            and_bits(mask, X_bits[f], L);
            andnot_bits(mask, X_bits[f], R);
            if (!L.any() || !R.any()) continue;

            int sum = train_greedy(L, depth_budget - 1) + train_greedy(R, depth_budget - 1);
            if (sum < best_sum) {
                best_sum = sum;
                best_feat = f;
                bestL.w = L.w; // deep element-by-element copy
                bestR.w = R.w;
            }
        }

        // int ans;
        // if (best_feat < 0 || best_sum >= leaf_loss) {
        //     ans = leaf_loss;
        // } else {
        //     int left_loss  = lickety_split(bestL, depth_budget - 1);
        //     int right_loss = lickety_split(bestR, depth_budget - 1);
        //     ans = left_loss + right_loss;
        // }
        int ans = leaf_loss; 
        if (best_feat >= 0) {
            int left_loss  = lickety_split(bestL, depth_budget - 1);
            int right_loss = lickety_split(bestR, depth_budget - 1);
            ans = min(ans, left_loss + right_loss); // do lickety even if leaf is better over greedy and see which is preferred
        }
        lickety_cache.emplace(key, ans);
        return ans;
    }

    int find_best_split(const Packed& mask) const {
        const int n_sub = count_total(mask);
        if (n_sub <= 1) return -1;

        const int pos_total = count_pos(mask);
        const double p0 = (double)pos_total / (double)n_sub;
        const double baseH = entropy(p0);

        int best_f = -1;
        double best_gain = -1e300;

        Packed L(n_words);
        for (int f = 0; f < n_features; ++f) {
            for (int i = 0; i < n_words; ++i) L.w[i] = mask.w[i] & X_bits[f].w[i];
            L.w[n_words-1] &= tail_mask;

            const int left_n = L.count();
            const int right_n = n_sub - left_n;
            if (left_n == 0 || right_n == 0) continue;

            const int left_pos  = popcount_and(L, Ypos);
            const int right_pos = pos_total - left_pos;

            const double wl = (double)left_n  / (double)n_sub;
            const double wr = (double)right_n / (double)n_sub;
            const double pl = (double)left_pos  / (double)left_n;
            const double pr = (double)right_pos / (double)right_n;

            const double gain = baseH - (wl*entropy(pl) + wr*entropy(pr));
            if (gain > best_gain) { best_gain = gain; best_f = f; }
        }
        return best_f;
    }
};

extern "C" {
    LicketyRESPLIT* create_model() {
        return new LicketyRESPLIT();
    }

    void delete_model(LicketyRESPLIT* model) {
        delete model;
    }

    void fit_model(LicketyRESPLIT* model,
                      const uint8_t* X_data, int n_samples, int n_features,
                      const int* y_data,
                      double lambda, int depth, double rashomon_mult,
                      int key_mode /*0=hash,1=exact*/,
                      int trie_cache_enabled /*0=off, 1=on*/) {
        vector<vector<bool>> X(n_features, vector<bool>(n_samples));
        for (int f = 0; f < n_features; ++f) {
            const uint8_t* col = X_data + (size_t)f * (size_t)n_samples;
            for (int i = 0; i < n_samples; ++i) X[f][i] = (col[i] != 0);
        }
        vector<int> y(y_data, y_data + n_samples);
        if (key_mode == 1) model->set_key_mode(LicketyRESPLIT::KeyMode::EXACT);
        else               model->set_key_mode(LicketyRESPLIT::KeyMode::HASH64);
        model->set_trie_cache_enabled(trie_cache_enabled != 0);
        model->fit(X, y, lambda, depth, rashomon_mult);
    }

    uint64_t get_tree_count(LicketyRESPLIT* model) {
        return model->result ? model->result->count_trees() : 0ULL;
    }

    int get_min_objective(LicketyRESPLIT* model) {
        return model->result ? model->result->min_objective : numeric_limits<int>::max();
    }
}
