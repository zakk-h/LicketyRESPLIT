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
#include <stdexcept>
#include <unordered_set>

using namespace std;

#if defined(_MSC_VER)
  #include <intrin.h>
  static inline int popcnt64(uint64_t x) {
      return static_cast<int>(__popcnt64(x));
  }
#else
  static inline int popcnt64(uint64_t x) {
      return __builtin_popcountll(x);
  }
#endif

using Lit = uint16_t; // 16-bit literal = 2*feat + sign
using PathKey = std::vector<Lit>;

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

// 2*feat + sign
static inline Lit enc_lit(int feat, int sign01) {
    // requires feat <= 32767 so that (feat<<1 | sign) fits in 16 bits
    return (Lit)((feat << 1) | (sign01 & 1));
}

static inline const PathKey& empty_pk() {
    static const PathKey k;
    return k;
}

// insert literal into PathKey, maintaining sorted canonical order
static inline void pk_insert_sorted(PathKey& pk, Lit lit) {
    auto it = std::lower_bound(pk.begin(), pk.end(), lit);
    pk.insert(it, lit);
}

// remove literal from PathKey (must exist)
static inline void pk_erase_sorted(PathKey& pk, Lit lit) {
    auto it = std::lower_bound(pk.begin(), pk.end(), lit);
    pk.erase(it);
}


// used for exact, non-probabilistic keyks at the expense of more memory. we intern the exact bytes of a mask/bitvector and assign a small integer ID.
// first unique mask id 0, second unique mask id 1 and so on.
class MaskIdTable {
public:
    uint32_t intern(const Packed& mask, int n_words, uint64_t tail_mask) {
        const size_t bytes = (size_t)n_words * sizeof(uint64_t); // constant across the dataset, how many words needed * 64 bit length
        string key;
        key.resize(bytes);
        // uint64_t* out = reinterpret_cast<uint64_t*>(&key[0]); // pointer to the start of key
        for (int i = 0; i < n_words; ++i) {
            uint64_t x = mask.w[i];
            if (i == n_words - 1) x &= tail_mask; // the last word may have padding bits, tail_mask zeroes out the unused bits.
            // out[i] = x; // the byte representation of mask.w - we need to convert to use as a key in the unordered map
            std::memcpy(&key[i * sizeof(uint64_t)], &x, sizeof(uint64_t));
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

class LitIdTable {
public:
    uint32_t intern(const std::vector<Lit>& lits) {
        const size_t bytes = lits.size() * sizeof(Lit);
        std::string key;
        key.resize(bytes);
        if (bytes) std::memcpy(&key[0], lits.data(), bytes);

        auto it = table.find(key);
        if (it != table.end()) return it->second;
        uint32_t id = (uint32_t)pool_size++;
        table.emplace(std::move(key), id);
        return id;
    }


    size_t size() const { return pool_size; }

private:
    std::unordered_map<std::string, uint32_t> table;
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

struct KLA {
    uint64_t k;
    int depth;
    int la; // lookahead used for this call
    bool operator==(const KLA& o) const { return k == o.k && depth == o.depth && la == o.la; }
    struct Hash {
        size_t operator()(const KLA& x) const noexcept {
            size_t h = (size_t)x.k;
            size_t d = (size_t)x.depth;
            size_t a = (size_t)x.la;
            h ^= d + 0x9e3779b97f4a7c15ULL + (h<<6) + (h>>2);
            h ^= a + 0x9e3779b97f4a7c15ULL + (h<<6) + (h>>2);
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
    bool hist_built = false; // wait until the end to build the histograms because we don't know if they'll be used in the final trie because of multipass

    uint64_t count_trees() const {
        ensure_hist_built();
        uint64_t s = 0;
        for (const auto& e : hist) s += e.cnt;
        return s;
    }

    uint64_t count_leq(int objective) const {
        ensure_hist_built();
        if (hist.empty()) return 0ULL;
        uint64_t total = 0ULL;
        for (const auto& e : hist) {
            if (e.obj > objective) break; // hist is sorted ascending by obj
            total += e.cnt;
        }
        return total;
    }

    void add_hist(int obj, uint64_t add_cnt = 1) {
        auto it = lower_bound(hist.begin(), hist.end(), HistEntry{obj,0}, hist_less); // find the first poisition in hist where obj could be inserted without breaking sort order
        if (it != hist.end() && it->obj == obj) it->cnt += add_cnt; // if it already exists, just increment
        else hist.insert(it, HistEntry{obj, add_cnt}); // otherwise, add
        if (obj < min_objective) min_objective = obj; // keep min_objective fresh
    }
    
    void add_leaf(int prediction, int loss) {
        leaves.push_back(LeafNode{prediction, loss});
        if (loss < min_objective) min_objective = loss;
    }
    
    void add_leaf_and_build(int prediction, int loss) { // assumes you call within budget
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
        s.num_valid_trees = 0; // will be filled in post-processing

        if (L && R) {
            int min_sum = (L->min_objective == numeric_limits<int>::max() ||
                        R->min_objective == numeric_limits<int>::max())
                        ? numeric_limits<int>::max()
                        : (L->min_objective + R->min_objective);
            if (min_sum < min_objective) min_objective = min_sum;
        }
        splits.push_back(std::move(s));
    }

    void add_split_and_build(int feat,
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

    // post-process the trie to build per-node histograms using the existing helpers.
    // assumes leaves/splits/min_objective/budget are already set by construct_trie.
    static void build_histograms_post(TreeTrieNode* node) {
        if (!node || node->hist_built) return;

        //nsur ee children are processed first (post-order)
        for (auto &s : node->splits) {
            if (s.left)  build_histograms_post(s.left.get());
            if (s.right) build_histograms_post(s.right.get());
        }

        // rebuild this node's histogram from scratch
        std::vector<SplitNode> saved = std::move(node->splits);
        node->splits.clear();
        node->hist.clear();
        node->hist_built = false; // (will set true at end)

        // add leaf contributions
        for (const auto &leaf : node->leaves) {
            node->add_hist(leaf.loss, 1); // could call add_leaf_and_build but that is overkill here
        }

        // re-add splits, letting add_split_and_build do the heavy lifting:
        // merges L/R histograms into node->hist
        // computes s.num_valid_trees
        // refreshes min_objective though that isn't needed
        for (auto &s : saved) {
            node->add_split_and_build(s.feature, s.left, s.right);
        }

        node->hist_built = true;
    }

    void ensure_hist_built() const {
        if (!hist_built) {
            TreeTrieNode::build_histograms_post(const_cast<TreeTrieNode*>(this));
        }
    }

};

struct PredNode {
    int feature;  // -1 for leaf
    int prediction; // only meaningful if feature == -1
    shared_ptr<PredNode> left;
    shared_ptr<PredNode> right;
};

// for joint rashomon set prediction / rid
struct ObjBucket {
    int obj;
    std::vector<Packed> preds; // each is a prediction bitvector for one tree at this obj. predictions for all trees with an objective.
};

struct EvalCtx {
    int n_eval = 0;
    int n_words = 0;
    uint64_t tail_mask = ~0ULL;
    std::vector<Packed> X_bits_eval; // everything needed for evaluation dataset
};


class LicketyRESPLIT {
public:
    enum class KeyMode { HASH64, EXACT, LITS_EXACT };

private:
    int n_samples = 0;
    int n_features = 0;
    int n_words = 0;
    uint64_t tail_mask = ~0ULL; // to clear high bits in last word
    int lamN = 0;

    int best_objective = 0;
    int obj_bound = 0;

    double multiplicative_slack = 0.0;

    vector<Packed> X_bits; // vector of Packed, each Packed is a feature column. packed is a sequence of 64-bit words where each bit corresponds to the row value for the column
    Packed Ypos; // each bit of a word is the label for the row

    KeyMode key_mode = KeyMode::HASH64; // will change later in fit
    bool trie_cache_enabled = false;
    bool proxy_caching_enabled = true;
    MaskIdTable mask_ids; // used only if in exact mode
    LitIdTable lit_ids; // for itemset mode

    int lookahead_init = 1; // will be changed later
    bool use_multipass = true; // sim
    bool rule_list_mode = false;
    bool majority_leaf_only = false;
    bool cache_cheap_subproblems = false;
    int greedy_split_mode = 1;
    int num_proxy_features = -1; // <=0 means use all feature. positive for feature selection

    
    int oracle_style = 0; // 0=constant k (current), 1=cyclic, 2=cyclic-consistent, 3=split
    std::vector<int> k_at_depth; // size depth_budget (only used for style 2)

    unordered_map<K2, int, K2::Hash> greedy_cache;
    unordered_map<K2,  int, K2::Hash>  lickety_cache_k2; // used when lookahead_init <= 1
    unordered_map<KLA, int, KLA::Hash>  lickety_cache_kla; // used when lookahead_init > 1

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
        if (key_mode == KeyMode::LITS_EXACT) {
            throw std::runtime_error("key_of_mask called in LITS_EXACT mode; use key_of_state(mask, pk)");
        }
        return key_of_state(mask, empty_pk());
    }



    inline uint64_t key_of_state(const Packed& mask, const PathKey& pk) {
        switch (key_mode) {
            case KeyMode::HASH64:
                return hash_mask64(mask.w.data(), n_words, tail_mask);
            case KeyMode::EXACT:
                return (uint64_t)mask_ids.intern(mask, n_words, tail_mask); // cast 32->64
            case KeyMode::LITS_EXACT:
                return (uint64_t)lit_ids.intern(pk);
        }
        return 0;
    }

    inline uint64_t key_of_subproblem(const Packed& mask, const PathKey& pk) {
        if (key_mode == KeyMode::LITS_EXACT) {
            return key_of_state(mask, pk); // interns pk
        } else {
            return key_of_mask(mask); 
        }
    }

    inline int proxy_feat_count_() const {
        return (num_proxy_features > 0) ? std::min(num_proxy_features, n_features) : n_features;
    }


    inline bool use_kla_cache() const { return lookahead_init > 1; }
    
    inline int count_total(const Packed& mask) const { return mask.count(); } // number of active samples
    inline int count_pos(const Packed& mask) const { return popcount_and(mask, Ypos); } // number of active samples that are positive

    static inline double entropy(double p) {
        const double eps = 1e-12;
        p = max(eps, min(1.0 - eps, p));
        return -(p * log2(p) + (1.0 - p) * log2(1.0 - p));
    }

    // count the distinct subproblems/bitvectors/literals/fingerprints, not considering depth or greedy/lickety
    size_t count_distinct_subproblems_union() const {
        std::unordered_set<uint64_t> seen;

        const size_t lsz = use_kla_cache() ? lickety_cache_kla.size() : lickety_cache_k2.size();
        const size_t approx = greedy_cache.size() + lsz;
        seen.reserve(approx * 2 + 16);

        // greedy: key is (k, depth). we ignore depth by inserting only k.
        for (const auto& kv : greedy_cache) {
            seen.insert(kv.first.k);
        }

        // lickety: either (k, depth) or (k, depth, la). ignore depth/la by inserting only k.
        if (use_kla_cache()) {
            for (const auto& kv : lickety_cache_kla) {
                seen.insert(kv.first.k);
            }
        } else {
            for (const auto& kv : lickety_cache_k2) {
                seen.insert(kv.first.k);
            }
        }

        return seen.size();
    }

    

public:
    shared_ptr<TreeTrieNode> result;

    void set_key_mode(KeyMode m) { key_mode = m; }
    void set_trie_cache_enabled(bool on) { trie_cache_enabled = on; }
    void set_multiplicative_slack(double s) { multiplicative_slack = s; }
    void set_use_multipass(bool on) { use_multipass = on; }
    void set_rule_list_mode(bool on) { rule_list_mode = on; }
    void set_majority_leaf_only(bool on) { majority_leaf_only = on; }
    void set_cache_cheap_subproblems(bool on) { cache_cheap_subproblems = on; }
    void set_greedy_split_mode(int m) { greedy_split_mode = m; }
    void set_proxy_caching_enabled(bool on) { proxy_caching_enabled = on; }

    void fit(const std::vector<std::vector<bool>>& X_col_major,
             const std::vector<int>& y,
             double lambda,
             int depth_budget,
             double rashomon_mult,
             int lookahead_k,
             int root_budget,
             bool use_multipass_flag,
             bool rule_list_mode_flag,
             int oracle_style_in,
             bool majority_leaf_only_flag,
             bool cache_cheap_subproblems_flag,
             bool proxy_caching_flag,
             int num_proxy_features_in
            ) {
        n_features = (int)X_col_major.size();
        n_samples  = (int)X_col_major[0].size();
        n_words = (n_samples + 63) / 64; // 64 -> 1, 65 -> 2
        tail_mask = (n_samples % 64) ? ((1ULL << (n_samples % 64)) - 1ULL) : ~0ULL; // if multiple of 64, all 1s. otherwise, n_samples % 64 1s followed by 0s.
        lamN = (int)llround(lambda * (double)n_samples);
        lookahead_init = lookahead_k;
        use_multipass = use_multipass_flag;
        rule_list_mode = rule_list_mode_flag;
        majority_leaf_only = majority_leaf_only_flag;
        cache_cheap_subproblems = cache_cheap_subproblems_flag;
        oracle_style = oracle_style_in;
        proxy_caching_enabled = proxy_caching_flag;
        if (num_proxy_features_in <= 0) num_proxy_features = n_features;
        else num_proxy_features = std::min(num_proxy_features_in, n_features);

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

        const PathKey& root_pk = empty_pk();

        if (root_budget >= 0) {
            // user-specified bound: skip reference solution
            obj_bound = root_budget;
            std::cout << "Objective bound (user-set): " << obj_bound << "\n";

        } else {
            if (lookahead_init <= 0) { // set based on greedy even if our proxy is a leaf
                best_objective = train_greedy(root, depth_budget, root_pk);
            } else {
                best_objective = lickety_split(root, depth_budget, lookahead_init, root_pk);
            }
            cout << "Best objective: " << best_objective
                << " (" << (double)best_objective / (double)n_samples << ")\n";

            obj_bound = (int)llround(best_objective * (1.0 + rashomon_mult) * (1.0 + multiplicative_slack));

            cout << "Objective bound: " << obj_bound << "\n";
        }

        if (oracle_style == 2) { // we define to be depth_budget+1 size but depth 0 doesn't actually matter
            k_at_depth.assign(depth_budget + 1, 1);
            int K = lookahead_init;
            int kk = K;
            for (int d = depth_budget; d >= 0; --d) {
                k_at_depth[d] = std::min(d, kk);
                kk = (kk > 1) ? (kk - 1) : K; // increment down then wrap
            }
        } else {
            k_at_depth.clear();
        }


        result = construct_trie(root, depth_budget, obj_bound, root_pk);

        // cout << "Found " << result->count_trees() << " trees\n"; // we'll let the user compute this query if they want it because it is somewhat expensive
        cout << "Minimum objective: " << result->min_objective << "\n";
        cout << "Cache sizes - Greedy: " << greedy_cache.size()
            << ", Lickety: " << (use_kla_cache() ? lickety_cache_kla.size() : lickety_cache_k2.size())
            << ", Trie: " << trie_cache.size();
        // cout << ", Distinct subproblems (greedy U lickety): " << count_distinct_subproblems_union();
        // if (key_mode == KeyMode::EXACT) {
        //     cout << ", Unique masks: " << mask_ids.size();
        // }
        // if (key_mode == KeyMode::LITS_EXACT) {
        //     cout << ", Unique literal subproblems: " << lit_ids.size();
        // }
        cout << ", Trie cache: " << (trie_cache_enabled ? "ON" : "OFF");
        cout << "\n";
    }

    // predict using the i-th tree in the Rashomon set: X_row_major: binary [n_samples][n_features]
    std::vector<uint8_t> get_predictions(uint64_t tree_index, const std::vector<std::vector<uint8_t>>& X_row_major) const {
        if (!result) {
            throw std::runtime_error("No Rashomon trie has been constructed. Call fit() first.");
        }

        const std::size_t n_samples = X_row_major.size();
        if (n_samples == 0) return {};

        const std::size_t n_features = X_row_major[0].size();
        if ((int)n_features != this->n_features) {
            throw std::runtime_error("Prediction X has different number of features than training.");
        }

        auto tree = get_ith_tree(tree_index);
        std::vector<uint8_t> out(n_samples, 0);

        std::vector<int> idx(n_samples);
        for (std::size_t i = 0; i < n_samples; ++i) {
            idx[i] = static_cast<int>(i);
        }

        predict_tree_recursive(tree.get(), X_row_major, out, idx);
        return out;
    }

    // get predictions from all trees in the rashomon set, as a vector of prediction vectors (one per tree).
    std::vector<std::vector<uint8_t>> get_all_predictions(
        const std::vector<std::vector<uint8_t>>& X_row_major
    ) const {
        uint64_t total = result ? result->count_trees() : 0ULL;
        std::vector<std::vector<uint8_t>> all;
        all.reserve(static_cast<std::size_t>(total));
        for (uint64_t i = 0; i < total; ++i) {
            all.push_back(get_predictions(i, X_row_major));
        }
        return all;
    }

    // paths[k] is a list of ints - one path per leaf; +f = went left on feature f, -f = went right. this array of a path encodes the splits in the tree to the kth leaf.
    // predictions[k] is the 0/1 label at that leaf.
    std::pair<std::vector<std::vector<int>>, std::vector<int>>
    get_tree_paths(std::uint64_t tree_index) const {
        if (!result) {
            throw std::runtime_error("No Rashomon trie has been constructed. Call fit() first.");
        }

        auto tree = get_ith_tree(tree_index);
        std::vector<std::vector<int>> paths;
        std::vector<int> preds;
        std::vector<int> current;

        collect_paths(tree.get(), current, paths, preds);
        return {paths, preds};
    }

    // return (unnormalized_objective, normalized_objective) for the ith tree
    std::pair<int, double> get_ith_tree_objective(std::uint64_t i) const {
        if (!result) {
            throw std::runtime_error("No Rashomon trie has been constructed. Call fit() first.");
        }

        result->ensure_hist_built();

        std::uint64_t total = result->count_trees();
        if (i >= total) {
            throw std::out_of_range("Tree index out of range in get_ith_tree_objective");
        }

        std::uint64_t cum = 0;
        int target_obj = -1;

        // hist is sorted by objective ascending
        for (const auto& e : result->hist) {
            if (i < cum + e.cnt) {
                target_obj = e.obj;
                break;
            }
            cum += e.cnt;
        }

        if (target_obj < 0) {
            throw std::runtime_error("Failed to locate objective bucket in get_ith_tree_objective");
        }

        double normalized = (n_samples > 0)
            ? static_cast<double>(target_obj) / static_cast<double>(n_samples)
            : 0.0;

        return {target_obj, normalized};
    }

    // (depth_from_root, frontier_sum_obj) for each internal node of a specific materialized tree
    // may be incompatible with certain subproblem ids (legacy)
    std::vector<std::pair<int,int>>
    get_tree_frontier_scores(uint64_t tree_index, int depth_budget) {
        if (!result) {
            throw std::runtime_error("No Rashomon trie has been constructed. Call fit() first.");
        }
        if (key_mode == KeyMode::LITS_EXACT) {
            throw std::runtime_error("frontier scoring not supported in LITS_EXACT");
        }

        auto tree = get_ith_tree(tree_index);

        // root mask = all samples active
        Packed root(n_words);
        for (int i = 0; i < n_words - 1; ++i) root.w[i] = ~0ULL;
        root.w[n_words - 1] = tail_mask;

        std::vector<SibEntry> sib_stack;
        std::vector<std::pair<int,int>> out;

        frontier_scores_dfs_(
            tree.get(),
            root,
            /*depth_remaining=*/depth_budget,
            /*depth_from_root=*/0,
            sib_stack,
            out
        );

        return out;
    }

    // root LicketySPLIT objective with lookahead=1 so the user can compare frontier cuts to the reference solution
    int root_lickety_objective_lookahead1(int depth_budget) {
        if (n_samples == 0) {
            throw std::runtime_error("Model not fitted.");
        }

        Packed root(n_words);
        for (int i = 0; i < n_words - 1; ++i) root.w[i] = ~0ULL;
        root.w[n_words - 1] = tail_mask;

        PathKey root_pk;
        return lickety_split(root, depth_budget, /*k=*/1, root_pk);
    }

private:
    shared_ptr<TreeTrieNode> construct_trie(const Packed& mask, int depth, int budget, const PathKey& pk) {
        const uint64_t k = key_of_subproblem(mask, pk);
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
        // {
        //     int mis0 = pos;
        //     int mis1 = n_sub - pos;
        //     int cost0 = lamN + mis0;
        //     int cost1 = lamN + mis1;
        //     if (cost0 <= budget) node->add_leaf(0, cost0);
        //     if (cost1 <= budget) node->add_leaf(1, cost1);
        // }

        {
            int mis0 = pos;
            int mis1 = n_sub - pos;

            if (!majority_leaf_only) {
                int cost0 = lamN + mis0;
                int cost1 = lamN + mis1;
                if (cost0 <= budget) node->add_leaf(0, cost0);
                if (cost1 <= budget) node->add_leaf(1, cost1);
            } else {
                // choose the majority-label leaf (i.e., the one with fewer misclassifications)
                // tie-break rule: predict 1 when mis0 == mis1
                int pred = (mis1 <= mis0) ? 1 : 0;
                int best_mis = std::min(mis0, mis1);
                int best_cost = lamN + best_mis;
                if (best_cost <= budget) node->add_leaf(pred, best_cost);
            }
        }


        if (depth == 0 || budget < 2 * lamN) {
            if (trie_cache_enabled) trie_cache.emplace(key, node);
            return node;
        }

        Packed L(n_words), R(n_words);

        const int k_here = (oracle_style == 2 && depth >= 0 && depth < (int)k_at_depth.size())
            ? k_at_depth[depth-1]
            : lookahead_init;

        for (int f = 0; f < n_features; ++f) {
            and_bits(mask, X_bits[f], L);
            andnot_bits(mask, X_bits[f], R);

            if (!L.any() || !R.any()) continue;

            // build child pks (canonical sorted)
            // pk refs default to EMPTY
            const PathKey* pkLp = &empty_pk();
            const PathKey* pkRp = &empty_pk();

            // only build PKs in LITS_EXACT
            PathKey pkL_local, pkR_local;
            if (key_mode == KeyMode::LITS_EXACT) {
                pkL_local = pk;
                pkR_local = pk;
                pk_insert_sorted(pkL_local, enc_lit(f, 1));
                pk_insert_sorted(pkR_local, enc_lit(f, 0));
                pkLp = &pkL_local;
                pkRp = &pkR_local;
            }

            int lossL, lossR;
            if (lookahead_init < 0) {
                lossL = leaf_objective(L);
                lossR = leaf_objective(R);
            } else if (lookahead_init == 0) {
                lossL = train_greedy(L, depth - 1, *pkLp);
                lossR = train_greedy(R, depth - 1, *pkRp);
            } else {
                lossL = lickety_split(L, depth - 1, k_here, *pkLp);
                lossR = lickety_split(R, depth - 1, k_here, *pkRp);
            }
            if (!rule_list_mode) {
                if (lossL + lossR > budget) continue; // approximation decision tree rashomon set
            } else {
                if (lossL > budget - lamN && lossR > budget - lamN) continue; // exact rule list rashomon set. if you can't afford a leaf on either side, stop
            }

            std::pair<std::shared_ptr<TreeTrieNode>, std::shared_ptr<TreeTrieNode>> LR; 
            if (rule_list_mode) {
                LR = rule_list_alloc(lossL, lossR, L, R, budget, depth, *pkLp, *pkRp);
            } else if (use_multipass) {
                LR = multipass_long(lossL, lossR, L, R, budget, depth, *pkLp, *pkRp);
            } else {
                LR = singlepass_alloc(lossL, lossR, L, R, budget, depth, *pkLp, *pkRp);
            }
            if (!LR.first || !LR.second) continue; // safeguard, especially needed if we allow non-injective keys

            node->add_split(f, LR.first, LR.second); // add split with left and right subtries
        }

        if (trie_cache_enabled) trie_cache.emplace(key, node);
        return node;
    }

    // returns left and righ treetrienode. the left and right mask are constants, even as you recurse on construct_trie
    pair<shared_ptr<TreeTrieNode>, shared_ptr<TreeTrieNode>>
    multipass_long(int loss_l, int loss_r,
               const Packed& Lmask, const Packed& Rmask,
               int budget, int depth,
               const PathKey& pkL, const PathKey& pkR) {
        int left_budget  = budget - loss_r;
        shared_ptr<TreeTrieNode> left_node =
            (left_budget >= 0) ? construct_trie(Lmask, depth - 1, left_budget, pkL)
                               : nullptr; // handles some potential issues with non-injective keys
        int min_left = (left_node ? left_node->min_objective : numeric_limits<int>::max());

        int right_budget = (min_left == numeric_limits<int>::max()) ? -1 : (budget - min_left);
        shared_ptr<TreeTrieNode> right_node =
            (right_budget >= 0) ? construct_trie(Rmask, depth - 1, right_budget, pkR)
                                : nullptr;
        int min_right = (right_node ? right_node->min_objective : numeric_limits<int>::max());

        while (true) {
            bool improved = false;

            int new_left_budget = (min_right == numeric_limits<int>::max()) ? -1 : (budget - min_right);
            if (new_left_budget > left_budget) {
                left_budget = new_left_budget;
                if (left_budget >= 0) {
                    left_node = construct_trie(Lmask, depth - 1, left_budget, pkL);
                    int new_min_left = left_node->min_objective;
                    if (new_min_left < min_left) min_left = new_min_left;
                }
            }

            int new_right_budget = (min_left == numeric_limits<int>::max()) ? -1 : (budget - min_left);
            if (new_right_budget > right_budget) {
                right_budget = new_right_budget;
                if (right_budget >= 0) {
                    right_node = construct_trie(Rmask, depth - 1, right_budget, pkR);
                    int new_min_right = right_node->min_objective;
                    if (new_min_right < min_right) { min_right = new_min_right; improved = true; }
                }
            }

            if (!improved) break;
        }

        return {left_node, right_node};
    }

    // this is solely for ableation study purposes - if practical we would subtract minobjective from the other side
    std::pair<std::shared_ptr<TreeTrieNode>, std::shared_ptr<TreeTrieNode>>
    singlepass_alloc(int loss_l, int loss_r,
                 const Packed& Lmask, const Packed& Rmask,
                 int budget, int depth,
                 const PathKey& pkL, const PathKey& pkR) {
        int left_budget  = budget - loss_r;
        int right_budget = budget - loss_l;

        std::shared_ptr<TreeTrieNode> left_node  = nullptr;
        std::shared_ptr<TreeTrieNode> right_node = nullptr;

        if (left_budget >= 0) { // robustness incase we change pruning
            left_node = construct_trie(Lmask, depth - 1, left_budget, pkL);
        }
        if (right_budget >= 0) {
            right_node = construct_trie(Rmask, depth - 1, right_budget, pkR);
        }

        return {left_node, right_node};
    
    }

    std::pair<std::shared_ptr<TreeTrieNode>, std::shared_ptr<TreeTrieNode>>
    rule_list_alloc(int loss_l, int loss_r,
                const Packed& Lmask, const Packed& Rmask,
                int budget, int depth,
                const PathKey& pkL, const PathKey& pkR) {
        using std::shared_ptr;
        const int INF = std::numeric_limits<int>::max();

        int left_candidate  = budget - loss_r;
        int right_candidate = budget - loss_l;

        // if we can't even fit a leaf on either side, then we cannot yield a valid rule list
        if (left_candidate < 0 && right_candidate < 0) {
            return {nullptr, nullptr};
        }

        // decide which side to solve first:
        // if only one is non-negative, use that one.
        // if both are non-negative, pick the one with more candidate budget (doesn't matter)
        // why do we need this? because rule lists put a leaf on one side. if putting one on one side would yield a negative budget for the other side, but putting a leaf on the other side wouldn't, then we know we put the leaf on that side and solve it first.
        bool solve_left_first;
        if (left_candidate >= 0 && right_candidate < 0) {
            solve_left_first = true;
        } else if (right_candidate >= 0 && left_candidate < 0) {
            solve_left_first = false;
        } else {
            // both >= 0
            solve_left_first = (left_candidate >= right_candidate);
        }

        shared_ptr<TreeTrieNode> left_node  = nullptr;
        shared_ptr<TreeTrieNode> right_node = nullptr;

        if (solve_left_first) {
            int left_budget = left_candidate;
            if (left_budget >= 0) {
                left_node = construct_trie(Lmask, depth - 1, left_budget, pkL);
                int min_left = left_node ? left_node->min_objective : INF;

                // remaining budget for the right side is B - min_left to be more optimal than assuming leaf loss on left
                //int right_budget = (min_left == INF) ? -1 : (budget - min_left);
                int right_budget = (loss_l == INF) ? -1 : (budget - loss_l);
                if (right_budget >= 0) {
                    right_node = construct_trie(Rmask, depth - 1, right_budget, pkR);
                }
            }
        } else {
            int right_budget = right_candidate;
            if (right_budget >= 0) {
                right_node = construct_trie(Rmask, depth - 1, right_budget, pkR);
                int min_right = right_node ? right_node->min_objective : INF;

                //int left_budget = (min_right == INF) ? -1 : (budget - min_right);
                int left_budget = (loss_r == INF) ? -1 : (budget - loss_r);
                if (left_budget >= 0) {
                    left_node = construct_trie(Lmask, depth - 1, left_budget, pkL);
                }
            }
        }

        return {left_node, right_node};
    }


    int leaf_objective(const Packed& mask) const {
        const int n_sub = count_total(mask);
        if (n_sub == 0) return 0; // should not really happen
        const int pos = count_pos(mask);
        return lamN + min(pos, n_sub - pos);
    }

    inline void make_child_pks_if_needed_(
        int feat,
        const PathKey& pk,
        const PathKey*& pkLp,
        const PathKey*& pkRp,
        PathKey& pkL_local,
        PathKey& pkR_local
    ) const {
        pkLp = &empty_pk();
        pkRp = &empty_pk();
        if (key_mode == KeyMode::LITS_EXACT) {
            pkL_local = pk;
            pkR_local = pk;
            pk_insert_sorted(pkL_local, enc_lit(feat, 1));
            pk_insert_sorted(pkR_local, enc_lit(feat, 0));
            pkLp = &pkL_local;
            pkRp = &pkR_local;
        }
    }


    int train_greedy(const Packed& mask, int depth_budget, const PathKey& pk) {
        if (depth_budget == 0) {
            return leaf_objective(mask);
        }
        if (depth_budget == 1 && (greedy_split_mode == 1 || greedy_split_mode == 2)) {
            return depth1_exact_solver_cached(mask, pk); 
        }
        // const uint64_t k = key_of_subproblem(mask, pk);
        // K2 key{k, depth_budget};
        // if (auto it = greedy_cache.find(key); it != greedy_cache.end()) return it->second;
        uint64_t kmask = 0;
        K2 key{0, depth_budget};

        if (proxy_caching_enabled) {
            kmask = key_of_subproblem(mask, pk);
            key.k = kmask;
            if (auto it = greedy_cache.find(key); it != greedy_cache.end()) return it->second;
        }


        const int n_sub = count_total(mask);
        if (n_sub == 0) {
            return 0; // should never happen
        }

        const int pos = count_pos(mask);
        const int leaf_loss = lamN + min(pos, n_sub - pos);

        if (leaf_loss <= 2 * lamN) {
            if (cache_cheap_subproblems && proxy_caching_enabled) 
                greedy_cache.emplace(key, leaf_loss);
            return leaf_loss;
        }

        // decide which split-selection heuristic to use
        bool use_entropy;
        if (greedy_split_mode == 0) {
            use_entropy = true;                  // always entropy-driven
        } else if (greedy_split_mode == 1) {
            use_entropy = (depth_budget != 1);   // special depth==1 solver
        } else { // greedy_split_mode == 2
            use_entropy = false;                 // always minimize child leaf objective
        }


        // choose split via entropy gain
        int best_feat = find_best_split(mask, use_entropy);
        if (best_feat < 0) {
            if (proxy_caching_enabled) greedy_cache.emplace(key, leaf_loss); // this should also never happen
            return leaf_loss;
        }
        
        Packed L(n_words), R(n_words);
        and_bits(mask, X_bits[best_feat], L);
        andnot_bits(mask, X_bits[best_feat], R);
        if (!L.any() || !R.any()) { // this should also never happen if no error is thrown with find best split
            return leaf_loss;
        }

        const PathKey* pkLp = &empty_pk();
        const PathKey* pkRp = &empty_pk();
        PathKey pkL_local, pkR_local;
        make_child_pks_if_needed_(best_feat, pk, pkLp, pkRp, pkL_local, pkR_local);

        int left_obj  = train_greedy(L, depth_budget - 1, *pkLp);
        int right_obj = train_greedy(R, depth_budget - 1, *pkRp);
        int split_obj = left_obj + right_obj;

        int ans = min(leaf_loss, split_obj);
        if (proxy_caching_enabled) greedy_cache.emplace(key, ans);
        return ans;
    }

    int eval_with_lookahead(const Packed& m, int depth, int k, const PathKey& pk) {
        if (k <= 0) return train_greedy(m, depth, pk);
        return lickety_split(m, depth, k, pk);
    }


    inline int next_k_cycle(int k) const {
        // cycle: ... 3->2->1->K->K-1->...
        return (k > 1) ? (k - 1) : lookahead_init;
    }

    int depth1_exact_solver_cached(const Packed& mask, const PathKey& pk) {
        // const uint64_t kmask = key_of_subproblem(mask, pk);
        // constexpr int DEPTH = 1;
        // constexpr int KTAG  = 0;

        // int cached;
        // if (try_get_lickety_cached_(kmask, DEPTH, KTAG, cached)) return cached;

        constexpr int DEPTH = 1;
        constexpr int KTAG  = 0;

        uint64_t kmask = 0;
        int cached;

        if (proxy_caching_enabled) {
            kmask = key_of_subproblem(mask, pk);
            if (try_get_lickety_cached_(kmask, DEPTH, KTAG, cached)) return cached;
        }


        const int n_sub = count_total(mask);
        if (n_sub == 0) {
            return 0;
        }

        const int pos = count_pos(mask);
        const int leaf_loss = lamN + std::min(pos, n_sub - pos);

        // only cache cheap subproblems if flag enabled
        if (leaf_loss <= 2 * lamN) {
            if (proxy_caching_enabled) {
                maybe_cache_lickety_(kmask, DEPTH, KTAG, leaf_loss,/*allow_cache=*/cache_cheap_subproblems);
            }
            return leaf_loss;
        }

        int best_sum = std::numeric_limits<int>::max();

        Packed L(n_words), R(n_words);
        const int F = proxy_feat_count_();
        for (int f = 0; f < F; ++f) {
            and_bits(mask, X_bits[f], L);
            andnot_bits(mask, X_bits[f], R);
            if (!L.any() || !R.any()) continue;

            const int sum = leaf_objective(L) + leaf_objective(R);
            if (sum < best_sum) best_sum = sum;
        }

        int ans = leaf_loss;
        if (best_sum != std::numeric_limits<int>::max()) ans = std::min(ans, best_sum);

        if (proxy_caching_enabled) maybe_cache_lickety_(kmask, DEPTH, KTAG, ans, /*allow_cache=*/true);
        return ans;
    }


    int depth2_special_solver_cached(const Packed& mask, const PathKey& pk){
        // const uint64_t kmask = key_of_subproblem(mask, pk);
        // constexpr int DEPTH = 2;
        // constexpr int KTAG  = 1;

        // int cached;
        // if (try_get_lickety_cached_(kmask, DEPTH, KTAG, cached)) return cached;

        constexpr int DEPTH = 2;
        constexpr int KTAG  = 1;

        uint64_t kmask = 0;
        int cached;

        if (proxy_caching_enabled) {
            kmask = key_of_subproblem(mask, pk);
            if (try_get_lickety_cached_(kmask, DEPTH, KTAG, cached)) return cached;
        }


        const int n_sub = count_total(mask);
        if (n_sub == 0) {
            return 0;
        }

        const int pos = count_pos(mask);
        const int leaf_loss = lamN + std::min(pos, n_sub - pos);

        if (leaf_loss <= 2 * lamN) {
            if (proxy_caching_enabled) {
                maybe_cache_lickety_(kmask, DEPTH, KTAG, leaf_loss,/*allow_cache=*/cache_cheap_subproblems);
            }
            return leaf_loss;
        }

        int best_sum = std::numeric_limits<int>::max();

        Packed L(n_words), R(n_words);
        const int F = proxy_feat_count_();
        for (int f = 0; f < F; ++f) {
            and_bits(mask, X_bits[f], L);
            andnot_bits(mask, X_bits[f], R);
            if (!L.any() || !R.any()) continue;

            const PathKey* pkLp = &empty_pk();
            const PathKey* pkRp = &empty_pk();
            PathKey pkL_local, pkR_local;
            make_child_pks_if_needed_(f, pk, pkLp, pkRp, pkL_local, pkR_local);

            const int left_best  = depth1_exact_solver_cached(L, *pkLp); // depth==1 uses k=0 internally
            const int right_best = depth1_exact_solver_cached(R, *pkRp);
            const int sum = left_best + right_best;

            if (sum < best_sum) best_sum = sum;
        }

        int ans = leaf_loss;
        if (best_sum != std::numeric_limits<int>::max()) ans = std::min(ans, best_sum);

        if (proxy_caching_enabled) maybe_cache_lickety_(kmask, DEPTH, KTAG, ans, /*allow_cache=*/true);
        return ans;
    }

    int depthd_exact_solver_cached(const Packed& mask, int depth_budget, const PathKey& pk) {
        if (depth_budget <= 0) return leaf_objective(mask);
        if (depth_budget == 1) return depth1_exact_solver_cached(mask, pk);
        if (depth_budget == 2) return depth2_special_solver_cached(mask, pk);

        // const uint64_t kmask = key_of_subproblem(mask, pk);
        // const int KTAG = depth_budget - 1; // generalization of exact depth 2, this is where we would store it.

        // int cached;
        // if (try_get_lickety_cached_(kmask, depth_budget, KTAG, cached)) return cached;

        const int DEPTH = depth_budget;
        const int KTAG  = depth_budget - 1;


        uint64_t kmask = 0;
        int cached;

        if (proxy_caching_enabled) {
            kmask = key_of_subproblem(mask, pk);
            if (try_get_lickety_cached_(kmask, DEPTH, KTAG, cached)) return cached;
        }


        const int n_sub = count_total(mask);
        if (n_sub == 0) {
            return 0;
        }

        const int pos = count_pos(mask);
        const int leaf_loss = lamN + std::min(pos, n_sub - pos);

        if (leaf_loss <= 2 * lamN) {
            if (proxy_caching_enabled) {
                maybe_cache_lickety_(kmask, DEPTH, KTAG, leaf_loss,/*allow_cache=*/cache_cheap_subproblems);
            }
            return leaf_loss;
        }

        int best_sum = std::numeric_limits<int>::max();

        Packed L(n_words), R(n_words);
        const int F = proxy_feat_count_();
        for (int f = 0; f < F; ++f) {
            and_bits(mask, X_bits[f], L);
            andnot_bits(mask, X_bits[f], R);
            if (!L.any() || !R.any()) continue;

            const PathKey* pkLp = &empty_pk();
            const PathKey* pkRp = &empty_pk();
            PathKey pkL_local, pkR_local;
            make_child_pks_if_needed_(f, pk, pkLp, pkRp, pkL_local, pkR_local);

            const int left_best  = depthd_exact_solver_cached(L, depth_budget - 1, *pkLp);
            const int right_best = depthd_exact_solver_cached(R, depth_budget - 1, *pkRp);

            const int sum = left_best + right_best;
            if (sum < best_sum) best_sum = sum;
        }

        int ans = leaf_loss;
        if (best_sum != std::numeric_limits<int>::max()) ans = std::min(ans, best_sum);

        if (proxy_caching_enabled) maybe_cache_lickety_(kmask, depth_budget, KTAG, ans, /*allow_cache=*/true);
        return ans;
    }


    inline void maybe_cache_lickety_(uint64_t kmask, int depth_budget, int k, int val, bool allow_cache) {
        if (!allow_cache) return;
        const bool use_kla = use_kla_cache();
        if (use_kla) lickety_cache_kla.emplace(KLA{kmask, depth_budget, k}, val);
        else         lickety_cache_k2.emplace(K2 {kmask, depth_budget},     val);
    }

    inline bool try_get_lickety_cached_(uint64_t kmask, int depth_budget, int k, int& out_val) const {
        const bool use_kla = use_kla_cache();
        if (use_kla) {
            auto it = lickety_cache_kla.find(KLA{kmask, depth_budget, k});
            if (it == lickety_cache_kla.end()) return false;
            out_val = it->second;
            return true;
        } else {
            auto it = lickety_cache_k2.find(K2{kmask, depth_budget});
            if (it == lickety_cache_k2.end()) return false;
            out_val = it->second;
            return true;
        }
    }

    int lickety_split(const Packed& mask, int depth_budget, int k, const PathKey& pk) {
        if (depth_budget == 0) {
            return leaf_objective(mask);
        }
        if (depth_budget == 1) return depth1_exact_solver_cached(mask, pk);

        if (k > depth_budget - 1) k = depth_budget - 1;

        if (depth_budget == 2 && k == 1) return depth2_special_solver_cached(mask, pk);
        if (k == depth_budget - 1) {
            return depthd_exact_solver_cached(mask, depth_budget, pk);
        }

        
        // const uint64_t kmask = key_of_subproblem(mask, pk);
        // const bool use_kla = use_kla_cache();
        // K2  key2{kmask, depth_budget}; // two keys but only 1 will be used
        // KLA keyla{kmask, depth_budget, k};

        // if (use_kla) {
        //     if (auto it = lickety_cache_kla.find(keyla); it != lickety_cache_kla.end())
        //         return it->second;
        // } else {
        //     if (auto it = lickety_cache_k2.find(key2); it != lickety_cache_k2.end())
        //         return it->second;
        // }

        uint64_t kmask = 0;
        K2  key2{0, depth_budget};
        KLA keyla{0, depth_budget, k};

        const bool use_kla = use_kla_cache();

        if (proxy_caching_enabled) {
            kmask = key_of_subproblem(mask, pk);
            key2.k = kmask;
            keyla.k = kmask;

            if (use_kla) {
                if (auto it = lickety_cache_kla.find(keyla); it != lickety_cache_kla.end())
                    return it->second;
            } else {
                if (auto it = lickety_cache_k2.find(key2); it != lickety_cache_k2.end())
                    return it->second;
            }
        }


        const int n_sub = count_total(mask);
        const int pos   = count_pos(mask);
        const int leaf_loss = lamN + min(pos, n_sub - pos);
        if (leaf_loss <= 2 * lamN) {
            if (cache_cheap_subproblems && proxy_caching_enabled) {
                if (use_kla) lickety_cache_kla.emplace(keyla, leaf_loss);
                else         lickety_cache_k2.emplace(key2,  leaf_loss);
            }
            return leaf_loss;
        }

        int best_feat = -1;
        int best_sum  = numeric_limits<int>::max();

        Packed L(n_words), R(n_words), bestL(n_words), bestR(n_words);

        const int child_k = k - 1;
        const int F = proxy_feat_count_();
        for (int f = 0; f < F; ++f) {
            and_bits(mask, X_bits[f], L);
            andnot_bits(mask, X_bits[f], R);
            if (!L.any() || !R.any()) continue;

            const PathKey* pkLp = &empty_pk();
            const PathKey* pkRp = &empty_pk();
            PathKey pkL_local, pkR_local;
            make_child_pks_if_needed_(f, pk, pkLp, pkRp, pkL_local, pkR_local);

            //int sum = train_greedy(L, depth_budget - 1) + train_greedy(R, depth_budget - 1);
            const int sum = eval_with_lookahead(L, depth_budget - 1, child_k, *pkLp) + eval_with_lookahead(R, depth_budget - 1, child_k, *pkRp);
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
        int k_recurse;
        if (oracle_style == 0) {
            // style 0: constant k (recursively choosing based on lower tier LicketySPLIT)
            k_recurse = k;
        } else if (oracle_style == 3) {
            // style 3: decrement by 1; when it hits 0, switch to greedy tail (SPLIT)
            k_recurse = child_k; // child_k = k - 1 (already computed above)
        } else {
            // styles 1/2: restart when it hits (recursively applying SPLIT)
            k_recurse = (child_k == 0) ? lookahead_init : child_k;
        }


        int ans = leaf_loss; 
        if (best_feat >= 0) {
            const int next_depth = depth_budget - 1;

            int left_loss, right_loss;

            const PathKey* pkLp = &empty_pk();
            const PathKey* pkRp = &empty_pk();
            PathKey pkL_local, pkR_local;
            make_child_pks_if_needed_(best_feat, pk, pkLp, pkRp, pkL_local, pkR_local);

            if (oracle_style == 3 && k_recurse <= 0) {
                // greedy tail once k runs out
                left_loss  = train_greedy(bestL, next_depth, *pkLp);
                right_loss = train_greedy(bestR, next_depth, *pkRp);
            } else {
                // normal lickety recursion (k might be <=0 for other oracle styles, but fine here)
                left_loss  = lickety_split(bestL, next_depth, k_recurse, *pkLp);
                right_loss = lickety_split(bestR, next_depth, k_recurse, *pkRp);
            }

            ans = std::min(ans, left_loss + right_loss); // do lickety even if leaf is better over greedy and see which is preferred

        }
        if (proxy_caching_enabled) {
            if (use_kla) lickety_cache_kla.emplace(keyla, ans);
            else lickety_cache_k2.emplace(key2,  ans);
        }
        
        return ans;
    }

    int find_best_split(const Packed& mask, bool use_entropy) const {
        const int n_sub = count_total(mask);
        if (n_sub <= 1) return -1;

        Packed L(n_words);

        if (use_entropy) {
            const int pos_total = count_pos(mask);
            const double p0 = (double)pos_total / (double)n_sub;
            const double baseH = entropy(p0);

            int best_f = -1;
            double best_gain = -1e300;

            const int F = proxy_feat_count_();
            for (int f = 0; f < F; ++f) {
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
        } else {
            // minimize child leaf objectives: leaf_objective(L)+leaf_objective(R)
            int best_f = -1;
            int best_sum = std::numeric_limits<int>::max();

            Packed R(n_words);
            const int F = proxy_feat_count_();
            for (int f = 0; f < F; ++f) {
                // L = mask & X_bits[f]
                for (int i = 0; i < n_words; ++i) L.w[i] = mask.w[i] & X_bits[f].w[i];
                L.w[n_words-1] &= tail_mask;

                const int left_n = L.count();
                const int right_n = n_sub - left_n;
                if (left_n == 0 || right_n == 0) continue;

                // R = mask & ~X_bits[f]
                for (int i = 0; i < n_words; ++i) R.w[i] = mask.w[i] & ~X_bits[f].w[i];
                R.w[n_words-1] &= tail_mask;

                const int sum = leaf_objective(L) + leaf_objective(R);
                if (sum < best_sum) { best_sum = sum; best_f = f; }
            }
            return best_f;
        }
    }

    shared_ptr<PredNode> get_ith_tree(uint64_t i) const {
        if (!result) {
            throw runtime_error("No Rashomon trie has been constructed. Call fit() first.");
        }
        result->ensure_hist_built();

        uint64_t total = result->count_trees();
        if (i >= total) {
            throw out_of_range("Tree index out of range in get_ith_tree");
        }

        uint64_t cum = 0;
        int target_obj = -1;
        uint64_t k_within = 0;

        // hist is sorted by objective ascending
        for (const auto& e : result->hist) {
            if (i < cum + e.cnt) {
                target_obj = e.obj;
                k_within = i - cum;
                break;
            }
            cum += e.cnt;
        }
        if (target_obj < 0) {
            throw runtime_error("Failed to locate objective bucket in get_ith_tree");
        }

        return get_kth_tree_with_objective(result.get(), target_obj, k_within);
    }

        shared_ptr<PredNode> get_kth_tree_with_objective(const TreeTrieNode* node, int target_obj, uint64_t k) const {
        if (!node) {
            throw runtime_error("Null node in get_kth_tree_with_objective");
        }

        node->ensure_hist_built();

        // handle leaf-only trees at this node
        for (const auto& leaf : node->leaves) {
            if (leaf.loss == target_obj) {
                if (k == 0) {
                    auto t = make_shared<PredNode>();
                    t->feature = -1;
                    t->prediction = leaf.prediction;
                    return t;
                }
                --k;
            }
        }

        // handle splits
        for (const auto& split : node->splits) {
            const TreeTrieNode* L = split.left.get();
            const TreeTrieNode* R = split.right.get();
            if (!L || !R) continue;

            L->ensure_hist_built();
            R->ensure_hist_built();

            // total_here = #trees under this split with exactly target_obj
            uint64_t total_here = 0;

            // R is sorted by obj, so we can binary search each r_obj
            for (const auto& le : L->hist) {
                int l_obj = le.obj;
                uint64_t lc = le.cnt;
                int r_obj = target_obj - l_obj;
                // binary search r_obj in R->hist
                auto it = lower_bound(
                    R->hist.begin(), R->hist.end(),
                    HistEntry{r_obj, 0},
                    hist_less
                );
                if (it != R->hist.end() && it->obj == r_obj) {
                    uint64_t rc = it->cnt;
                    total_here += lc * rc;
                }
            }

            if (k < total_here) { // we've been decrementing k so if we are now less than the number of trees in this split, it is in this split, the kth tree in this split
                // the desired tree lies under this split.
                uint64_t running = 0;
                // counting how many trees each (l_obj, r_obj) contributes: for each l_obj, match the r_obj that meets target.
                for (const auto& le : L->hist) {
                    int l_obj = le.obj;
                    uint64_t lc = le.cnt;
                    int r_obj = target_obj - l_obj;

                    auto it = lower_bound(
                        R->hist.begin(), R->hist.end(),
                        HistEntry{r_obj, 0},
                        hist_less
                    );
                    if (it == R->hist.end() || it->obj != r_obj) continue;

                    uint64_t rc = it->cnt;
                    uint64_t pairs = lc * rc;

                    if (running + pairs > k) { // k is smaller than the culm amount in this split we've seen so far (for the first time), so we know that we want to recurse on this split (which was already known), with this particular l_obj and r_obj, but we also need what index within each objective to recurse 
                        uint64_t rel = k - running; // what index inside this block the tree lives (again, 0 indexed)
                        uint64_t left_idx  = rel / rc; // left contributes lc possibilities, right contributes rc, a cross product without filtering, this indexing scheme works to break ties
                        uint64_t right_idx = rel % rc;

                        auto left_tree  = get_kth_tree_with_objective(L, l_obj,  left_idx); // now we have all the information we need, recurse
                        auto right_tree = get_kth_tree_with_objective(R, r_obj, right_idx);

                        auto t = make_shared<PredNode>();
                        t->feature = split.feature;
                        t->prediction = -1;
                        t->left = left_tree;
                        t->right = right_tree;
                        return t;
                    }

                    running += pairs; // updating culm amount that work in this split
                }

                throw runtime_error("Inconsistent histogram counts in get_kth_tree_with_objective");
            } else {
                // skip all trees from this split that achieve target_obj
                k -= total_here;
            }
        }

        throw out_of_range("Index out of range for given objective in get_kth_tree_with_objective");
    }

    void predict_tree_recursive(const PredNode* node, const std::vector<std::vector<uint8_t>>& X_row_major, std::vector<uint8_t>& out, const std::vector<int>& idx) const {
        if (!node) return;

        // leaf: assign prediction to all indices in this subset.
        if (node->feature < 0) {
            uint8_t pred = static_cast<uint8_t>(node->prediction);
            for (int row : idx) {
                out[row] = pred;
            }
            return;
        }

        // internal node: split indices by feature
        int f = node->feature;
        std::vector<int> left_idx;
        std::vector<int> right_idx;
        left_idx.reserve(idx.size());
        right_idx.reserve(idx.size());

        for (int row : idx) {
            uint8_t v = X_row_major[row][f];
            if (v) left_idx.push_back(row); // 1 is left
            else   right_idx.push_back(row);
        }

        if (!left_idx.empty()) {
            predict_tree_recursive(node->left.get(), X_row_major, out, left_idx);
        }
        if (!right_idx.empty()) {
            predict_tree_recursive(node->right.get(), X_row_major, out, right_idx);
        }
    }

    void collect_paths(const PredNode* node, std::vector<int>& current, std::vector<std::vector<int>>& paths, std::vector<int>& preds) const {
        if (!node) return;

        // leaf: record this path and prediction
        if (node->feature < 0) {
            paths.push_back(current); // current starts empty and is appended to along the dfs
            preds.push_back(node->prediction);
            return;
        }

        int f = node->feature;
        // IMPORTANT: we have to switch to 1-indexing here so that +- for the 0th (1st) feature means something


        // go left (true) -> +f or rather f+1
        current.push_back(f+1);
        collect_paths(node->left.get(), current, paths, preds);
        current.pop_back(); // backtrack after we complete a path so we have 1 vector that is updated in a nice way throughout this

        // go right (false) -> -f (technically -(f+1))
        current.push_back(-(f+1));
        collect_paths(node->right.get(), current, paths, preds);
        current.pop_back();
    }

    // helper: add node (mask, depth_remaining) uniquely; if new, add lickety_split(mask, depth_remaining, 1)
    inline void add_frontier_unique_(
        const Packed& mask,
        int depth_remaining,
        std::unordered_set<K2, K2::Hash>& seen,
        int& running_sum
    ) {
        if (depth_remaining < 0) return;
        const uint64_t km = key_of_mask(mask);
        K2 key{km, depth_remaining};
        auto [it, inserted] = seen.insert(key);
        if (!inserted) return;
        PathKey empty_pk; // does not correctly use pk, don't do frontier with literal
        running_sum += lickety_split(mask, depth_remaining, 1, empty_pk);

    }

    struct SibEntry {
        Packed mask;
        int depth_remaining;
    };

    // dfs over a concrete tree, computing frontier score at each internal node.
    void frontier_scores_dfs_(
        const PredNode* node,
        const Packed& cur_mask,
        int depth_remaining,
        int depth_from_root,
        std::vector<SibEntry>& sib_stack,
        std::vector<std::pair<int,int>>& out
    ) {
        if (!node) return;
        if (node->feature < 0) return; // leaf: no entry

        const int f = node->feature;

        // children masks for this node
        Packed L(n_words), R(n_words);
        and_bits(cur_mask, X_bits[f], L);
        andnot_bits(cur_mask, X_bits[f], R);

        // frontier score for this internal node
        {
            std::unordered_set<K2, K2::Hash> seen;
            int sum_obj = 0;

            // siblings along the path (excluding root, including current node via sib_stack invariant)
            for (const auto& sib : sib_stack) {
                add_frontier_unique_(sib.mask, sib.depth_remaining, seen, sum_obj);
            }

            // union with the two children of this node
            add_frontier_unique_(L, depth_remaining - 1, seen, sum_obj);
            add_frontier_unique_(R, depth_remaining - 1, seen, sum_obj);

            out.emplace_back(depth_from_root, sum_obj);
        }

        if (depth_remaining <= 0) return;

        // recurse left: push sibling (R) for the left child
        if (node->left) {
            sib_stack.push_back({R, depth_remaining - 1});
            frontier_scores_dfs_(node->left.get(), L, depth_remaining - 1, depth_from_root + 1, sib_stack, out);
            sib_stack.pop_back(); // undo after exploring
        }

        // recurse right: push sibling (L) for the right child
        if (node->right) {
            sib_stack.push_back({L, depth_remaining - 1});
            frontier_scores_dfs_(node->right.get(), R, depth_remaining - 1, depth_from_root + 1, sib_stack, out);
            sib_stack.pop_back();
        }
    }

// whole trie prediction for RID

struct PredPackWithObj {
    int obj;     // training objective (lamN*leaves + miscls)
    Packed pred1; // bitset over the evaluation dataset rows: 1 iff prediction == 1
};

private:
    static inline void and_bits_eval(const Packed& a, const Packed& b, Packed& out, int n_words, uint64_t tail_mask) {
        for (int i = 0; i < n_words; ++i) out.w[i] = a.w[i] & b.w[i];
        out.w[n_words - 1] &= tail_mask;
    }

    static inline void andnot_bits_eval(const Packed& a, const Packed& b, Packed& out, int n_words, uint64_t tail_mask) {
        for (int i = 0; i < n_words; ++i) out.w[i] = a.w[i] & ~b.w[i];
        out.w[n_words - 1] &= tail_mask;
    }

    static inline void or_bits_eval(const Packed& a, const Packed& b, Packed& out, int n_words, uint64_t tail_mask) {
        for (int i = 0; i < n_words; ++i) out.w[i] = a.w[i] | b.w[i];
        out.w[n_words - 1] &= tail_mask;
    }

    static inline bool any_eval(const Packed& a) {
        for (uint64_t t : a.w) if (t) return true;
        return false;
    }

    static inline void clear_eval(Packed& a) {
        std::fill(a.w.begin(), a.w.end(), 0ULL);
    }

    static inline Packed zeros_eval(int n_words) {
        return Packed((size_t)n_words); // ctor zeros words
    }

    static inline Packed copy_eval_mask(const Packed& m, int n_words, uint64_t tail_mask) {
        Packed out((size_t)n_words);
        out.w = m.w; // vector copy
        out.w[n_words - 1] &= tail_mask;
        return out;
    }

    // build packed feature columns for EVAL X (row-major uint8 because predictions per row lend themselves to row major)
    static inline EvalCtx build_eval_ctx_(const std::vector<std::vector<uint8_t>>& X_row_major, int n_features_expected) {
        EvalCtx ctx;

        ctx.n_eval = (int)X_row_major.size();
        if (ctx.n_eval == 0) {
            ctx.n_words = 0;
            ctx.tail_mask = ~0ULL;
            return ctx;
        }

        const int d = (int)X_row_major[0].size();
        if (d != n_features_expected) {
            throw std::runtime_error("Eval X has different number of features than training.");
        }

        ctx.n_words = (ctx.n_eval + 63) / 64;
        ctx.tail_mask = (ctx.n_eval % 64) ? ((1ULL << (ctx.n_eval % 64)) - 1ULL) : ~0ULL;

        ctx.X_bits_eval.assign((size_t)d, Packed((size_t)ctx.n_words));

        for (int f = 0; f < d; ++f) {
            Packed &col = ctx.X_bits_eval[f];
            clear_eval(col);
            for (int i = 0; i < ctx.n_eval; ++i) {
                if (X_row_major[(size_t)i][(size_t)f]) {
                    col.w[(size_t)(i >> 6)] |= (1ULL << (i & 63));
                }
            }
            col.w[(size_t)(ctx.n_words - 1)] &= ctx.tail_mask;
        }

        return ctx;
    }

    // all 1s bitvector (for the evaluation passed in dataset not train)
    static inline Packed eval_root_mask_(int n_words, uint64_t tail_mask) {
        Packed m((size_t)n_words);
        if (n_words == 0) return m;
        for (int i = 0; i < n_words - 1; ++i) m.w[(size_t)i] = ~0ULL;
        m.w[(size_t)(n_words - 1)] = tail_mask;
        return m;
    }

    // convert unordered_map<int, vector<Packed>> -> sorted vector<ObjBucket>
    // before, map objective to lists of predictions
    // after conversion, it is a list of objective-bucket objects, each of which stores the list of predictions for all trees with that objective
    static inline std::vector<ObjBucket> to_sorted_buckets_(
        std::unordered_map<int, std::vector<Packed>>& acc
    ) {
        std::vector<ObjBucket> out;
        out.reserve(acc.size());
        for (auto &kv : acc) {
            ObjBucket b;
            b.obj = kv.first;
            b.preds = std::move(kv.second);
            out.push_back(std::move(b));
        }
        std::sort(out.begin(), out.end(), [](const ObjBucket& a, const ObjBucket& b){ return a.obj < b.obj; });
        return out;
    }

    // core recursion: returns buckets of predictions grouped by objective for ALL trees rooted at node with obj <= budget.
    std::vector<ObjBucket> collect_preds_by_obj_(
        const TreeTrieNode* node,
        int budget,
        const Packed& eval_mask, // does not decrease size, just gets sparser
        const EvalCtx& ctx
    ) const {
        if (!node) return {};
        if (budget < 0) return {};

        if (node->min_objective == std::numeric_limits<int>::max()) return {};
        if (node->min_objective > budget) return {};

        // accumulate as obj (training) -> list of preds on evaluation (Packed)
        std::unordered_map<int, std::vector<Packed>> acc;
        // heuristic reserve
        const int max_objs = budget - node->min_objective + 1;
        acc.reserve((size_t)std::max(1, max_objs));

        // leaves at this node
        for (const auto& leaf : node->leaves) {
            if (leaf.loss > budget) continue;

            Packed p((size_t)ctx.n_words);
            if (ctx.n_words > 0) {
                if (leaf.prediction == 1) {
                    p.w = eval_mask.w; // pred=1 on this subset
                    p.w[(size_t)(ctx.n_words - 1)] &= ctx.tail_mask;
                } else {
                    // pred=0 -> all zeros
                    clear_eval(p);
                }
            }

            acc[leaf.loss].push_back(std::move(p)); // storing the predictions in the map with that objective.
        }

        // splits
        const int INF = std::numeric_limits<int>::max();

        for (const auto& split : node->splits) {
            const TreeTrieNode* L = split.left.get();
            const TreeTrieNode* R = split.right.get();
            if (!L || !R) continue;

            const int minL = L->min_objective;
            const int minR = R->min_objective;
            if (minL == INF || minR == INF) continue;

            // cap child budgets using the other side's min objective so everything found will pair with exactly one subtree on the other side
            int bL = budget - minR;
            int bR = budget - minL;
            if (bL < 0 || bR < 0) continue;

            // also cap by the budgets actually used to build those trie nodes. should never change anything.
            bL = std::min(bL, L->budget);
            bR = std::min(bR, R->budget);

            // evaluation dataset routing masks
            Packed Lmask((size_t)ctx.n_words), Rmask((size_t)ctx.n_words);
            if (ctx.n_words > 0) {
                and_bits_eval(eval_mask, ctx.X_bits_eval[(size_t)split.feature], Lmask, ctx.n_words, ctx.tail_mask);
                andnot_bits_eval(eval_mask, ctx.X_bits_eval[(size_t)split.feature], Rmask, ctx.n_words, ctx.tail_mask);
            }

            // recurse
            auto Lb = collect_preds_by_obj_(L, bL, Lmask, ctx); // these return sorted lists of objective bucket objects
            auto Rb = collect_preds_by_obj_(R, bR, Rmask, ctx);
            if (Lb.empty() || Rb.empty()) continue;

            // for filtering by <= budget, both Lb and Rb are sorted by obj.
            // we'll two-pointer for each left obj to find all right objs <= (budget - l_obj).
            size_t r_hi = 0; // exclusive upper bound index in Rb
            for (size_t li = 0; li < Lb.size(); ++li) {
                const int lo = Lb[li].obj; // smallest objective initially
                if (lo > budget) break;
                const int rem = budget - lo; // how far do we have to look

                while (r_hi < Rb.size() && Rb[r_hi].obj <= rem) ++r_hi; // never look past the remainder because RHS is also sorted
                if (r_hi == 0) continue; // no right objs fit

                // cross product (filtered)
                for (size_t ri = 0; ri < r_hi; ++ri) { // r_hi is one past the last valid
                    const int ro = Rb[ri].obj;
                    const int tot = lo + ro;

                    // combine each left pred with each right pred (disjoint masks so OR is correct)
                    const auto& Lpreds = Lb[li].preds;
                    const auto& Rpreds = Rb[ri].preds;

                    // reserve some space in this objective bucket to reduce reallocs
                    auto &dest = acc[tot];
                    // rough reserve: only if currently empty
                    if (dest.empty()) {
                            dest.reserve(std::max(Lpreds.size(), Rpreds.size()));
                        }


                    for (const auto& lp : Lpreds) {
                        for (const auto& rp : Rpreds) {
                            Packed comb((size_t)ctx.n_words);
                            if (ctx.n_words > 0) {
                                // comb = lp | rp
                                for (int w = 0; w < ctx.n_words; ++w) {
                                    comb.w[(size_t)w] = lp.w[(size_t)w] | rp.w[(size_t)w]; // OR, combining where 1. 0s will stay 0 which is fine. if they were predicted 0 they'll never change, if they are not set yet they'll change eventually.
                                }
                                comb.w[(size_t)(ctx.n_words - 1)] &= ctx.tail_mask;
                            }
                            dest.push_back(std::move(comb));
                        }
                    }
                }
            }
        }

        return to_sorted_buckets_(acc);
    }

public:
    // main entry: enumerate ALL valid trees under the trie root (or budget_override if >=0),
    // returning (training_objective, prediction vector on evaluation dataset) for each tree.
    // NOTE: this can be extremely large in memory if the Rashomon set is huge.
    std::vector<PredPackWithObj> get_all_predictions_packed_trie(const std::vector<std::vector<uint8_t>>& X_row_major, int budget_override = -1) const {
        if (!result) {
            throw std::runtime_error("No Rashomon trie has been constructed. Call fit() first.");
        }

        EvalCtx ctx = build_eval_ctx_(X_row_major, this->n_features);

        // decide budget
        int budget = (budget_override >= 0) ? budget_override : result->budget;

        // root eval mask = all eval rows
        Packed root_mask = eval_root_mask_(ctx.n_words, ctx.tail_mask);

        // collect grouped by objective
        auto buckets = collect_preds_by_obj_(result.get(), budget, root_mask, ctx);

        // flatten
        std::vector<PredPackWithObj> out;
        // compute total count for reserve
        size_t total = 0;
        for (const auto& b : buckets) total += b.preds.size();
        out.reserve(total);

        for (auto &b : buckets) {
            for (auto &p : b.preds) {
                out.push_back(PredPackWithObj{b.obj, std::move(p)});
            }
        }
        return out;
    }
};


// extern "C" {
//     LicketyRESPLIT* create_model() {
//         return new LicketyRESPLIT();
//     }

//     void delete_model(LicketyRESPLIT* model) {
//         delete model;
//     }

//     void fit_model(LicketyRESPLIT* model,
//                       const uint8_t* X_data, int n_samples, int n_features,
//                       const int* y_data,
//                       double lambda, int depth, double rashomon_mult,
//                       double multiplicative_slack,
//                       int key_mode,
//                       int trie_cache_enabled,
//                       int lookahead_k) {
//         vector<vector<bool>> X(n_features, vector<bool>(n_samples));
//         for (int f = 0; f < n_features; ++f) {
//             const uint8_t* col = X_data + (size_t)f * (size_t)n_samples;
//             for (int i = 0; i < n_samples; ++i) X[f][i] = (col[i] != 0);
//         }
//         vector<int> y(y_data, y_data + n_samples);
//         if (key_mode == 1) model->set_key_mode(LicketyRESPLIT::KeyMode::EXACT);
//         else model->set_key_mode(LicketyRESPLIT::KeyMode::HASH64);
//         model->set_trie_cache_enabled(trie_cache_enabled != 0);
//         model->set_multiplicative_slack(multiplicative_slack);
//         model->fit(X, y, lambda, depth, rashomon_mult, lookahead_k);
//     }

//     uint64_t get_tree_count(LicketyRESPLIT* model) {
//         return model->result ? model->result->count_trees() : 0ULL;
//     }

//     uint64_t count_trees_leq(LicketyRESPLIT* model, int objective) {
//         if (!model || !model->result) return 0ULL;
//         return model->result->count_leq(objective);
//     }

//     int get_min_objective(LicketyRESPLIT* model) {
//         return model->result ? model->result->min_objective : numeric_limits<int>::max();
//     }

//     // number of distinct objective values at the root node - may or may not be useful
//     size_t get_root_hist_size(LicketyRESPLIT* model) {
//         if (!model || !model->result) return 0;
//         model->result->ensure_hist_built();
//         return model->result->hist.size();
//     }

//     // fill caller-provided buffers with (objective, count) pairs
//     void get_root_histogram(LicketyRESPLIT* model,
//                             int* objs_out,
//                             uint64_t* cnts_out) {
//         if (!model || !model->result) return;
//         model->result->ensure_hist_built();
//         const auto& hist = model->result->hist;
//         const size_t m = hist.size();
//         for (size_t i = 0; i < m; ++i) {
//             objs_out[i] = hist[i].obj;
//             cnts_out[i] = hist[i].cnt;
//         }
//     }    
// }