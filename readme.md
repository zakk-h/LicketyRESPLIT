# LicketyRESPLIT: Fast Rashomon-set Enumeration for Sparse Decision Trees

This module exposes two tightly connected pieces:

## Core Components

### 1. LicketyRESPLIT (class)
- Enumerates the Rashomon set (all trees within a multiplicative objective bound of the best tree) for classification problems (currently only supporting binary)
- Uses a powerful oracle (LicketySPLIT with different generalizations to higher lookaheads) to quickly approximate optimal subproblems for pruning and budgeting
- Materializes results as a *Trie of subproblems* (`TreeTrieNode`) that supports fast counting, indexing, and prediction.

### 2. TreeTrie (TreeTrieNode / SplitNode)
- A compact combinatorial structure that records, for a given budget, all leaf choices and feature splits and their achievable objectives
- Most functionality is accessed via the LicketyRESPLIT class, but direct interaction with the trie (e.g. model.trie) may be necessary for advanced operations such as truncation, ranking, or retrieving prediction vectors.

## Quickstart

```python
from licketyresplit import LicketyRESPLIT

config = {
    "regularization": 0.01,          # lambda (misclass + λN·leaves)
    "depth_budget": 5,               # tree depth (root depth = 0 here)
    "rashomon_bound_multiplier": 0.01
}

model = LicketyRESPLIT(
    config,
    lookahead=1,                     # 1 = standard lickety (greedy completion)
    trie_cache_strategy="compact",   # example other parameter - default functionality works perfectly fine
)

# X, y must be binary (0/1). If not, set binarize=True.
model.fit(X, y)

# Access the trie of solutions and count/list trees in the Rashomon set:
n = model.count_trees()                                   # within (1+mult)*min
trees = model.list_trees()                                # materialize trees
for preds in model.iter_predictions(X):                   # iterate predictions
    ...

# You can also select narrower/wider objective bands:
n_tight = model.count_trees(max_obj=int(model.best*1.005))
```

## Objective, Budgets, and the Rashomon Set

We use an integer objective (the standard objective in GOSDT/TreeFARMS/RESPLIT) but scaled by N to become integral and avoid FP drift: `Obj(tree) = errors + (λ·N)·#leaves`

- "best" is the integer objective value at the root (found by LicketySPLIT unless provided). The Rashomon bound is `round(best·(1 + mult))`

## Memory Layout and Data Contracts

### Input expectations (when `binarize=False`):

Any reasonable input will suffice and be handled internally, however, we note details about how we store the input internally as it requires a conversion otherwise. 

- **X_bool**: boolean array (preferred Fortran order for fast column access). If you pass anything else, fit() converts with:
  ```python
  self.X_bool = np.asfortranarray(np.asarray(X) != 0, dtype=bool)
  ```
- **y_full**: uint8 vector with values in {0,1}
- **y_bool**: a bool view on y_full (no copy), created via:
  ```python
  self.y_bool = self.y_full.view(np.bool_)
  ```

### If `binarize=True`:
- A ThresholdGuessBinarizer pre-processes X; y is interpreted as {0,1}

## Caching & Keys

### 1. `cache_key_mode`: "bitvector" or "literal"
- **"bitvector"**: keys derive from the *data subset* bitmask
  ```python
  key = packbits(bitvector).tobytes()
  ```
  (Optionally interned to reuse the same bytes object.)
- **"literal"**: keys derive from the *path constraint set*
  ```python
  key = frozenset({ encode(feature,val_bit), ... })
  ```
  Literal keys are an order-invariant way to record the boolean conditions that identify this subproblem (work in progress)

### 2. Packed-bit interning (`cache_packbits`)
- When True and the relevant cache is ON for a call site (greedy/lickety/trie), packed bytes are *interned* via a per-instance memo dict: identical bitmasks share the same bytes object (saves memory)
- If all caches are disabled (`cache_greedy=False`, `cache_lickety=False`, `trie_cache_strategy=None`) we automatically disable interning

### Trie cache strategies:
- **None**: no memoization of subtries
- **"compact"**: cache two "envelopes" per key — the largest depth seen and the largest budget seen — and auto-truncate for smaller queries
- **"superset"**: actively probe supersets (slightly heavier) to reuse work

## Lookahead and Pruning

- **lookahead = k**:
  - k=1: standard LicketySPLIT (choose next feature by greedy-completed subtrees)
- **prune_style="H"**: uses a lower tier oracle to choose the best split consistently (k-1)
- **prune_style="Z"**: cycle k, k-1, ..., 1, k, ... as the lookahead for choosing each split. You can also enable `consistent_lookahead=True` to map depth->k deterministically (helps cache hits)
- **try_greedy_first=True** allows cheap pruning: if greedy(left)+greedy(right) already is within budget, skip expensive oracles and explore

## What the Trie Stores

```python
TreeTrieNode(budget):
    children      : [Leaf(...)] and/or [SplitNode(feature, leftTrie, rightTrie)]
    min_objective : best achievable objective under this subproblem/budget
    objectives    : Counter mapping objective_value -> count of trees achieving it
```

Each `SplitNode` precomputes how many trees under (left,right) *fit parent budget* and records it as `num_valid_trees` for fast i-th tree selection.

### Key operations you'll use:
- `trie.count_trees()` → total # trees ≤ budget
- `trie.count_trees_within_objective(max_obj, min_obj=None, inclusive=True)`
- `trie.list_trees()` / `trie.list_trees_within_objective(...)`
- `trie[i]` → i-th tree (0-indexed)
- `trie.get_predictions(i, X)` → predictions from i-th tree
- `trie.iter_predictions(X)` / `get_all_predictions(X, stack=False)`
- `trie.truncated_copy(max_depth, budget=None)` - Make a deep copy that limits the number of splits and/or shrinks budget (this is used in LicketyRESPLIT depending on the cache strategy)

LicketyRESPLIT convenience wrappers call into the trie with the correct Rashomon bound, e.g.:
- `model.count_trees(...)`
- `model.list_trees(...)`
- `model.iter_predictions(...)`

## Indexing, Objectives, and Enumeration Order

- Trees are ordered by increasing objective, breaking ties with a deterministic internal method.

## Interfacing Notes

- Depth convention in this module is "root depth = 0". 
- Objective values are integers; when you supply best_objective in `config`, you can pass count-form (int) or normalized (float in [0,1]) — it's normalized internally to counts.
- If you pass X as a bool Fortran-array and y as uint8 0/1, fit() does no extra allocation for those arrays (only views)
- In "literal" cache mode currently you still compute bitvectors for data slicing, but trie/greedy/lickety cache keys come from frozensets of encoded literals

## Troubleshooting & Performance

- If memory spikes: consider `trie_cache_strategy=None` and `greedy_cache=lickety_cache=cache_packbits=False` to storing tons of subproblem solutions (though this will greatly increase runtime). 


## Testing

See `run_one.py` to run and compare Rashomon set algorithms (TreeFARMS, RESPLIT, LicketyRESPLIT) with a binarized dataset in the repository (such as `bike_binarized.csv`)