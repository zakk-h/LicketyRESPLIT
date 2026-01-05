import numpy as np
from licketyresplit import LicketyRESPLIT
import matplotlib.pyplot as plt

n_samples = 10000
n_features = 10

rng = np.random.default_rng(0)

X = (rng.random((n_samples, n_features)) > 0.5).astype(np.uint8)

# label: odd/even parity of first 3 features
three_sum = X[:, 0] + X[:, 1] + X[:, 2]
y = (three_sum % 2).astype(int)

model = LicketyRESPLIT()

model.fit(
    X, y,
    lambda_reg=0.01,        # sparsity penalty - recommended
    depth_budget=5,         # max tree depth as defined by number of splits along any path
    rashomon_mult=0.05,     # rashomon bound: 5% worse than initial oracle objective
    # optional parameters (default is sufficient)
    multiplicative_slack=0, # extra slack factor 
    key_mode="hash",        # "hash" for memory savings at an astronomically low change of error or "exact"
    trie_cache_enabled=False, # not recommended with hash
    lookahead_k=1, # 1 = LicketySPLIT oracle, 0 = greedy, >1 higher-tiered oracles
)

print("Minimum objective:", model.get_min_objective())
print("Rashomon set size:", model.count_trees())

hist = model.get_root_histogram()
print("Histogram entries:", hist)

for obj, cnt in hist:
    print(f"Objective = {obj}, Count = {cnt}")

tree_idx = 0
preds0 = model.get_predictions(tree_idx, X)   # shape: (n_samples,)
print(f"\nPredictions from tree {tree_idx} (first 10 samples):")
print(preds0[:10])

paths, preds = model.get_tree_paths(tree_idx)
print(f"\nTree {tree_idx} has {len(paths)} leaves.")
example_path = paths[0]
example_pred = preds[0]
print("Example root-to-leaf path:", example_path)
print("Prediction at that leaf:", example_pred)
fig, ax = plt.subplots(figsize=(6, 4))
model.plot_tree(tree_idx, ax=ax)
ax.set_title(f"LicketyRESPLIT tree {tree_idx}")
plt.tight_layout()
plt.show()


all_preds_list = model.get_all_predictions(X, stack=False)
print(f"\nNumber of trees in all_preds_list: {len(all_preds_list)}")
print("Shape of first tree's prediction vector:", all_preds_list[0].shape)

all_preds_mat = model.get_all_predictions(X, stack=True)
print("\nStacked prediction matrix shape (n_trees, n_samples):", all_preds_mat.shape)

majority_vote = (all_preds_mat.mean(axis=0) >= 0.5).astype(np.uint8)
print("Majority-vote predictions (first 10 samples):")
print(majority_vote[:10])