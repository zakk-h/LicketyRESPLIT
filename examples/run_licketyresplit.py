import numpy as np
from licketyresplit import LicketyRESPLIT

n_samples = 100
n_features = 10

rng = np.random.default_rng(0)
X = (rng.random((n_samples, n_features)) > 0.5).astype(np.uint8)
y = (rng.random(n_samples) > 0.5).astype(int)

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
