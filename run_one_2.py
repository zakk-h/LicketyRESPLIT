#!/usr/bin/env python3
import argparse, time, math, json
import pandas as pd
import numpy as np
import gc
import matplotlib.pyplot as plt
from itertools import product

from memory_profiler import memory_usage

from resplit import RESPLIT
from licketyresplit import LicketyRESPLIT
from treefarms import TREEFARMS
from gosdt import GOSDTClassifier
from Tree import Node, Leaf

def plot_tree(root, feature_names=None, figsize=(12, 6)):
    positions = {}
    x_counter = [0]
    node_list = []

    def _assign_coords(node, depth=0):
        if isinstance(node, Leaf):
            x = x_counter[0]
            positions[node] = (x, -depth)
            x_counter[0] += 1
            node_list.append(node)
        else:
            _assign_coords(node.left_child, depth+1)
            _assign_coords(node.right_child, depth+1)
            x_left, _  = positions[node.left_child]
            x_right, _ = positions[node.right_child]
            x = 0.5 * (x_left + x_right)
            positions[node] = (x, -depth)
            node_list.append(node)

    _assign_coords(root)

    fig, ax = plt.subplots(figsize=figsize)
    for node in node_list:
        x, y = positions[node]
        if isinstance(node, Leaf):
            txt = f"Leaf\npred={node.prediction}\nloss={node.loss:.3f}"
            box = dict(boxstyle="round,pad=0.3", fc="lightgreen", ec="gray")
        else:
            fname = f"f{node.feature}"
            if feature_names is not None and node.feature < len(feature_names):
                fname = feature_names[node.feature]
            txt = f"{fname}\nloss={node.loss:.3f}"
            box = dict(boxstyle="round,pad=0.3", fc="lightblue", ec="gray")

        ax.text(x, y, txt, ha="center", va="center", bbox=box)

        if isinstance(node, Node):
            for child in (node.left_child, node.right_child):
                x2, y2 = positions[child]
                ax.plot([x, x2], [y, y2], "-", color="gray")

    ax.set_axis_off()
    plt.tight_layout()
    plt.show()


## TreeFARMS Attempts
def structurally_equal(a, b):
    if type(a) != type(b):
        return False
    if hasattr(a, "prediction") and hasattr(b, "prediction"):
        return a.prediction == b.prediction
    if hasattr(a, "feature") and hasattr(b, "feature"):
        return (a.feature == b.feature and
                structurally_equal(a.left_child, b.left_child) and
                structurally_equal(a.right_child, b.right_child))
    return False

def tree_structure_signature(tree):  # hashable nested tuples
    if hasattr(tree, "prediction"):
        return ('leaf', int(tree.prediction))
    if hasattr(tree, "feature"):
        return ('node',
                int(tree.feature),
                tree_structure_signature(tree.left_child),
                tree_structure_signature(tree.right_child))
    raise ValueError(f"Unknown tree type: {type(tree)}")

def _collect_trees_from_treefarms(model, max_try=10000000):
    trees, i = [], 0
    while i < max_try:
        try:
            t = model[i]
        except Exception:
            break
        trees.append(t)
        i += 1
   
    return trees

def _tree_errors(tree, X, y):
    N = len(y)
    if hasattr(tree, "score"):
        acc = float(tree.score(X, y))
        return int(round((1.0 - acc) * N)), acc
    raise RuntimeError("Tree does not expose .score(X,y)")

def _num_leaves(tree):
    if hasattr(tree, "leaves"):
        return int(getattr(tree, "leaves")())
    raise RuntimeError("Tree does not expose leaves()/num_leaves().")

def summarize_treefarms_objectives(model, X, y, reg, epsilon=0.01):
    trees = _collect_trees_from_treefarms(model)
    if not trees:
        raise RuntimeError("No trees retrieved from TREEFARMS (indexing/iter failed).")

    N = len(y)
    lamN = int(round(float(reg) * N))

    objs, err_list, leaf_list, acc_list = [], [], [], []
    for i, t in enumerate(trees):
        errs, acc = _tree_errors(t, X, y)
        leaves = _num_leaves(t)
        obj = errs + lamN * leaves
        objs.append(obj)
        err_list.append(errs)
        leaf_list.append(leaves)
        acc_list.append(acc)

    objs = np.asarray(objs, dtype=float)
    min_obj = float(np.min(objs))
    max_obj = float(np.max(objs))
    add_gap = max_obj - min_obj
    multiplicative_range = max_obj / min_obj # = 1 + x
    x = multiplicative_range - 1.0 # solve min*(1+x)=max

    cutoff = (1.0 + float(epsilon)) * min_obj
    passing_idx = [i for i, v in enumerate(objs) if v <= cutoff]

    print("\n[TREEFARMS Objective Summary]")
    print(f"trees: {len(trees)} | N: {N} | reg: {reg} | lamN: {lamN}")
    print(f"min_objective: {int(min_obj)}  |  max_objective: {int(max_obj)}")
    print(f"additive_gap (max - min): {int(add_gap)}")
    print(f"multiplicative_range = max/min = {multiplicative_range:.6f}  (x such that min*(1+x)=max is x = {x:.6f})")
    print(f"epsilon: {epsilon} | cutoff = min*(1+epsilon) = {cutoff:.6f}")
    print(f"trees within cutoff: {len(passing_idx)}  |  indices: {passing_idx}")

    plt.figure(figsize=(8, 4))
    plt.hist(objs, bins=20, color='skyblue', edgecolor='black')
    plt.axvline(min_obj, color='green', linestyle='--', label='min objective')
    plt.axvline(cutoff, color='red', linestyle='--', label='(1 + ε) · min')
    plt.axvline(min_obj + epsilon * N, color='purple', linestyle='--', label='min + ε · N')

    plt.title("Histogram of TREEFARMS Objectives")
    plt.xlabel("Objective value (errors + λN·leaves)")
    plt.ylabel("Count")
    plt.legend()
    plt.tight_layout()
    plt.show()

    objs_normalized = [obj / N for obj in objs]
    min_obj_norm = min_obj / N
    cutoff_norm = cutoff / N

    plt.figure(figsize=(8, 4))
    plt.hist(objs_normalized, bins=20, color='lightcoral', edgecolor='black')
    plt.axvline(min_obj_norm, color='green', linestyle='--', label='min objective')
    plt.axvline(cutoff_norm, color='red', linestyle='--', label='(1 + ε) · min')

    plt.title("Histogram of TREEFARMS Objectives (Normalized by N [Standard Formulation])")
    plt.xlabel("Normalized Objective (misclassification_prop + λ·leaves)")
    plt.ylabel("Count")
    plt.legend()
    plt.tight_layout()
    plt.show()

    return {
        "num_trees": len(trees),
        "lamN": lamN,
        "min_objective": min_obj,
        "max_objective": max_obj,
        "additive_gap": add_gap,
        "multiplicative_range": multiplicative_range,  # equals 1 + x
        "x": x,
        "epsilon": float(epsilon),
        "cutoff": cutoff,
        "passing_indices": passing_idx,
        "per_tree": [
            {"index": i, "objective": int(objs[i]), "errors": int(err_list[i]),
             "leaves": int(leaf_list[i]), "accuracy": float(acc_list[i])}
            for i in range(len(trees))
        ],
    }

## End TreeFARMS Attempts

# -------------- helpers --------------

def load_dataset(path):
    df = pd.read_csv(path)
    X, y = df.iloc[:, :-1], df.iloc[:, -1]
    print(f"Loaded {df.shape[0]} rows, {df.shape[1]-1} features from {path}")
    return X, y

def fit_gosdt_get_objective(X, y, reg, depth_budget=5, verbose=False):
    """Returns reg*#leaves + model_loss from GOSDT at the given depth."""
    clf = GOSDTClassifier(
        regularization=reg,
        time_limit=6000,
        depth_budget=int(depth_budget),
        verbose=verbose
    )
    clf.fit(X, y)

    model_loss = clf.result_.model_loss
    raw_model = clf.result_.model

    def count_leaves(node):
        if isinstance(node, str):
            node = json.loads(node)
        if isinstance(node, list):
            return count_leaves(node[0])
        if "true" not in node and "false" not in node:
            return 1
        return sum(count_leaves(node[br]) for br in ("true","false") if br in node)

    n_leaves = count_leaves(raw_model)
    return model_loss + reg * n_leaves

def run_resplit(X, y, reg, mult, depth):
    # RESPLIT inherits TREEFARMS' depth convention (root depth = 1)
    depth_tf = depth + 1
    config = {
        "regularization": reg,
        "rashomon_bound_multiplier": mult,
        "depth_budget": depth_tf,
        "cart_lookahead_depth": math.ceil((depth_tf-1) / 2),
        "verbose": False
    }
    model = RESPLIT(config, fill_tree="treefarms")
    t0 = time.perf_counter()
    model.fit(X, y)
    dt = time.perf_counter() - t0
    n_trees = len(model)
    label = "RESPLIT-treefarms"
    return dt, n_trees, label, model

def run_lickety(X, y, reg, mult, depth, best_objective=None, lookahead=1, prune_style="Z", consistent_lookahead=True, better_than_greedy=False, try_greedy_first=False, trie_cache_strategy = "compact", multiplicative_slack=0, cache_greedy=True, cache_lickety=True, cache_packbits=True, cache_key_mode="bitvector", stop_caching_at_depth=-1):
    config = {
        "regularization": reg,
        "rashomon_bound_multiplier": mult,
        "depth_budget": depth,
    }
    if best_objective is not None:
        config["best_objective"] = best_objective
    model = LicketyRESPLIT(config, multipass=True, lookahead=int(lookahead), optimal=False, pruning=True, prune_style=prune_style, consistent_lookahead=consistent_lookahead, better_than_greedy=better_than_greedy, try_greedy_first=try_greedy_first, multiplicative_slack=multiplicative_slack, trie_cache_strategy = trie_cache_strategy, cache_greedy=cache_greedy, cache_lickety=cache_lickety, cache_packbits=cache_packbits, cache_key_mode=cache_key_mode, stop_caching_at_depth=stop_caching_at_depth)
    t0 = time.perf_counter()
    model.fit(X, y)
    dt = time.perf_counter() - t0
    n_trees = model.count_trees()
    print(f"Found {model.trie.count_trees()} trees but truncated to {n_trees}")
    label = f"LicketyRESPLIT[lookahead={lookahead}{prune_style}, best={'gosdt' if best_objective is not None else 'lickety'}]"
    return dt, n_trees, label, model

def run_treefarms(X, y, reg, mult, depth):
    config = {
        "regularization": reg,
        "rashomon_bound_multiplier": mult,
        "depth_budget": depth + 1,  # TF counts a single leaf as depth 1
        "verbose": False
    }
    model = TREEFARMS(config)
    t0 = time.perf_counter()
    model.fit(X, y)
    dt = time.perf_counter() - t0
    if hasattr(model, "get_tree_count"):
        n_trees = model.get_tree_count()
    elif hasattr(model, "__len__"):
        n_trees = len(model)
    else:
        n_trees = None
    label = "TREEFARMS"

    stats = summarize_treefarms_objectives(model, X, y, reg=reg, epsilon=mult)
    return dt, n_trees, label, model

# -------------- main --------------
def main(data_path, algo="lickety", reg=0.01, depth=10, mult=0.01, use_gosdt_objective=False, better_than_greedy = False, lookahead_k = 1, prune_style = "H", consistent_lookahead=False, try_greedy_first=False, trie_cache_strategy = "compact", multiplicative_slack=0.00, cache_greedy=True, cache_lickety=True, cache_packbits=True, cache_key_mode="bitvector", stop_caching_at_depth=-1):

    # =======================================

    X, y = load_dataset(data_path)

    if algo == "resplit":
        target = run_resplit
        kwargs = dict(X=X, y=y, reg=reg, mult=mult, depth=depth)

    elif algo == "lickety":
        best_obj = None
        if use_gosdt_objective:
            print("Computing GOSDT objective as baseline...")
            best_obj = fit_gosdt_get_objective(X, y, reg=reg, depth_budget=depth)
            #best_obj = 702
            print(f"GOSDT objective: {best_obj:.6f}")

        X_arr = X.to_numpy(copy=False) if hasattr(X, "to_numpy") else np.asarray(X)
        y_arr = y.to_numpy(copy=False) if hasattr(y, "to_numpy") else np.asarray(y)
        X_bool = np.asfortranarray(X_arr != 0)
        y_uint8 = np.ascontiguousarray((y_arr != 0).astype(np.uint8, copy=False))
        del X, y, X_arr, y_arr
        gc.collect()
        target = run_lickety
        kwargs = dict(X=X_bool, y=y_uint8, reg=reg, mult=mult, depth=depth, best_objective=best_obj, lookahead=lookahead_k, prune_style=prune_style, consistent_lookahead=consistent_lookahead, better_than_greedy=better_than_greedy, try_greedy_first=try_greedy_first, trie_cache_strategy = trie_cache_strategy, multiplicative_slack=multiplicative_slack, cache_greedy=cache_greedy, cache_lickety=cache_lickety, cache_packbits=cache_packbits, cache_key_mode=cache_key_mode, stop_caching_at_depth=stop_caching_at_depth)

    elif algo == "treefarms":
        target = run_treefarms
        kwargs = dict(X=X, y=y, reg=reg, mult=mult, depth=depth)

    else:
        raise ValueError(f"Unknown algo: {algo}")

    baseline_mb = memory_usage(-1, interval=0.05, timeout=1)[0]
    # Measure peak RSS during the call (includes child processes if any)
    peak_mb, retval = memory_usage(
        (target, (), kwargs),
        max_usage=True,
        retval=True,
        interval=0.01,
        include_children=True
    )
    delta_mb = peak_mb - baseline_mb
    duration_s, n_trees, label, model = retval

    if False and algo == "lickety":
        trie = model.trie
        num_unique = trie.count_unique_prediction_vectors(X=X_bool)
        print(f"Number of unique predicting trees : {num_unique}")

        keys = sorted(trie.objectives)
        total = sum(trie.objectives.values())
        min_obj, max_obj = keys[0], keys[-1]
        mean_obj = sum(k * c for k, c in trie.objectives.items()) / total
        print(f"objective_stats     : min={min_obj} mean={mean_obj:.3f} max={max_obj}")

      
        print(f"best_objective      : {model.best}")
        print(f"rashomon_obj_bound  : {model.obj_bound}")

        #for i, tree in enumerate(model.list_trees()):
        #    plot_tree(tree, feature_names=None, figsize=(12, 6))



    print("\n=== RESULT ===")
    print(f"algo               : {label}")
    print(f"data               : {data_path}")
    print(f"reg (lambda)       : {reg}")
    print(f"depth              : {depth}")
    print(f"rashomon_mult      : {mult}")
    if algo == "lickety":
        print(f"lookahead   : {lookahead_k}")
        print(f"use_gosdt_objective: {use_gosdt_objective}")
    print(f"time_sec           : {duration_s:.4f}")
    print(f"peak_rss_mb        : {peak_mb:.1f}")
    print(f"delta_rss_mb        : {delta_mb:.1f}")
    print(f"num_trees          : {n_trees}")

    return duration_s, peak_mb, delta_mb, n_trees

    

if __name__ == "__main__":
    main("bike_binarized_many.csv", "lickety", reg=0.005, depth=10, mult=0.015, lookahead_k=1, prune_style="H", consistent_lookahead=False, better_than_greedy=False, use_gosdt_objective=False, try_greedy_first=False, trie_cache_strategy = "superset", cache_greedy=True, cache_lickety=True, cache_packbits=True, cache_key_mode="bitvector", stop_caching_at_depth=0)
    main("bike_binarized_many.csv", "lickety", reg=0.005, depth=10, mult=0.015, lookahead_k=2, prune_style="H", consistent_lookahead=False, better_than_greedy=False, use_gosdt_objective=False, try_greedy_first=False, trie_cache_strategy = "superset", cache_greedy=True, cache_lickety=True, cache_packbits=True, cache_key_mode="bitvector", stop_caching_at_depth=0)
    main("bike_binarized_many.csv", "lickety", reg=0.005, depth=10, mult=0.015, lookahead_k=10, prune_style="H", consistent_lookahead=False, better_than_greedy=False, use_gosdt_objective=False, try_greedy_first=False, trie_cache_strategy = "superset", cache_greedy=True, cache_lickety=True, cache_packbits=True, cache_key_mode="bitvector", stop_caching_at_depth=0)