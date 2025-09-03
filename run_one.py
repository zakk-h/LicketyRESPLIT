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

## TreeFARMS Attempts

def _collect_trees_from_treefarms(model, max_try=100000):
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

def _num_leaves(tree):
    if hasattr(tree, "leaves"):
        return int(getattr(tree, "leaves")())

def summarize_treefarms_objectives(model, X, y, reg, epsilon=0.01):
    trees = _collect_trees_from_treefarms(model)

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

def run_resplit(X, y, reg, mult, depth, method="resplitt"):
    # RESPLIT inherits TREEFARMS' depth convention (root depth = 1)
    depth_tf = depth + 1
    config = {
        "regularization": reg,
        "rashomon_bound_multiplier": mult,
        "depth_budget": depth_tf,
        "cart_lookahead_depth": math.ceil((depth_tf) / 2),
        "verbose": False
    }
    if method == "resplitt": fill_method="treefarms"
    else: fill_method = "greedy"
    model = RESPLIT(config, fill_tree=fill_method)
    t0 = time.perf_counter()
    model.fit(X, y)
    dt = time.perf_counter() - t0
    n_trees = len(model)
    label = f"RESPLIT[{fill_method}]"
    return dt, n_trees, label, model

def run_lickety(X, y, reg, mult, depth, best_objective=None, lookahead=1, prune_style="Z", consistent_lookahead=True, better_than_greedy=False, try_greedy_first=False, trie_cache_strategy = "compact", multiplicative_slack=0):
    config = {
        "regularization": reg,
        "rashomon_bound_multiplier": mult,
        "depth_budget": depth,
    }
    if best_objective is not None:
        config["best_objective"] = best_objective
    model = LicketyRESPLIT(config, multipass=True, lookahead=int(lookahead), optimal=False, pruning=True, prune_style=prune_style, consistent_lookahead=consistent_lookahead, better_than_greedy=better_than_greedy, try_greedy_first=try_greedy_first, trie_cache_strategy = trie_cache_strategy, multiplicative_slack=multiplicative_slack)
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
def main(data_path, algo="lickety", reg=0.01, depth=10, mult=0.01, use_gosdt_objective=False, better_than_greedy = False, lookahead_k = 1, prune_style = "H", consistent_lookahead=False, try_greedy_first=False, trie_cache_strategy = "compact", multiplicative_slack=0.00):

    # =======================================

    X, y = load_dataset(data_path)

    if algo == "resplitt" or algo=="resplitg":
        target = run_resplit
        kwargs = dict(X=X, y=y, reg=reg, mult=mult, depth=depth, method=algo)

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
        kwargs = dict(X=X_bool, y=y_uint8, reg=reg, mult=mult, depth=depth, best_objective=best_obj, lookahead=lookahead_k, prune_style=prune_style, consistent_lookahead=consistent_lookahead, better_than_greedy=better_than_greedy, try_greedy_first=try_greedy_first, trie_cache_strategy = trie_cache_strategy, multiplicative_slack=multiplicative_slack)

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

    if algo == "lickety":
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
    #main("bike_binarized.csv", "resplit", reg=0.005, depth=4, mult=0.01, lookahead_k=1, prune_style="H", consistent_lookahead=False, better_than_greedy=False, use_gosdt_objective=False, try_greedy_first=False, trie_cache_strategy = "compact")
    parser = argparse.ArgumentParser()

    parser.add_argument("--data", type=str, default="spambase_binarized.csv")
    parser.add_argument("--algo", type=str, default="lickety", choices=["resplitt", "resplitg", "lickety", "treefarms"])
    parser.add_argument("--reg", type=float, default=0.005)
    parser.add_argument("--depth", type=int, default=4)
    parser.add_argument("--mult", type=float, default=0.01)

    # Lickety-specific
    parser.add_argument("--lookahead_k", type=int, default=2)
    parser.add_argument("--prune_style", type=str, default="Z", choices=["H", "Z"])
    parser.add_argument("--consistent_lookahead", action="store_true")
    parser.add_argument("--better_than_greedy", action="store_true")
    parser.add_argument("--try_greedy_first", action="store_true")
    parser.add_argument("--trie_cache_strategy", type=str, default="compact")
    parser.add_argument("--multiplicative_slack", type=float, default=0.0)
    parser.add_argument("--use_gosdt_objective", action="store_true")

    args = parser.parse_args()

    main(
        data_path=args.data,
        algo=args.algo,
        reg=args.reg,
        depth=args.depth,
        mult=args.mult,
        use_gosdt_objective=args.use_gosdt_objective,
        better_than_greedy=args.better_than_greedy,
        lookahead_k=args.lookahead_k,
        prune_style=args.prune_style,
        consistent_lookahead=args.consistent_lookahead,
        try_greedy_first=args.try_greedy_first,
        trie_cache_strategy=args.trie_cache_strategy,
        multiplicative_slack=args.multiplicative_slack
    )