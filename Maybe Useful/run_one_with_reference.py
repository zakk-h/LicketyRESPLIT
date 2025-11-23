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

import os
from pathlib import Path

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

def run_lickety(X, y, reg, mult, depth, best_objective=None, lookahead=1, prune_style="Z", consistent_lookahead=True, better_than_greedy=False, try_greedy_first=False, trie_cache_strategy = "compact", multiplicative_slack=0, cache_greedy=True, cache_lickety=True, cache_packbits=True, cache_key_mode="bitvector", stop_caching_at_depth=-1, oracle_top_k=None):
    config = {
        "regularization": reg,
        "rashomon_bound_multiplier": mult,
        "depth_budget": depth,
    }
    if best_objective is not None:
        config["best_objective"] = best_objective
    model = LicketyRESPLIT(config, multipass=True, lookahead=int(lookahead), optimal=False, pruning=True, prune_style=prune_style, consistent_lookahead=consistent_lookahead, better_than_greedy=better_than_greedy, try_greedy_first=try_greedy_first, multiplicative_slack=multiplicative_slack, trie_cache_strategy = trie_cache_strategy, cache_greedy=cache_greedy, cache_lickety=cache_lickety, cache_packbits=cache_packbits, cache_key_mode=cache_key_mode, stop_caching_at_depth=stop_caching_at_depth, oracle_top_k=oracle_top_k)
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


def count_true_rashomon_lickety(trie, true_bound_int: int) -> int:
    return sum(cnt for obj, cnt in trie.objectives.items() if obj <= true_bound_int)

def count_true_rashomon_resplit(model_resplit, X, y, true_bound_norm: float, reg: float) -> int:
    cnt, _ = model_resplit.count_objectives_between(
        X, y, low=0.0, high=true_bound_norm, reg=reg, return_list=True
    )
    return cnt

def fit_lickety_best_objective(X, y, reg, depth,
                               prune_style="H",
                               trie_cache_strategy="compact",
                               multiplicative_slack=0.0):
    """
    Runs LicketyRESPLIT with lookahead==depth and returns the best integer objective
    in Lickety's native scale: errors + (λN)*leaves.
    """
    # Prepare boolean/uint8 data as Lickety expects
    X_arr = X.to_numpy(copy=False) if hasattr(X, "to_numpy") else np.asarray(X)
    y_arr = y.to_numpy(copy=False) if hasattr(y, "to_numpy") else np.asarray(y)
    X_bool = np.asfortranarray(X_arr != 0)
    y_uint8 = np.ascontiguousarray((y_arr != 0).astype(np.uint8, copy=False))

    config = {
        "regularization": reg,
        "rashomon_bound_multiplier": 0.0,  # baseline search; bound not needed
        "depth_budget": int(depth),
    }
    # Full lookahead == depth
    model = LicketyRESPLIT(
        config,
        multipass=True,
        lookahead=int(depth),
        optimal=False,
        pruning=True,
        prune_style=prune_style,
        consistent_lookahead=False,
        better_than_greedy=False,
        try_greedy_first=False,
        multiplicative_slack=multiplicative_slack,
        trie_cache_strategy=trie_cache_strategy,
        cache_greedy=False,
        cache_lickety=True,
        cache_packbits=True,
        cache_key_mode="bitvector",
        stop_caching_at_depth=2,
        oracle_top_k=None
    )
    model.fit(X_bool, y_uint8)
    print("Best:", model.best)
    return int(model.best)  # Lickety's integer objective


# -------------- main --------------
def main(data_path, algo="lickety", reg=0.01, depth=10, mult=0.01, use_gosdt_objective=False, better_than_greedy = False, lookahead_k = 1, prune_style = "H", consistent_lookahead=False, try_greedy_first=False, trie_cache_strategy = "compact", multiplicative_slack=0.00, cache_greedy=True, cache_lickety=True, cache_packbits=True, cache_key_mode="bitvector", stop_caching_at_depth=-1):

    # =======================================

    X, y = load_dataset(data_path)

    if algo == "resplit":
        target = run_resplit
        kwargs = dict(X=X, y=y, reg=reg, mult=mult, depth=depth)


    elif algo == "lickety":
        X_arr = X.to_numpy(copy=False) if hasattr(X, "to_numpy") else np.asarray(X)
        y_arr = y.to_numpy(copy=False) if hasattr(y, "to_numpy") else np.asarray(y)
        X_bool = np.asfortranarray(X_arr != 0)
        y_uint8 = np.ascontiguousarray((y_arr != 0).astype(np.uint8, copy=False))
        del X, y, X_arr, y_arr
        gc.collect()
        target = run_lickety
        kwargs = dict(X=X_bool, y=y_uint8, reg=reg, mult=mult, depth=depth, best_objective=None, lookahead=lookahead_k, prune_style=prune_style, consistent_lookahead=consistent_lookahead, better_than_greedy=better_than_greedy, try_greedy_first=try_greedy_first, trie_cache_strategy = trie_cache_strategy, multiplicative_slack=multiplicative_slack, cache_greedy=cache_greedy, cache_lickety=cache_lickety, cache_packbits=cache_packbits, cache_key_mode=cache_key_mode, stop_caching_at_depth=stop_caching_at_depth)

    elif algo == "treefarms":
        target = run_treefarms
        kwargs = dict(X=X, y=y, reg=reg, mult=mult, depth=depth)

    else:
        raise ValueError(f"Unknown algo: {algo}")

    baseline_mb = memory_usage(-1, interval=0.05, timeout=1)[0]
    # measure peak RSS during the call (includes child processes if any)
    peak_mb, retval = memory_usage(
        (target, (), kwargs),
        max_usage=True,
        retval=True,
        interval=0.01,
        include_children=True
    )
    print("Done with main algorithm")
    delta_mb = peak_mb - baseline_mb
    duration_s, n_trees, label, model = retval

    true_rashomon_count = None
    slack_counts = None  # will hold 5 integers

    if algo in ("resplit", "lickety"):
        # (Re)load normalized X,y for GOSDT
        X_g, y_g = load_dataset(data_path)
        # Lickety-as-baseline: full lookahead to get best integer objective
        best_int = fit_lickety_best_objective(
            X_g, y_g, reg=reg, depth=depth,
            prune_style=prune_style,
            trie_cache_strategy=trie_cache_strategy,
            multiplicative_slack=multiplicative_slack
        )

        multipliers = [1.01, 1.02, 1.03, 1.04, 1.05, 1.1, 1.15, 1.2, 1.25, 1.5]

        if algo == "lickety":
            # Lickety uses integer objective natively
            base_int = best_int
            bound_int = int(round((1.0 + float(mult)) * base_int))
            true_rashomon_count = count_true_rashomon_lickety(model.trie, bound_int)
            slack_counts = [
                count_true_rashomon_lickety(model.trie, int(round(s * bound_int)))
                for s in multipliers
            ]

        elif algo == "resplit":
            # RESPLIT uses normalized objective: divide Lickety's integer objective by N
            N = len(y_g)
            base_norm = float(best_int) / float(N)
            bound_norm = (1.0 + float(mult)) * base_norm  # no rounding
            true_rashomon_count = count_true_rashomon_resplit(model, X_g, y_g, bound_norm, reg=reg)
            slack_counts = [
                count_true_rashomon_resplit(model, X_g, y_g, s * bound_norm, reg=reg)
                for s in multipliers
            ]



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

    #return duration_s, peak_mb, delta_mb, n_trees
    return duration_s, peak_mb, delta_mb, n_trees, true_rashomon_count, slack_counts
    

if __name__ == "__main__":
    #main("bike_binarized_many.csv", "lickety", reg=0.01, depth=5, mult=0.01, lookahead_k=1, prune_style="H", consistent_lookahead=False, better_than_greedy=False, use_gosdt_objective=False, try_greedy_first=False, trie_cache_strategy = None, cache_greedy=False, cache_lickety=False, cache_packbits=False, cache_key_mode="bitvector", stop_caching_at_depth=0)
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="mushroom_binarized.csv")
    parser.add_argument("--algo", type=str, choices=["lickety", "resplit", "treefarms"], default="resplit")
    parser.add_argument("--reg", type=float, default=0.005)
    parser.add_argument("--depth", type=int, default=5)
    parser.add_argument("--mult", type=float, default=0.03)
    parser.add_argument("--lookahead_k", type=int, default=1)
    parser.add_argument("--prune_style", type=str, default="H")
    parser.add_argument("--consistent_lookahead", type=lambda s: s.lower() == "true", default=False)
    parser.add_argument("--better_than_greedy", type=lambda s: s.lower() == "true", default=False)
    parser.add_argument("--use_gosdt_objective", type=lambda s: s.lower() == "true", default=False)
    parser.add_argument("--try_greedy_first", type=lambda s: s.lower() == "true", default=False)
    parser.add_argument("--trie_cache_strategy", type=str, default="compact")
    parser.add_argument("--multiplicative_slack", type=float, default=0.00)
    parser.add_argument("--cache_greedy", type=lambda s: s.lower() == "true", default=True)
    parser.add_argument("--cache_lickety", type=lambda s: s.lower() == "true", default=True)
    parser.add_argument("--cache_packbits", type=lambda s: s.lower() == "true", default=True)
    parser.add_argument("--cache_key_mode", type=str, default="bitvector")
    parser.add_argument("--stop_caching_at_depth", type=int, default=0)

    args = parser.parse_args()
    duration_s, peak_mb, delta_mb, n_trees, true_count, slack_counts = main(
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
        multiplicative_slack=args.multiplicative_slack,
        cache_greedy=args.cache_greedy,
        cache_lickety=args.cache_lickety,
        cache_packbits=args.cache_packbits,
        cache_key_mode=args.cache_key_mode,
        stop_caching_at_depth=args.stop_caching_at_depth
    )

    def _slug(v):
        if isinstance(v, bool):
            return "1" if v else "0"
        s = str(v)
        s = s.replace(os.sep, "_").replace("/", "_").replace(" ", "")
        s = s.replace(".", "p").replace("+", "plus").replace("-", "m")
        return s

    stem = Path(args.data).stem
    fname = (
        f"{args.algo}"
        f"__data-{_slug(stem)}"
        f"__reg-{_slug(args.reg)}"
        f"__depth-{_slug(args.depth)}"
        f"__mult-{_slug(args.mult)}"
        f"__lookahead-{_slug(args.lookahead_k)}"
        f"__prune-{_slug(args.prune_style)}"
        f"__consLook-{_slug(args.consistent_lookahead)}"
        f"__btg-{_slug(args.better_than_greedy)}"
        f"__gosdt-{_slug(args.use_gosdt_objective)}"
        f"__tryGreedy-{_slug(args.try_greedy_first)}"
        f"__trie-{_slug(args.trie_cache_strategy)}"
        f"__mslack-{_slug(args.multiplicative_slack)}"
        f"__cg-{_slug(args.cache_greedy)}"
        f"__cl-{_slug(args.cache_lickety)}"
        f"__cp-{_slug(args.cache_packbits)}"
        f"__ckm-{_slug(args.cache_key_mode)}"
        f"__stopCache-{_slug(args.stop_caching_at_depth)}"
        f".csv"
    )

    out_dir = Path("OutputsThroughput")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / fname

    multipliers = [1.01, 1.02, 1.03, 1.04, 1.05, 1.1, 1.15, 1.2, 1.25, 1.5]
    slack_counts = slack_counts or [None] * len(multipliers)
    slack_csv_fields = ",".join(f"count_{str(m).replace('.', 'p')}" for m in multipliers)
    slack_csv_values = ",".join("" if v is None else str(v) for v in slack_counts)

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("duration_s,peak_mb,delta_mb,num_trees,true_rashomon_count," + slack_csv_fields + "\n")
        f.write(f"{duration_s:.6f},{peak_mb:.2f},{delta_mb:.2f},{n_trees},"
                f"{'' if true_count is None else true_count},"
                f"{slack_csv_values}\n")
