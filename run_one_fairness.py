#!/usr/bin/env python3
import argparse, time
import pandas as pd
import numpy as np
import gc
from memory_profiler import memory_usage
from licketyresplit import LicketyRESPLIT
from treefarms import TREEFARMS
from tqdm import tqdm
from fairlearn.metrics import equalized_odds_difference, demographic_parity_difference
import pickle
from split import LicketySPLIT
from sklearn.tree import DecisionTreeClassifier

# -------------- helpers --------------

def load_dataset(path):
    df = pd.read_csv(path)
    X, y = df.iloc[:, :-1], df.iloc[:, -1]
    print(f"Loaded {df.shape[0]} rows, {df.shape[1]-1} features from {path}")
    return X, y
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
    return dt, n_trees, label, model

# -------------- main --------------
def main(data_path, algo="lickety", reg=0.01, depth=10, mult=0.01, better_than_greedy = False, lookahead_k = 1, prune_style = "H", consistent_lookahead=False, try_greedy_first=False, trie_cache_strategy = "compact", multiplicative_slack=0.00, cache_greedy=True, cache_lickety=True, cache_packbits=True, cache_key_mode="bitvector", stop_caching_at_depth=-1,
         protected_col_name="race:African-American"):
    result_json = {}
    result_json['data'] = data_path
    result_json['algo'] = algo
    result_json['reg'] = reg
    result_json['depth'] = depth
    result_json['mult'] = mult
    # result_json['better_than_greedy'] = better_than_greedy
    # result_json['lookahead_k'] = lookahead_k
    # result_json['prune_style'] = prune_style
    # result_json['consistent_lookahead'] = consistent_lookahead
    # result_json['try_greedy_first'] = try_greedy_first
    # result_json['trie_cache_strategy'] = trie_cache_strategy
    # result_json['multiplicative_slack'] = multiplicative_slack
    # result_json['cache_greedy'] = cache_greedy
    # result_json['cache_lickety'] = cache_lickety
    # result_json['cache_packbits'] = cache_packbits
    # result_json['cache_key_mode'] = cache_key_mode
    # result_json['stop_caching_at_depth'] = stop_caching_at_depth
    # =======================================

    X, y = load_dataset(data_path)
    protected_attr = X[protected_col_name]

    if algo == "lickety":
        best_obj = None

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
    
    elif 'bootstrap' in algo:
        X_arr = X.to_numpy(copy=False) if hasattr(X, "to_numpy") else np.asarray(X)
        y_arr = y.to_numpy(copy=False) if hasattr(y, "to_numpy") else np.asarray(y)
        X_bool = np.asfortranarray(X_arr != 0)
        y_uint8 = np.ascontiguousarray((y_arr != 0).astype(np.uint8, copy=False))
    else:
        raise ValueError(f"Unknown algo: {algo}")
    
    if 'bootstrap' not in algo:
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

    if algo == 'treefarms':
        equal_odds_list = []
        dem_parity_list = []
        acc_list = []
        all_list = []
        for i in range(n_trees):
            tree = model[i]
            y_pred = tree.predict(X)
            equal_odds_list.append(equalized_odds_difference(y, y_pred, sensitive_features=X['race:African-American']))
            dem_parity = demographic_parity_difference(y, y_pred, sensitive_features=X['race:African-American'])
            dem_parity_list.append(dem_parity)
            acc_list.append((y == y_pred).mean())
            all_list.append( (acc_list[-1], equal_odds_list[-1], dem_parity_list[-1]) )
    
    elif algo == 'lickety':
        equal_odds_list = []
        dem_parity_list = []
        acc_list = []
        all_list = []
        for i in tqdm(range(n_trees)):
            y_pred = model.trie.get_predictions(i, X_bool)
            equal_odds_list.append(equalized_odds_difference(y_uint8, y_pred, sensitive_features=protected_attr))
            dem_parity = demographic_parity_difference(y_uint8, y_pred, sensitive_features=protected_attr)
            dem_parity_list.append(dem_parity)
            acc_list.append((y_uint8 == y_pred).mean())
            all_list.append( (acc_list[-1], equal_odds_list[-1], dem_parity_list[-1]) )

    elif 'bootstrap' in algo:
        equal_odds_list = []
        dem_parity_list = []
        acc_list = []
        all_list = []
        for i in tqdm(range(100)):
            np.random.seed(i)
            row_indices = np.random.choice(X_bool.shape[0], size=X_bool.shape[0], replace=True)
            X_sample = X_bool[row_indices]
            y_sample = y_uint8[row_indices]
            # fit tree (TODO arguments)
            tree = DecisionTreeClassifier(max_depth=depth, min_samples_leaf=reg) if algo == 'bootstrap_greedy' else LicketySPLIT(full_depth_budget=depth, reg=reg)
            if 'greedy' in algo:
                tree.fit(X_sample, y_sample)
                y_pred = tree.predict(X_bool)
            else:
                tree.fit(pd.DataFrame(X_sample), pd.Series(y_sample))
                y_pred = tree.predict(pd.DataFrame(X_bool))
            equal_odds_list.append(equalized_odds_difference(y_uint8, y_pred, sensitive_features=protected_attr))
            dem_parity = demographic_parity_difference(y_uint8, y_pred, sensitive_features=protected_attr)
            dem_parity_list.append(dem_parity)
            acc_list.append((y_uint8 == y_pred).mean())
            all_list.append( (acc_list[-1], equal_odds_list[-1], dem_parity_list[-1]) )
    
    result_json['equalized_odds_difference_best'] = min(equal_odds_list)
    result_json['equalized_odds_difference_worst'] = max(equal_odds_list)
    result_json['demographic_parity_difference_best'] = min(dem_parity_list)
    result_json['demographic_parity_difference_worst'] = max(dem_parity_list)
    result_json['accuracy_best'] = max(acc_list)
    result_json['accuracy_worst'] = min(acc_list)

    if 'bootstrap' in algo:
        duration_s = None
        peak_mb = None
        n_trees = 100
    result_json['time_sec'] = duration_s
    result_json['peak_rss_mb'] = peak_mb
    result_json['num_trees'] = n_trees

    pd.DataFrame(result_json, index=[0]).to_csv(f"results/fairness_{data_path.split('/')[-1].replace('.csv','')}_{algo}_{depth}_{reg}_{mult}_{protected_col_name[-4:]}.csv", index=False)

    result_json['ordered_by_equalized_odds'] = sorted(equal_odds_list)
    result_json['ordered_by_demographic_parity'] = sorted(dem_parity_list)
    result_json['ordered_by_accuracy'] = sorted(all_list, key = lambda x: x[0])
    with open(f"results/fairness_detailed_{data_path.split('/')[-1].replace('.csv','')}_{algo}_{depth}_{reg}_{mult}_{protected_col_name[-4:]}.pkl", "wb") as f:
        pickle.dump(result_json, f)

    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="data/compas_w_demographics.csv")
    parser.add_argument("--algo", type=str, choices=["lickety", "resplit", "treefarms", "bootstrap_greedy", "bootstrap_lickety"], default="lickety")
    parser.add_argument("--reg", type=float, default=0.01)
    parser.add_argument("--depth", type=int, default=5)
    parser.add_argument("--mult", type=float, default=0.03)
    parser.add_argument("--lookahead_k", type=int, default=1)
    parser.add_argument("--prune_style", type=str, default="H")
    parser.add_argument("--consistent_lookahead", type=lambda s: s.lower() == "true", default=False)
    parser.add_argument("--better_than_greedy", type=lambda s: s.lower() == "true", default=False)
    parser.add_argument("--try_greedy_first", type=lambda s: s.lower() == "true", default=False)
    parser.add_argument("--trie_cache_strategy", type=str, default="compact")
    parser.add_argument("--multiplicative_slack", type=float, default=0.03)
    parser.add_argument("--cache_greedy", type=lambda s: s.lower() == "true", default=True)
    parser.add_argument("--cache_lickety", type=lambda s: s.lower() == "true", default=True)
    parser.add_argument("--cache_packbits", type=lambda s: s.lower() == "true", default=True)
    parser.add_argument("--cache_key_mode", type=str, default="bitvector")
    parser.add_argument("--stop_caching_at_depth", type=int, default=0)
    parser.add_argument("--protected_col_name", type=str, default="race:African-American")

    args = parser.parse_args()
    main(
        data_path=args.data,
        algo=args.algo,
        reg=args.reg,
        depth=args.depth,
        mult=args.mult,
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
        stop_caching_at_depth=args.stop_caching_at_depth,
        protected_col_name=args.protected_col_name,
    )