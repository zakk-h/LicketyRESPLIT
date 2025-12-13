import sys
import time
import resource
from pathlib import Path

import numpy as np
import pandas as pd
from licketyresplit import LicketyRESPLIT

DEFAULT_CSV = "Datasets/Processed/churn_binarized.csv"
DEFAULT_LAMBDA = 0.002
DEFAULT_DEPTH = 10
DEFAULT_MULT = 0.0003
DEFAULT_KEYS = "hash"          # {"hash","exact"}
DEFAULT_TRIE_CACHE = False     # True/False
DEFAULT_LOOKAHEAD = 0
DEFAULT_MULT_SLACK = 0.0


def _peak_rss_bytes() -> int:
    """Peak RSS in bytes for this process (since start)."""
    ru = resource.getrusage(resource.RUSAGE_SELF)
    rss = ru.ru_maxrss
    # On Linux ru_maxrss is KiB, so convert to bytes
    rss *= 1024
    return int(rss)


def _fmt_bytes(b: int) -> str:
    v = float(b)
    for unit in ["B", "KiB", "MiB", "GiB", "TiB"]:
        if v < 1024.0 or unit == "TiB":
            return f"{v:.2f} {unit}"
        v /= 1024.0
    return f"{v:.2f} B"


def main():
    import argparse

    p = argparse.ArgumentParser(description="LicketyRESPLIT (Python package)")
    p.add_argument("--csv", type=str, default=DEFAULT_CSV,
                   help="Path to CSV (last col = label).")
    p.add_argument("--lambda", dest="lambda_reg", type=float,
                   default=DEFAULT_LAMBDA,
                   help="Lambda regularization (lambda_reg).")
    p.add_argument("--depth", type=int, default=DEFAULT_DEPTH,
                   help="Depth budget.")
    p.add_argument("--mult", dest="mult", type=float,
                   default=DEFAULT_MULT,
                   help="Rashomon multiplier (rashomon_mult).")
    p.add_argument("--keys", choices=["hash", "exact"],
                   default=DEFAULT_KEYS,
                   help="Key mode for caching.")
    p.add_argument("--trie-cache", choices=["on", "off"],
                   default=("on" if DEFAULT_TRIE_CACHE else "off"),
                   help="Enable trie cache.")
    p.add_argument("--lookahead", type=int, default=DEFAULT_LOOKAHEAD,
                   help="Lookahead k (oracle strength).")
    p.add_argument("--mult-slack", type=float, default=DEFAULT_MULT_SLACK,
                   help="Multiplicative slack on Rashomon bound.")
    args = p.parse_args()

    csv_path = Path(args.csv).resolve()
    if not csv_path.exists():
        print(f"CSV not found: {csv_path}")
        sys.exit(2)

    df = pd.read_csv(csv_path)
    X = df.iloc[:, :-1].to_numpy(dtype=np.uint8)
    y = df.iloc[:, -1].to_numpy(dtype=np.int32)

    model = LicketyRESPLIT()

    rss_before = _peak_rss_bytes()
    t0 = time.perf_counter()

    model.fit(
        X,
        y,
        lambda_reg=args.lambda_reg,
        depth_budget=args.depth,
        rashomon_mult=args.mult,
        multiplicative_slack=args.mult_slack,
        key_mode=args.keys,
        trie_cache_enabled=(args.trie_cache == "on"),
        lookahead_k=args.lookahead,
        use_multipass=True,
    )

    t1 = time.perf_counter()
    rss_after = _peak_rss_bytes()

    peak_delta = (rss_after - rss_before) if rss_before >= 0 and rss_after >= 0 else -1

    print("trees:", model.count_trees())
    print("min_objective:", model.get_min_objective())
    print(f"fit_sec: {t1 - t0:.6f}")
    print(f"peak_rss_before: {_fmt_bytes(rss_before)}")
    print(f"peak_rss_after:  {_fmt_bytes(rss_after)}")
    print(f"peak_rss_delta:  {_fmt_bytes(peak_delta)}")

    hist = model.get_root_histogram()
    print("root_histogram:", hist)


if __name__ == "__main__":
    main()
