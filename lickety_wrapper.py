'''
g++ lickety_resplit.cpp \
  -std=c++17 \
  -O3 -DNDEBUG \
  -march=core-avx-i \
  -mtune=generic \
  -flto \
  -funroll-loops \
  -fPIC -shared \
  -o liblickety2.so \
  -lm
'''
import sys
import os
import time
import numpy as np
import pandas as pd
import ctypes
from pathlib import Path
from typing import Optional
import resource


DEFAULT_CSV = "Datasets/Processed/magic_binarized.csv"
DEFAULT_LIB = "liblickety2.so"
DEFAULT_LAMBDA = 0.01
DEFAULT_DEPTH = 5
DEFAULT_MULT = 0.01
DEFAULT_KEYS = "hash"          # {"hash","exact"}
DEFAULT_TRIE_CACHE = False     # True/False
DEFAULT_LOOKAHEAD = 1
DEFAULT_MULT_SLACK = 0.0

def _default_lib_name():
    return DEFAULT_LIB

KEYS_HASH = 0
KEYS_EXACT = 1

def _peak_rss_bytes() -> int:
    ru = resource.getrusage(resource.RUSAGE_SELF)
    rss = ru.ru_maxrss
    rss *= 1024  # KiB -> bytes
    return int(rss)


def _fmt_bytes(b: int) -> str:
    v = float(b)
    for unit in ["B", "KiB", "MiB", "GiB", "TiB"]:
        if v < 1024.0 or unit == "TiB":
            return f"{v:.2f} {unit}"
        v /= 1024.0
    return f"{v:.2f} B"


class LicketyRESPLIT_CPP:
    def __init__(self, lib_path: Optional[str] = None):
        if lib_path is None:
            lib_path = _default_lib_name()
        lib_path = Path(lib_path).resolve()
        if not lib_path.exists():
            raise FileNotFoundError(f"Shared library not found: {lib_path}")

        self.lib = ctypes.CDLL(str(lib_path))

        self.lib.create_model.restype = ctypes.c_void_p

        self.lib.delete_model.argtypes = [ctypes.c_void_p]
        self.lib.delete_model.restype = None

        self.lib.fit_model.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_uint8),
            ctypes.c_int,
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_int),
            ctypes.c_double,
            ctypes.c_int,
            ctypes.c_double,
            ctypes.c_double,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
        ]
        self.lib.fit_model.restype = None

        self.lib.get_tree_count.argtypes = [ctypes.c_void_p]
        self.lib.get_tree_count.restype = ctypes.c_uint64

        self.lib.get_min_objective.argtypes = [ctypes.c_void_p]
        self.lib.get_min_objective.restype = ctypes.c_int

        self.lib.get_root_hist_size.argtypes = [ctypes.c_void_p]
        self.lib.get_root_hist_size.restype = ctypes.c_size_t

        self.lib.get_root_histogram.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_int),
            ctypes.POINTER(ctypes.c_uint64),
        ]
        self.lib.get_root_histogram.restype = None

        self.model = None

        # keep the data alive while C++ is using it
        self._X_hold = None
        self._y_hold = None

    def fit(
        self,
        X,
        y,
        lambda_reg: float = DEFAULT_LAMBDA,
        depth_budget: int = DEFAULT_DEPTH,
        rashomon_mult: float = DEFAULT_MULT,
        keys: str = DEFAULT_KEYS,
        trie_cache: bool = DEFAULT_TRIE_CACHE,
        multiplicative_slack: float = DEFAULT_MULT_SLACK,
        lookahead_k: int = DEFAULT_LOOKAHEAD,
    ) -> dict:

        key_mode = KEYS_HASH if str(keys).lower() == "hash" else KEYS_EXACT
        trie_flag = 1 if bool(trie_cache) else 0

        X = np.asarray(X, dtype=np.uint8)
        y = np.asarray(y, dtype=np.int32)
        n_samples, n_features = X.shape

        X_f = np.asfortranarray(X)
        self._X_hold = X_f
        self._y_hold = y

        if self.model is not None:
            self.lib.delete_model(self.model)

        self.model = self.lib.create_model()
        if not self.model:
            raise RuntimeError("create_model() failed")

        X_ptr = X_f.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8))
        y_ptr = y.ctypes.data_as(ctypes.POINTER(ctypes.c_int))

        rss_before = _peak_rss_bytes()
        t0 = time.perf_counter()

        self.lib.fit_model(
            self.model,
            X_ptr,
            ctypes.c_int(n_samples),
            ctypes.c_int(n_features),
            y_ptr,
            ctypes.c_double(lambda_reg),
            ctypes.c_int(depth_budget),
            ctypes.c_double(rashomon_mult),
            ctypes.c_double(multiplicative_slack),
            ctypes.c_int(key_mode),
            ctypes.c_int(trie_flag),
            ctypes.c_int(lookahead_k),
        )

        t1 = time.perf_counter()
        rss_after = _peak_rss_bytes()

        return {
            "fit_sec": t1 - t0,
            "peak_rss_before": rss_before,
            "peak_rss_after": rss_after,
            "peak_rss_delta": (rss_after - rss_before)
            if rss_before >= 0 and rss_after >= 0
            else -1,
        }

    def count_trees(self) -> int:
        return int(self.lib.get_tree_count(self.model))

    def get_min_objective(self) -> int:
        return int(self.lib.get_min_objective(self.model))

    def get_root_histogram(self):
        size = int(self.lib.get_root_hist_size(self.model))
     
        objs = (ctypes.c_int * size)()
        cnts = (ctypes.c_uint64 * size)()

        self.lib.get_root_histogram(self.model, objs, cnts)

        return [(int(objs[i]), int(cnts[i])) for i in range(size)]

    def __del__(self):
        self.lib.delete_model(self.model)
        self.model = None

if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description="C++ LicketyRESPLIT")
    p.add_argument("--csv", type=str, default=DEFAULT_CSV, help="Path to CSV (last col = label).")
    p.add_argument("--lib", type=str, default=_default_lib_name(), help="Shared library path.")
    p.add_argument("--lambda", dest="lambda_reg", type=float, default=DEFAULT_LAMBDA)
    p.add_argument("--depth", type=int, default=DEFAULT_DEPTH)
    p.add_argument("--mult", dest="mult", type=float, default=DEFAULT_MULT)
    p.add_argument("--keys", choices=["hash", "exact"], default=DEFAULT_KEYS)
    p.add_argument("--trie-cache", choices=["on", "off"], default=("on" if DEFAULT_TRIE_CACHE else "off"))
    p.add_argument("--lookahead", type=int, default=DEFAULT_LOOKAHEAD)
    p.add_argument("--mult-slack", type=float, default=DEFAULT_MULT_SLACK)
    args = p.parse_args()

    csv_path = Path(args.csv).resolve()
    if not csv_path.exists():
        print(f"CSV not found: {csv_path}")
        sys.exit(2)

    df = pd.read_csv(csv_path)
    X = df.iloc[:, :-1].to_numpy(dtype=np.uint8)
    y = df.iloc[:, -1].to_numpy(dtype=np.int32)

    model = LicketyRESPLIT_CPP(args.lib)
    prof = model.fit(
        X, y,
        lambda_reg=args.lambda_reg,
        depth_budget=args.depth,
        rashomon_mult=args.mult,
        keys=args.keys,
        trie_cache=(args.trie_cache == "on"),
        multiplicative_slack=args.mult_slack,
        lookahead_k=args.lookahead
    )

    print("trees:", model.count_trees())
    print("min_objective:", model.get_min_objective())
    print(f"fit_sec: {prof['fit_sec']:.6f}")
    print(f"peak_rss_before: {_fmt_bytes(prof['peak_rss_before'])}")
    print(f"peak_rss_after:  {_fmt_bytes(prof['peak_rss_after'])}")
    print(f"peak_rss_delta:  {_fmt_bytes(prof['peak_rss_delta'])}")
    print("root_histogram:", model.get_root_histogram())
