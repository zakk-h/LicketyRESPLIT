from ucimlrepo import fetch_ucirepo
import numpy as np
from sklearn.datasets import fetch_openml
import pandas as pd
from sklearn.datasets import fetch_covtype
from sklearn.datasets import fetch_covtype
from sklearn.model_selection import train_test_split
import time

from licketyresplit_rashomon_importance_distribution import RashomonImportanceDistribution as LicketyRID
TreeFarmsRID = LicketyRID  # given that TreeFARMS has a bug, we are going to view Lickety with max lookahead as TreeFARMS because it is optimal.
from resplit_rashomon_importance_distribution import RashomonImportanceDistribution as ResplitRID

if False:
    spambase = fetch_ucirepo(id=94)
    X = spambase.data.features
    y = spambase.data.targets
    df = pd.concat([X, y], axis=1)  # y already binary

if False:
    bike = fetch_ucirepo(id=275)
    X = bike.data.features
    y = bike.data.targets
    X = pd.get_dummies(X, columns=['season', 'mnth', 'weekday', 'weathersit'], drop_first=False)
    X['dteday'] = pd.to_datetime(X['dteday']).dt.year
    label_col = "cnt"
    label_quantile = 0.50
    thr = y[label_col].quantile(label_quantile)
    y_bin = (y[label_col] >= thr).astype(int).rename("label")
    df = pd.concat([X, y_bin], axis=1)

if False:
    X_raw, y_raw = fetch_openml(data_id=42193, as_frame=True, return_X_y=True)
    df_all = X_raw.copy() # compas
    df_all["label"] = y_raw
    df_all = df_all.dropna(axis=0, how="any").reset_index(drop=True) # drop rows with any NA
    X_clean = df_all.drop(columns=["label"])
    y_clean = df_all["label"]
    X = pd.get_dummies(
        X_clean,
        drop_first=False,
        dtype="uint8"
    )
    y = pd.to_numeric(y_clean, errors="coerce").astype("uint8")
    df = pd.concat([X.reset_index(drop=True), y.rename("label").reset_index(drop=True)], axis=1)

if False:
    adult = fetch_openml("adult", version=2, as_frame=True)
    X_raw = adult.data.copy()
    y_raw = adult.target.copy()

    X_raw = X_raw.replace('?', pd.NA)
    X_raw = X_raw.dropna()
    y_raw = y_raw.loc[X_raw.index]
    X = pd.get_dummies(X_raw, drop_first=False)
    y = (y_raw.astype(str).str.contains(">50K")).astype(int)
    df = pd.concat([X.reset_index(drop=True), y.rename("income_gt_50k").reset_index(drop=True)], axis=1)

if True:
    cov = fetch_covtype(as_frame=True)
    X = cov.data
    y = cov.target
    y_bin = (y == 2).astype(np.uint8).rename("label")
    df = pd.concat([X.reset_index(drop=True), y_bin.reset_index(drop=True)], axis=1)


common_kwargs = dict(
    input_df=df,
    binning_map=None, 
    db=5,
    lam=0.01,
    eps=0.01,
    vi_metric='sub_mr',
    dataset_name='covertype6',  # share bootstraps
    n_resamples=10,
    verbose=False,
    max_par_for_gosdt=2,
    allow_binarize_internally=True
)

t0 = time.perf_counter()
LRID = LicketyRID(**common_kwargs, lickety_lookahead=1)
t1 = time.perf_counter()
print(f"LicketyRID (lookahead=1) runtime: {t1 - t0:.3f} sec")

# we alias to keep the "TreeFarms" name externally, but underneath it’s Lickety with lh=db
t0 = time.perf_counter()
TRID = TreeFarmsRID(**common_kwargs, lickety_lookahead=common_kwargs["db"])
t1 = time.perf_counter()
print(f"TreeFarmsRID (optimal; lookahead=depth) runtime: {t1 - t0:.3f} sec")

t0 = time.perf_counter()
RRID = ResplitRID(**common_kwargs)  # RESPLIT uses TREEFARMS fill + lookahead=ceil(d/2) internally
t1 = time.perf_counter()
print(f"ResplitRID runtime: {t1 - t0:.3f} sec")

for v in range(TRID.n_vars):
    col_name = df.columns[v]

    # TreeFarms (lookahead = db, i.e., optimal)
    t_low, t_high = TRID.bwr(v)
    t_mean, t_median = TRID.mean(v), TRID.median(v)

    # Lickety (lookahead = 1)
    l_low, l_high = LRID.bwr(v)
    l_mean, l_median = LRID.mean(v), LRID.median(v)

    # RESPLIT
    r_low, r_high = RRID.bwr(v)
    r_mean, r_median = RRID.mean(v), RRID.median(v)

    print(f"Variable {v} ({col_name}) --------------")
    print(f"TreeFarms (opt): range=({t_low:.4f}, {t_high:.4f}), mean={t_mean:.4f}, median={t_median:.4f}")
    print(f"Lickety (lh=1) : range=({l_low:.4f}, {l_high:.4f}), mean={l_mean:.4f}, median={l_median:.4f}")
    print(f"RESPLIT (lh≈d/2): range=({r_low:.4f}, {r_high:.4f}), mean={r_mean:.4f}, median={r_median:.4f}")
    print()

from scipy.stats import pearsonr, spearmanr

tree_means = np.array([TRID.mean(v) for v in range(TRID.n_vars)])
lickety_means = np.array([LRID.mean(v) for v in range(LRID.n_vars)])

corr, pval = pearsonr(tree_means, lickety_means)
print("=== Linear Correlation: TreeFarms (opt) vs Lickety (lh=1) ===")
print(f"Pearson r = {corr:.6f}, p-value = {pval:.4e}")

rank_corr, rank_pval = spearmanr(tree_means, lickety_means)
print(f"Spearman rho = {rank_corr:.6f}, p-value = {rank_pval:.4e}")

top20_idx = np.argsort(tree_means)[-20:]
tree_top20 = tree_means[top20_idx]
lickety_top20 = lickety_means[top20_idx]

corr, pval = pearsonr(tree_top20, lickety_top20)
print("=== Linear Correlation on Top 20 (TreeFarms opt vs Lickety) ===")
print(f"Pearson r = {corr:.6f}, p-value = {pval:.6e}")

rank_corr, rank_pval = spearmanr(tree_top20, lickety_top20)
print(f"Spearman rho = {rank_corr:.6f}, p-value = {rank_pval:.6e}")
print()

resplit_means = np.array([RRID.mean(v) for v in range(RRID.n_vars)])

corr, pval = pearsonr(tree_means, resplit_means)
print("=== Linear Correlation: TreeFarms (opt) vs RESPLIT ===")
print(f"Pearson r = {corr:.6f}, p-value = {pval:.4e}")

rank_corr, rank_pval = spearmanr(tree_means, resplit_means)
print(f"Spearman rho = {rank_corr:.6f}, p-value = {rank_pval:.4e}")

resplit_top20 = resplit_means[top20_idx]

corr, pval = pearsonr(tree_top20, resplit_top20)
print("=== Linear Correlation on Top 20 (TreeFarms opt vs RESPLIT) ===")
print(f"Pearson r = {corr:.6f}, p-value = {pval:.6e}")

rank_corr, rank_pval = spearmanr(tree_top20, resplit_top20)
print(f"Spearman rho = {rank_corr:.6f}, p-value = {rank_pval:.6e}")