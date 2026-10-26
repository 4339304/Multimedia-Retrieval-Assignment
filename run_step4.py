"""
Step 4 (first part): feature normalization, query shapes and the Euclidean distance.

Run from the project folder:
    python run_step4.py --build                  normalize the feature table of Step 3.2 and save it
    python run_step4.py --show                   print the normalized feature table (first rows and summary)
    python run_step4.py --query Data/Mug/D00685.obj -k 5
                                                 query with a mesh file (any OBJ, also outside the database)
    python run_step4.py --query Mug/D00685.obj   query with a shape of the database (class/file)
    python run_step4.py --compare                compare the normalization and weighting options

Input:  report/data/step3/features_step3_2.csv (Step 3.2)
Output: report/data/step4/features_normalized.csv     normalized feature table of the database
        report/data/step4/normalization_params.csv    parameters, used again for every query shape
        report/data/step4/query_shapes.csv            shapes taken out of the database to be used as queries

A query mesh goes through the same steps as the database: cleaning, orientation and resampling (Step 2.3),
the four-step normalization (Steps 2.5 and 3.1) and the descriptors (Step 3.2). Its features are normalized
with the parameters of the database. The distance is the Euclidean distance between the normalized feature
vectors, where every descriptor is weighted so that all descriptors have the same spread of distances
(technical tips 4). Other distance functions for the histograms can be passed as hist_metric.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist

from mr_io import PROJECT_ROOT, read_obj_raw
from mr_normalize import STEP3_DIR, normalize
from mr_resample import clean_mesh, fix_orientation, resample_mesh
from run_s3_2 import FEATURES_CSV, NAMES, features_of

OUT = PROJECT_ROOT / "report" / "data" / "step4"
NORM_CSV = OUT / "features_normalized.csv"
PARAMS_CSV = OUT / "normalization_params.csv"
QUERY_CSV = OUT / "query_shapes.csv"

GLOBALS = ["surface_area", "sphericity", "rectangularity_3d", "diameter", "convexity", "eccentricity"]
# descriptors whose values cover several orders of magnitude (the diameter only goes from 1 to 1.6)
LOG_FEATURES = ["surface_area", "sphericity", "rectangularity_3d", "convexity", "eccentricity"]
HIST = {n: [f"{n}_bin_{i:02d}" for i in range(20)] for n in NAMES}
DESCRIPTORS = GLOBALS + NAMES
QUERY_CLASSES = ["Bicycle", "Bird", "Car", "Chess", "Guitar", "Humanoid", "Jet", "Mug", "Sword", "Tree"]
PAIR_SAMPLES = 200_000   # random pairs of database shapes used to measure the spread of each distance


# ------------------------------------------------------------------ query shapes taken out of the database
def near_duplicates(table, limit=0.02):
    """Shapes whose histograms are almost the same as those of another shape (the same model stored twice):
    average difference per histogram (half the sum of the absolute bin differences) below `limit`."""
    H = table[sum(HIST.values(), [])].to_numpy(float)
    D = cdist(H, H, "cityblock") / (2 * len(NAMES))
    np.fill_diagonal(D, np.inf)
    return set(table.loc[(D < limit).any(axis=1), "class"] + "/" + table.loc[(D < limit).any(axis=1), "file"])


def choose_query_shapes(table, seed=0):
    """One random shape of each class in QUERY_CLASSES, without a copy in the database; these are
    removed from the database and used as query shapes from outside the database."""
    rng = np.random.default_rng(seed)
    dup = near_duplicates(table)
    rows = []
    for c in QUERY_CLASSES:
        files = sorted(f for f in table.loc[table["class"] == c, "file"] if f"{c}/{f}" not in dup)
        rows.append({"class": c, "file": files[rng.integers(len(files))]})
    return pd.DataFrame(rows)


def split_database(table):
    """(database, query shapes): the query shapes of QUERY_CSV are not part of the database."""
    q = pd.read_csv(QUERY_CSV)
    key = table["class"] + "/" + table["file"]
    held = key.isin(q["class"] + "/" + q["file"])
    return table[~held].reset_index(drop=True), table[held].reset_index(drop=True)


# ------------------------------------------------------------------ normalization
def transform(df):
    """Logarithm of the features whose values cover several orders of magnitude."""
    df = df.copy()
    numeric = GLOBALS + sum(HIST.values(), [])
    df[numeric] = df[numeric].apply(pd.to_numeric)
    for c in LOG_FEATURES:
        df[c] = np.log10(df[c])
    return df


def fit_normalization(db, method="standard"):
    """Parameters of the normalization, computed on the database shapes only (not on the query shapes)."""
    t = transform(db)
    rows = []
    for c in GLOBALS:
        x = t[c].dropna()
        rows.append({"feature": c, "log10": c in LOG_FEATURES, "method": method,
                     "mean": x.mean(), "std": x.std(), "min": x.min(), "max": x.max()})
    return pd.DataFrame(rows).set_index("feature")


def apply_normalization(df, params):
    """Standardization (or min-max) of the global descriptors; the histograms already sum to one."""
    t = transform(df)
    for c in GLOBALS:
        p = params.loc[c]
        if p["method"] == "standard":
            t[c] = (t[c] - p["mean"]) / p["std"]
        else:
            t[c] = (t[c] - p["min"]) / (p["max"] - p["min"])
    return t


# ------------------------------------------------------------------ distances
def euclidean(a, b):
    return np.sqrt(((a - b) ** 2).sum(axis=-1))


def descriptor_distances(q, X, hist_metric=euclidean):
    """Distance of the query row q to every row of X, separately for every descriptor: an (n, 11) array.
    A global descriptor gives |q - x|; a histogram gives hist_metric between the two histograms."""
    D = np.empty((len(X), len(DESCRIPTORS)))
    for k, c in enumerate(GLOBALS):
        D[:, k] = np.abs(X[c].to_numpy(float) - float(q[c]))
    for k, n in enumerate(NAMES, start=len(GLOBALS)):
        D[:, k] = hist_metric(X[HIST[n]].to_numpy(float), q[HIST[n]].to_numpy(float)[None, :])
    return D


def fit_distance_scales(db_norm, hist_metric=euclidean, seed=0):
    """Standard deviation of every descriptor distance over random pairs of database shapes."""
    rng = np.random.default_rng(seed)
    i = rng.integers(len(db_norm), size=PAIR_SAMPLES)
    j = rng.integers(len(db_norm), size=PAIR_SAMPLES)
    keep = i != j
    A, B = db_norm.iloc[i[keep]].reset_index(drop=True), db_norm.iloc[j[keep]].reset_index(drop=True)
    scales = {}
    for c in GLOBALS:
        scales[c] = np.nanstd(np.abs(A[c].to_numpy(float) - B[c].to_numpy(float)))
    for n in NAMES:
        scales[n] = np.std(hist_metric(A[HIST[n]].to_numpy(float), B[HIST[n]].to_numpy(float)))
    return pd.Series(scales)


def combined_distance(D, scales):
    """Euclidean combination of the weighted descriptor distances (every shape has all descriptors)."""
    W = D / scales[DESCRIPTORS].to_numpy(float)
    return np.sqrt((W ** 2).sum(axis=1))


# ------------------------------------------------------------------ query shape: same steps as the database
def query_features(path):
    """Features (not yet normalized) of any OBJ mesh: Step 2.3, Steps 2.5 + 3.1 and Step 3.2."""
    path = Path(path)
    verts, sizes, idx = read_obj_raw(path)
    V, F, _ = clean_mesh(verts, sizes, idx)
    F, _ = fix_orientation(V, F)
    V, F, _ = resample_mesh(V, F)
    F, _ = fix_orientation(V, F)
    V, F = normalize(V, F)
    return pd.Series(features_of(V, F, path.parent.name, path.name)), (V, F)


def load_database():
    norm = pd.read_csv(NORM_CSV)
    params = pd.read_csv(PARAMS_CSV, index_col=0)
    scales = params["distance_scale"]
    return norm, params, scales


def query(features, k=5, hist_metric=euclidean):
    """The k database shapes closest to a query (a row of features, not normalized). A query that is itself
    a database shape is left out of its own result. For another histogram distance than the Euclidean one,
    the weights of the descriptors are measured again for that distance."""
    norm, params, scales = load_database()
    if hist_metric is not euclidean:
        scales = fit_distance_scales(norm, hist_metric)
    q = apply_normalization(features.to_frame().T, params).iloc[0]
    d = combined_distance(descriptor_distances(q, norm, hist_metric), scales)
    d[((norm["class"] == q["class"]) & (norm["file"] == q["file"])).to_numpy()] = np.inf
    order = np.argsort(d, kind="stable")[:k]
    return pd.DataFrame({"rank": np.arange(1, len(order) + 1), "class": norm["class"].iloc[order].values,
                         "file": norm["file"].iloc[order].values, "distance": d[order]})


def features_for(name):
    """Features of a query given as a mesh file, or as class/file of a shape in the feature table."""
    path = Path(name)
    if path.exists():
        return query_features(path)[0]
    table = pd.read_csv(FEATURES_CSV)
    cls, file = name.replace("\\", "/").split("/")[-2:]
    hit = table[(table["class"] == cls) & (table["file"] == file)]
    if not len(hit):
        raise FileNotFoundError(f"{name} is neither a file nor a shape of the feature table")
    return hit.iloc[0]


# ------------------------------------------------------------------ build, show, compare
def build():
    table = pd.read_csv(FEATURES_CSV)
    OUT.mkdir(parents=True, exist_ok=True)
    if not QUERY_CSV.exists():
        choose_query_shapes(table).to_csv(QUERY_CSV, index=False)
    db, held = split_database(table)
    params = fit_normalization(db)
    norm = apply_normalization(db, params)[["class", "file"] + GLOBALS + sum(HIST.values(), [])]
    scales = fit_distance_scales(norm)
    params["distance_scale"] = scales[GLOBALS]
    params = pd.concat([params, pd.DataFrame({"distance_scale": scales[NAMES]})])
    norm.to_csv(NORM_CSV, index=False)
    params.to_csv(PARAMS_CSV)
    print(f"database: {len(db)} shapes, query shapes taken out: {len(held)} ({QUERY_CSV.name})")
    print(f"normalized feature table: {NORM_CSV}")
    print(f"parameters: {PARAMS_CSV}")
    print(params.round(4).to_string())


def show(rows=8):
    norm = pd.read_csv(NORM_CSV)
    pd.set_option("display.width", 200)
    print(f"{NORM_CSV}  ({len(norm)} shapes, {norm.shape[1]} columns)\n")
    print(norm[["class", "file"] + GLOBALS + HIST["D2"][:3]].head(rows).round(3).to_string(index=False))
    print("(the table also holds all 100 histogram values: A3, D1, D2, D3 and D4 with 20 bins each)")
    print("\nglobal descriptors after normalization:")
    print(norm[GLOBALS].describe().T[["count", "mean", "std", "min", "max"]].round(3).to_string())


def precision_at_k(dist, classes, k=10):
    """Share of the k nearest shapes (without the shape itself) that have the class of the query."""
    np.fill_diagonal(dist, np.inf)
    nn = np.argpartition(dist, k, axis=1)[:, :k]
    return (classes[nn] == classes[:, None]).mean()


def compare(k=10):
    """Leave-one-out check of the options on the database: every shape is a query for all others."""
    table = pd.read_csv(FEATURES_CSV)
    db, _ = split_database(table)
    classes = db["class"].to_numpy()
    H = db[sum(HIST.values(), [])].to_numpy(float)
    results = []

    def single_vector(G):
        """All values in one vector, plain Euclidean distance."""
        return precision_at_k(cdist(np.hstack([G, H]), np.hstack([G, H])), classes, k)

    raw = db[GLOBALS].to_numpy(float)
    mm = (raw - np.nanmin(raw, 0)) / (np.nanmax(raw, 0) - np.nanmin(raw, 0))
    results.append(("one vector, min-max, no log", single_vector(mm)))
    z = (raw - np.nanmean(raw, 0)) / np.nanstd(raw, 0)
    results.append(("one vector, standardized, no log", single_vector(z)))
    params = fit_normalization(db)
    results.append(("one vector, standardized, log", single_vector(apply_normalization(db, params)[GLOBALS].to_numpy(float))))
    results.append(("only the histograms", precision_at_k(cdist(H, H), classes, k)))

    def weighted(norm):
        """Weighted descriptor distances (the method used), every shape against all others."""
        scales = fit_distance_scales(norm)
        D = np.empty((len(norm), len(norm)))
        for a in range(len(norm)):
            D[a] = combined_distance(descriptor_distances(norm.iloc[a], norm), scales)
        return precision_at_k(D, classes, k)

    norm = apply_normalization(db, params)
    results.append(("weighted descriptors, standardized, log", weighted(norm)))
    results.append(("weighted descriptors, min-max, log", weighted(apply_normalization(db, fit_normalization(db, "minmax")))))
    global LOG_FEATURES
    keep, LOG_FEATURES = LOG_FEATURES, []
    results.append(("weighted descriptors, standardized, no log", weighted(apply_normalization(db, fit_normalization(db)))))
    LOG_FEATURES = ["surface_area", "eccentricity"]
    results.append(("weighted descriptors, standardized, log of surface area and eccentricity only",
                    weighted(apply_normalization(db, fit_normalization(db)))))
    LOG_FEATURES = keep
    n_c = db["class"].value_counts()
    results.append(("random order (expected value)", float(((n_c * (n_c - 1)).sum()) / (len(db) * (len(db) - 1)))))
    r = pd.DataFrame(results, columns=["option", f"precision@{k}"])
    OUT.mkdir(parents=True, exist_ok=True)
    r.to_csv(OUT / "s4_options.csv", index=False)
    print(r.round(4).to_string(index=False))


def main():
    ap = argparse.ArgumentParser(description="Step 4: normalization, query and Euclidean distance")
    ap.add_argument("--build", action="store_true")
    ap.add_argument("--show", action="store_true")
    ap.add_argument("--compare", action="store_true")
    ap.add_argument("--query")
    ap.add_argument("-k", type=int, default=5)
    a = ap.parse_args()
    if a.build:
        build()
    if a.show:
        show()
    if a.compare:
        compare()
    if a.query:
        print(query(features_for(a.query), a.k).round(4).to_string(index=False))


if __name__ == "__main__":
    main()
