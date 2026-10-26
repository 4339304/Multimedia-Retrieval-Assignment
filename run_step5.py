"""
Step 5: approximate nearest-neighbour search with an HNSW index (FAISS, faiss-cpu).

Run from the project folder:
    python run_step5.py --build                                   build the index of the database vectors
    python run_step5.py --query Mug/D00685.obj -k 10              the K nearest shapes
    python run_step5.py --query Mug/D00685.obj --compare -k 10    compare with the exact search of Step 4
    python run_step5.py --query Mug/D00685.obj --radius 2.7 --max-candidates 2473
                                                                  all shapes closer than the radius

Output: report/data/step4/step5/ann_index.bin, ann_vectors.npy, ann_metadata.csv (index and its vectors)
        report/data/step4/step5/ann_query_results.csv          result of the last query
        report/data/step4/step5/ann_exact_comparison.csv       time and recall against Step 4

The vectors are those of step5_common.py, so the Euclidean (L2) distance of the index is the distance of
Step 4. A radius query asks the index for max_candidates neighbours and keeps the ones within the radius.
"""
import argparse
import time

import numpy as np
import pandas as pd

try:
    import faiss
except ImportError:
    raise SystemExit("FAISS is not installed, run: pip install -r requirements.txt")

import run_step4 as s4
from step5_common import (INDEX_PATH, META_PATH, STEP5_DIR, VECTORS_PATH, ensure_step5_dir, load_step4_data,
                          query_vector, vectorize_table)


# ------------------------------------------------------------------ index
def build_index(m=16, ef_construction=200, ef_search=100):
    """Build the HNSW index of all database vectors and save it with the vectors and the shape names."""
    ensure_step5_dir()
    norm, params, scales = load_step4_data()
    X = vectorize_table(norm, scales)
    index = faiss.IndexHNSWFlat(X.shape[1], m)          # Euclidean (L2) distance, m links per node
    index.hnsw.efConstruction = ef_construction
    index.add(X)                                        # the shapes get the numbers 0 .. n-1 in table order
    index.hnsw.efSearch = max(ef_search, 50)
    faiss.write_index(index, str(INDEX_PATH))
    norm[["class", "file"]].to_csv(META_PATH, index=False)
    np.save(VECTORS_PATH, X)
    print(f"HNSW index of {len(norm):,} shapes, {X.shape[1]} values per shape: {INDEX_PATH}")


def load_index(ef=100):
    """Load the saved index; it is built first if it is missing or older than the features of Step 4."""
    files = [INDEX_PATH, META_PATH, VECTORS_PATH]
    if not all(p.exists() for p in files):
        print("no index found, building it first")
        build_index()
    elif min(p.stat().st_mtime for p in files) < s4.NORM_CSV.stat().st_mtime:
        print("the features of Step 4 are newer than the index, building it again")
        build_index()
    norm, params, scales = load_step4_data()
    X = np.load(VECTORS_PATH)
    index = faiss.read_index(str(INDEX_PATH))
    index.hnsw.efSearch = max(ef, 50)
    return norm, params, scales, X, index, pd.read_csv(META_PATH)


# ------------------------------------------------------------------ queries
def ann_query(query_name, k=10, radius=None, max_candidates=200, ef=100):
    """The k nearest shapes, or all shapes within the radius, of a query. The query itself is left out."""
    if k < 1:
        raise ValueError("k must be at least 1")
    if radius is not None and radius < 0:
        raise ValueError("the radius must not be negative")
    norm, params, scales, X, index, meta = load_index(ef)
    features = s4.features_for(query_name)
    qvec = query_vector(features, params, scales)
    cls, file = str(features.get("class", "")), str(features.get("file", ""))
    hits = meta.index[(meta["class"].astype(str) == cls) & (meta["file"].astype(str) == file)].tolist()
    self_idx = hits[0] if hits else None
    n = len(meta)
    want = min(n, max_candidates if radius is not None else min(n, k + (1 if self_idx is not None else 0) + 10))
    t0 = time.perf_counter()
    dist_sq, labels = index.search(qvec[None, :].astype(np.float32), want)    # squared L2 distances
    elapsed = (time.perf_counter() - t0) * 1000
    rows = []
    for idx, d2 in zip(labels[0], dist_sq[0]):
        idx, d = int(idx), float(np.sqrt(max(float(d2), 0.0)))
        if idx < 0:                                     # fewer neighbours found than asked for
            continue
        if self_idx is not None and idx == self_idx:
            continue
        if radius is not None and d > radius:
            continue
        rows.append({"class": meta.iloc[idx]["class"], "file": meta.iloc[idx]["file"], "distance": d})
        if radius is None and len(rows) >= k:
            break
    result = pd.DataFrame(rows)
    if not result.empty:
        result.insert(0, "rank", np.arange(1, len(result) + 1))
    out = STEP5_DIR / "ann_query_results.csv"
    result.to_csv(out, index=False)
    mode = f"radius <= {radius}" if radius is not None else f"K = {k}"
    print(f"\nANN query: {query_name}, {mode}")
    print(f"search time: {elapsed:.3f} ms")
    print(result.round(5).to_string(index=False) if not result.empty else "no shapes within the radius")
    print(f"saved: {out}")
    if radius is not None and want < n:
        print(f"radius search over the {want} nearest candidates only; use --max-candidates for more")
    return result, elapsed, features


def compare(query_name, k=10, ef=100):
    """Time and recall of the index against the exact search of Step 4 for one query."""
    ann, ann_ms, features = ann_query(query_name, k=k, ef=ef)
    t0 = time.perf_counter()
    exact = s4.query(features, k=k)
    exact_ms = (time.perf_counter() - t0) * 1000
    a = set(zip(ann.head(k)["class"].astype(str), ann.head(k)["file"].astype(str)))
    e = set(zip(exact["class"].astype(str), exact["file"].astype(str)))
    recall = len(a & e) / max(1, len(e))
    report = pd.DataFrame([
        {"method": "HNSW ANN", "query_ms": ann_ms, f"returned_{k}": len(ann.head(k)), f"recall_at_{k}": recall},
        {"method": "Step 4 exact", "query_ms": exact_ms, f"returned_{k}": len(exact), f"recall_at_{k}": np.nan},
    ])
    out = STEP5_DIR / "ann_exact_comparison.csv"
    report.to_csv(out, index=False)
    print("\nANN against the exact search")
    print(report.round(4).to_string(index=False))
    print(f"saved: {out}")


def main():
    ap = argparse.ArgumentParser(description="Step 5: approximate nearest-neighbour search (HNSW)")
    ap.add_argument("--build", action="store_true")
    ap.add_argument("--query")
    ap.add_argument("-k", type=int, default=10)
    ap.add_argument("--radius", type=float, help="radius in the weighted feature distance")
    ap.add_argument("--max-candidates", type=int, default=200)
    ap.add_argument("--ef", type=int, default=100)
    ap.add_argument("--compare", action="store_true")
    a = ap.parse_args()
    if a.build:
        build_index()
    if a.query:
        if a.compare:
            compare(a.query, a.k, a.ef)
        else:
            ann_query(a.query, a.k, a.radius, a.max_candidates, a.ef)
    if not a.build and not a.query:
        ap.print_help()


if __name__ == "__main__":
    main()
