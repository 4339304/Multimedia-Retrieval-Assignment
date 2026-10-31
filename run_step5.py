"""
Step 5: approximate nearest-neighbor search with an HNSW index (FAISS, faiss-cpu).

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
Step 4. A radius query asks the index for max_candidates neighbors and keeps the ones within the radius.
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
        if idx < 0:                                     # fewer neighbors found than asked for
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



def compare(query_name, k=10, ef=100, repeats=20):
    """Compare ANN and exact vector search after loading the database and index."""
    if k < 1:
        raise ValueError("k must be at least 1")
    if repeats < 1:
        raise ValueError("repeats must be at least 1")

    norm, params, scales, X, index, meta = load_index(ef)
    features = s4.features_for(query_name)
    qvec = query_vector(features, params, scales).astype(np.float32)

    cls = str(features.get("class", ""))
    file = str(features.get("file", ""))
    hits = meta.index[
        (meta["class"].astype(str) == cls)
        & (meta["file"].astype(str) == file)
    ].tolist()
    self_idx = hits[0] if hits else None

    n = len(X)
    if k >= n - (1 if self_idx is not None else 0):
        raise ValueError("k must be smaller than the number of eligible database shapes")

    # Warm up both search methods before timing.
    index.search(qvec[None, :], min(n, k + 10))
    distances = np.sum((X - qvec) ** 2, axis=1)
    if self_idx is not None:
        distances[self_idx] = np.inf
    np.argsort(distances)[:k]

    ann_times = []
    exact_times = []

    # Repeat search-only timings; setup and query feature extraction are excluded.
    for _ in range(repeats):
        t0 = time.perf_counter()
        dist_sq, labels = index.search(
            qvec[None, :], min(n, k + (1 if self_idx is not None else 0) + 10)
        )
        ann_times.append((time.perf_counter() - t0) * 1000)

        t0 = time.perf_counter()
        exact_dist_sq = np.sum((X - qvec) ** 2, axis=1)
        if self_idx is not None:
            exact_dist_sq[self_idx] = np.inf
        exact_ids = np.argsort(exact_dist_sq)[:k]
        exact_times.append((time.perf_counter() - t0) * 1000)

    # Retrieve the final ANN top-k after excluding the query itself.
    dist_sq, labels = index.search(
        qvec[None, :], min(n, k + (1 if self_idx is not None else 0) + 10)
    )
    ann_ids = []
    for idx in labels[0]:
        idx = int(idx)
        if idx < 0 or (self_idx is not None and idx == self_idx):
            continue
        ann_ids.append(idx)
        if len(ann_ids) == k:
            break

    # Recompute exact top-k, with self-retrieval excluded.
    exact_dist_sq = np.sum((X - qvec) ** 2, axis=1)
    if self_idx is not None:
        exact_dist_sq[self_idx] = np.inf
    exact_ids = np.argsort(exact_dist_sq)[:k].tolist()

    recall = len(set(ann_ids) & set(exact_ids)) / k

    report = pd.DataFrame([
        {
            "method": "HNSW ANN",
            "query": query_name,
            "k": k,
            "repeats": repeats,
            "mean_search_ms": np.mean(ann_times),
            "median_search_ms": np.median(ann_times),
            f"returned_{k}": len(ann_ids),
            f"recall_at_{k}": recall,
        },
        {
            "method": "Exact vector search",
            "query": query_name,
            "k": k,
            "repeats": repeats,
            "mean_search_ms": np.mean(exact_times),
            "median_search_ms": np.median(exact_times),
            f"returned_{k}": len(exact_ids),
            f"recall_at_{k}": 1.0,
        },
    ])

    out = STEP5_DIR / "ann_exact_comparison.csv"
    report.to_csv(out, index=False)
    print("\nANN against exact vector search (search-only timing)")
    print(report.round(5).to_string(index=False))
    print(f"saved: {out}")


def evaluate_full_database(k=10, ef=100, repeats=3, batch_size=128):
    """Evaluate HNSW against exact L2 retrieval using every database shape as a query."""
    if k < 1 or repeats < 1 or batch_size < 1:
        raise ValueError("k, repeats and batch_size must be positive")

    norm, params, scales, X, index, meta = load_index(ef)
    n = len(X)

    if k >= n:
        raise ValueError("k must be smaller than the database size")

    X = np.asarray(X, dtype=np.float32)
    index.hnsw.efSearch = max(ef, 50)

    # ANN queries are the database vectors themselves.
    # Request extra neighbors so the query shape can be removed.
    ann_k = min(n, k + 10)

    # Warm up the index and exact-distance calculation.
    index.search(X[:min(5, n)], ann_k)
    _ = X[:min(5, n)] @ X.T

    ann_times = []
    exact_times = []
    ann_ids = None
    exact_ids = None

    # Measure total search time over the complete query set.
    for _ in range(repeats):
        start = time.perf_counter()
        all_ann_ids = []

        for start_i in range(0, n, batch_size):
            batch = X[start_i:start_i + batch_size]
            _, labels = index.search(batch, ann_k)
            all_ann_ids.append(labels)

        ann_ids = np.vstack(all_ann_ids)
        ann_times.append((time.perf_counter() - start) * 1000)

        start = time.perf_counter()

        # Exact pairwise squared L2 distances, processed in blocks.
        # Matrix multiplication avoids creating a huge 3-D array.
        all_exact_ids = []
        norms = np.sum(X * X, axis=1)

        for start_i in range(0, n, batch_size):
            stop_i = min(start_i + batch_size, n)
            Q = X[start_i:stop_i]
            d2 = norms[None, :] + norms[start_i:stop_i, None] - 2 * (Q @ X.T)
            np.maximum(d2, 0, out=d2)

            # Exclude each query's own database row.
            local_rows = np.arange(stop_i - start_i)
            global_rows = np.arange(start_i, stop_i)
            d2[local_rows, global_rows] = np.inf

            # Select the nearest k, then sort them by distance.
            candidates = np.argpartition(d2, kth=k - 1, axis=1)[:, :k]
            candidate_distances = np.take_along_axis(d2, candidates, axis=1)
            order = np.argsort(candidate_distances, axis=1)
            all_exact_ids.append(
                np.take_along_axis(candidates, order, axis=1)
            )

        exact_ids = np.vstack(all_exact_ids)
        exact_times.append((time.perf_counter() - start) * 1000)

    # Filter each query's own row from ANN results and calculate recall@k.
    recalls = np.zeros(n, dtype=float)
    returned = np.zeros(n, dtype=int)

    for i in range(n):
        neighbors = []
        for idx in ann_ids[i]:
            idx = int(idx)
            if idx < 0 or idx == i:
                continue
            neighbors.append(idx)
            if len(neighbors) == k:
                break

        returned[i] = len(neighbors)
        recalls[i] = len(set(neighbors) & set(exact_ids[i])) / k

    query_results = meta[["class", "file"]].copy()
    query_results.insert(0, "query_id", np.arange(n))
    query_results["returned_ann"] = returned
    query_results[f"recall_at_{k}"] = recalls

    per_query_path = STEP5_DIR / "ann_full_database_per_query.csv"
    query_results.to_csv(per_query_path, index=False)

    summary = pd.DataFrame([
        {
            "method": "HNSW ANN",
            "database_shapes": n,
            "k": k,
            "repeats": repeats,
            "mean_total_ms": np.mean(ann_times),
            "median_total_ms": np.median(ann_times),
            "mean_ms_per_query": np.mean(ann_times) / n,
            "mean_recall_at_k": np.mean(recalls),
            "queries_with_k_results": int(np.sum(returned == k)),
        },
        {
            "method": "Exact vector search",
            "database_shapes": n,
            "k": k,
            "repeats": repeats,
            "mean_total_ms": np.mean(exact_times),
            "median_total_ms": np.median(exact_times),
            "mean_ms_per_query": np.mean(exact_times) / n,
            "mean_recall_at_k": 1.0,
            "queries_with_k_results": n,
        },
    ])

    summary_path = STEP5_DIR / "ann_full_database_summary.csv"
    summary.to_csv(summary_path, index=False)

    class_summary = (
        query_results.groupby("class", dropna=False)
        .agg(
            queries=("query_id", "count"),
            mean_recall_at_k=(f"recall_at_{k}", "mean"),
            queries_with_k_results=("returned_ann", lambda s: int((s == k).sum())),
        )
        .reset_index()
    )

    class_path = STEP5_DIR / "ann_full_database_class_summary.csv"
    class_summary.to_csv(class_path, index=False)

    print(f"\nFull-database evaluation: {n:,} shapes, k={k}, repeats={repeats}")
    print("\nOverall summary (batch search timings)")
    print(summary.round(5).to_string(index=False))
    print("\nLowest-recall classes")
    print(class_summary.sort_values("mean_recall_at_k").head(10).round(4).to_string(index=False))
    print(f"\nPer-query results: {per_query_path}")
    print(f"Overall summary:   {summary_path}")
    print(f"Class summary:     {class_path}")

def main():
    ap = argparse.ArgumentParser(description="Step 5: approximate nearest-neighbor search (HNSW)")
    ap.add_argument("--build", action="store_true")
    ap.add_argument("--query")
    ap.add_argument("-k", type=int, default=10)
    ap.add_argument("--radius", type=float, help="radius in the weighted feature distance")
    ap.add_argument("--max-candidates", type=int, default=200)
    ap.add_argument("--ef", type=int, default=100)
    ap.add_argument("--compare", action="store_true")
    ap.add_argument("--evaluate-all",
                     action="store_true",
                help="evaluate ANN against exact search using every database shape")

    a = ap.parse_args()
    
    if a.evaluate_all:
        evaluate_full_database(k=a.k, ef=a.ef)
        return

    if a.build:
        build_index()

    if a.compare and not a.query:
        ap.error("--compare requires --query, e.g. --query Mug/D00685.obj")

    if a.query:
        if a.compare:
            compare(a.query, a.k, a.ef)
        else:
            ann_query(a.query, a.k, a.radius, a.max_candidates, a.ef)

    if not a.build and not a.query and not a.compare:
        ap.print_help()




if __name__ == "__main__":
    main()
