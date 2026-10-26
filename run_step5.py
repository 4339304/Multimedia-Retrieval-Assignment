"""Step 5 ANN search. Run from the folder containing run_step4.py."""
import argparse, time
import numpy as np
import pandas as pd
import hnswlib
import run_step4 as s4
from step5_common import STEP5_DIR, INDEX_PATH, META_PATH, VECTORS_PATH, load_step4_data, vectorize_table, query_vector, ensure_step5_dir

def build_index(m=16, ef_construction=200, ef_search=100):
    ensure_step5_dir()
    norm, params, scales = load_step4_data()
    X = vectorize_table(norm, scales)
    index = hnswlib.Index(space="l2", dim=X.shape[1])
    index.init_index(max_elements=len(X), ef_construction=ef_construction, M=m, random_seed=42)
    index.add_items(X, np.arange(len(X), dtype=np.int32), num_threads=-1)
    index.set_ef(max(ef_search, 50))
    index.save_index(str(INDEX_PATH))
    norm[["class", "file"]].to_csv(META_PATH, index=False)
    np.save(VECTORS_PATH, X)
    print(f"Built HNSW index for {len(norm):,} shapes; vector dimensions={X.shape[1]}")
    print(f"Saved index: {INDEX_PATH}")

def load_index(ef=100):
    norm, params, scales = load_step4_data()
    if not all(p.exists() for p in [INDEX_PATH, META_PATH, VECTORS_PATH]):
        raise FileNotFoundError("Index missing. Run: python run_step5_ann.py --build")
    X = np.load(VECTORS_PATH)
    index = hnswlib.Index(space="l2", dim=X.shape[1])
    index.load_index(str(INDEX_PATH), max_elements=len(norm))
    index.set_ef(max(ef, 50))
    return norm, params, scales, X, index, pd.read_csv(META_PATH)

def ann_query(query_name, k=10, radius=None, max_candidates=200, ef=100):
    if k < 1: raise ValueError("k must be >= 1")
    if radius is not None and radius < 0: raise ValueError("radius must be >= 0")
    norm, params, scales, X, index, meta = load_index(ef)
    features = s4.features_for(query_name)
    qvec = query_vector(features, params, scales)
    cls, file = str(features.get("class", "")), str(features.get("file", ""))
    hits = meta.index[(meta["class"].astype(str) == cls) & (meta["file"].astype(str) == file)].tolist()
    self_idx = hits[0] if hits else None
    n = len(meta)
    want = min(n, max_candidates if radius is not None else min(n, k + (1 if self_idx is not None else 0) + 10))
    t0 = time.perf_counter()
    labels, dist_sq = index.knn_query(qvec, k=want, num_threads=-1)
    elapsed = (time.perf_counter() - t0) * 1000
    rows = []
    for idx, d2 in zip(labels[0], dist_sq[0]):
        idx, d = int(idx), float(np.sqrt(max(float(d2), 0.0)))
        if self_idx is not None and idx == self_idx: continue
        if radius is not None and d > radius: continue
        rows.append({"class": meta.iloc[idx]["class"], "file": meta.iloc[idx]["file"], "distance": d})
        if radius is None and len(rows) >= k: break
    result = pd.DataFrame(rows)
    if not result.empty: result.insert(0, "rank", np.arange(1, len(result)+1))
    out = STEP5_DIR / "ann_query_results.csv"
    result.to_csv(out, index=False)
    print(f"\nANN query: {query_name}; mode={'radius <= '+str(radius) if radius is not None else 'K='+str(k)}")
    print(f"Search time: {elapsed:.3f} ms")
    print(result.round(5).to_string(index=False) if not result.empty else "No results found within the radius/candidate limit.")
    print(f"Saved: {out}")
    if radius is not None and want < n: print(f"Approximate radius search: inspected {want} candidates; increase --max-candidates to inspect more.")
    return result, elapsed, features

def compare(query_name, k=10, ef=100):
    ann, ann_ms, features = ann_query(query_name, k=k, ef=ef)
    t0 = time.perf_counter(); exact = s4.query(features, k=k); exact_ms = (time.perf_counter()-t0)*1000
    a = set(zip(ann.head(k)["class"].astype(str), ann.head(k)["file"].astype(str)))
    e = set(zip(exact["class"].astype(str), exact["file"].astype(str)))
    recall = len(a & e) / max(1, len(e))
    report = pd.DataFrame([
        {"method":"HNSW ANN", "query_ms":ann_ms, f"returned_{k}":len(ann.head(k)), f"recall_at_{k}":recall},
        {"method":"Step 4 exact", "query_ms":exact_ms, f"returned_{k}":len(exact), f"recall_at_{k}":np.nan},
    ])
    out = STEP5_DIR / "ann_exact_comparison.csv"; report.to_csv(out, index=False)
    print("\nExact-vs-ANN comparison")
    print(report.round(4).to_string(index=False))
    print(f"Saved: {out}")

def main():
    p=argparse.ArgumentParser(description="Step 5: HNSW approximate nearest-neighbour search")
    p.add_argument("--build", action="store_true")
    p.add_argument("--query")
    p.add_argument("-k", type=int, default=10)
    p.add_argument("--radius", type=float, help="approximate range-search radius in weighted feature-vector distance")
    p.add_argument("--max-candidates", type=int, default=200)
    p.add_argument("--ef", type=int, default=100)
    p.add_argument("--compare", action="store_true")
    a=p.parse_args()
    if a.build: build_index()
    if a.query:
        if a.compare: compare(a.query, a.k, a.ef)
        else: ann_query(a.query, a.k, a.radius, a.max_candidates, a.ef)
    if not a.build and not a.query: p.print_help()
if __name__ == "__main__": main()
