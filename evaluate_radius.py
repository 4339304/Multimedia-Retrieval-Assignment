
"""Evaluate HNSW radius retrieval against exact L2 distance filtering."""

import time
import numpy as np
import pandas as pd

import run_step4 as s4
from step5_common import STEP5_DIR, load_step4_data, query_vector
from run_step5 import load_index

QUERIES = [
    ("Car", "D00168.obj"),
    ("Humanoid", "m118.obj"),
    ("Insect", "D00081.obj"),
    ("Jet", "m1147.obj"),
    ("Ship", "D00027.obj"),
]

RADII = [1.0, 1.5, 2.0]
CANDIDATE_LIMITS = [200, 2473]


def main():
    STEP5_DIR.mkdir(parents=True, exist_ok=True)
    norm, params, scales, X, index, meta = load_index(ef=100)
    X = np.asarray(X, dtype=np.float32)
    n = len(X)
    index.hnsw.efSearch = 100

    summary_rows = []
    detail_rows = []

    for query_class, query_file in QUERIES:
        query_name = f"{query_class}/{query_file}"
        features = s4.features_for(query_name)
        qvec = query_vector(features, params, scales).astype(np.float32)

        hits = meta.index[
            (meta["class"].astype(str) == query_class)
            & (meta["file"].astype(str) == query_file)
        ].tolist()
        self_idx = hits[0] if hits else None

        # Exact distance to every indexed vector.
        exact_d2 = np.sum((X - qvec) ** 2, axis=1)
        exact_d = np.sqrt(np.maximum(exact_d2, 0))
        if self_idx is not None:
            exact_d[self_idx] = np.inf

        for radius in RADII:
            exact_ids = set(np.flatnonzero(exact_d <= radius).tolist())
            for limit in CANDIDATE_LIMITS:
                limit = min(limit, n)

                start = time.perf_counter()
                d2, labels = index.search(qvec[None, :], limit)
                elapsed_ms = (time.perf_counter() - start) * 1000

                ann_ids = set()
                for idx, dist2 in zip(labels[0], d2[0]):
                    idx = int(idx)
                    if idx < 0 or idx == self_idx:
                        continue
                    dist = float(np.sqrt(max(float(dist2), 0.0)))
                    if dist <= radius:
                        ann_ids.add(idx)

                common = ann_ids & exact_ids
                recall = len(common) / len(exact_ids) if exact_ids else np.nan
                precision = len(common) / len(ann_ids) if ann_ids else np.nan
                
                missed_ids = exact_ids - ann_ids
                for idx in sorted(missed_ids, key=lambda i: exact_d[i]):
                    detail_rows.append({
                        "query": query_name,
                        "radius": radius,
                        "max_candidates": limit,
                        "result_class": meta.iloc[idx]["class"],
                        "result_file": meta.iloc[idx]["file"],
                        "distance": float(exact_d[idx]),
                        "also_in_exact_radius": True,
                        "retrieved_by_ann": False,
                    })


                summary_rows.append({
                    "query": query_name,
                    "query_class": query_class,
                    "radius": radius,
                    "max_candidates": limit,
                    "exact_count": len(exact_ids),
                    "ann_count": len(ann_ids),
                    "matched_count": len(common),
                    "radius_recall": recall,
                    "radius_precision": precision,
                    "ann_search_ms": elapsed_ms, 
                })

                
                for idx in sorted(ann_ids, key=lambda i: exact_d[i]):
                    detail_rows.append({
                        "query": query_name,
                        "radius": radius,
                        "max_candidates": limit,
                        "result_class": meta.iloc[idx]["class"],
                        "result_file": meta.iloc[idx]["file"],
                        "distance": float(exact_d[idx]),
                        "also_in_exact_radius": True,
                        "retrieved_by_ann": True,
                    })

    summary = pd.DataFrame(summary_rows)
    details = pd.DataFrame(detail_rows)

    summary_path = STEP5_DIR / "ann_radius_evaluation.csv"
    details_path = STEP5_DIR / "ann_radius_results.csv"
    summary.to_csv(summary_path, index=False)
    details.to_csv(details_path, index=False)

    print("\nRadius retrieval evaluation")
    print(summary.round(4).to_string(index=False))
    print(f"\nSummary saved: {summary_path}")
    print(f"Retrieved shapes saved: {details_path}")


if __name__ == "__main__":
    main()
