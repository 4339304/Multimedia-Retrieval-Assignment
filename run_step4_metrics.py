"""
Step 4 (second part): cosine distance and Earth Mover's Distance (EMD) for the histograms.

Run from the project folder:
    python run_step4_metrics.py --query Mug/D00685.obj -k 5                   both distances
    python run_step4_metrics.py --query Mug/D00685.obj -k 5 --metric emd     one distance
    python run_step4_metrics.py --compare                                     precision on the query shapes

Output of --compare: report/data/step4/metric_comparison_per_query.csv, metric_comparison_summary.csv,
                     metric_comparison_by_class.csv

The new distance replaces only the histogram distance of run_step4.py; the global descriptors, the
normalization and the weighting stay the same, and the weights rho_g are measured again for the new distance.
"""
import argparse

import numpy as np
import pandas as pd

import run_step4 as s4


# ------------------------------------------------------------------ histogram distances
def cosine_distance(a, b):
    """1 - cosine similarity, row by row."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    dot = np.sum(a * b, axis=-1)
    norm_a = np.linalg.norm(a, axis=-1)
    norm_b = np.linalg.norm(b, axis=-1)
    denom = norm_a * norm_b
    similarity = np.divide(dot, denom, out=np.zeros_like(dot, dtype=float), where=denom > 0)
    similarity = np.where((norm_a == 0) & (norm_b == 0), 1.0, similarity)    # two empty histograms are equal
    return np.clip(1.0 - similarity, 0.0, 2.0)


def earth_movers_distance(a, b):
    """EMD of two histograms with equal bins: the summed difference of their cumulative sums, divided by
    the number of bins minus one, so that it is relative to the range of the histogram."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if a.shape[-1] != b.shape[-1]:
        raise ValueError("the two histograms must have the same number of bins")
    bins = a.shape[-1]
    if bins <= 1:
        return np.zeros(np.broadcast_shapes(a.shape[:-1], b.shape[:-1]), dtype=float)
    cdf_difference = np.cumsum(a - b, axis=-1)
    return np.sum(np.abs(cdf_difference[..., :-1]), axis=-1) / (bins - 1)


METRICS = {"cosine": cosine_distance, "emd": earth_movers_distance}


# ------------------------------------------------------------------ query and comparison
def query_with_metric(features, k=5, metric_name="cosine"):
    """The k database shapes closest to a shape of the feature table, with the chosen histogram distance."""
    if metric_name not in METRICS:
        raise ValueError(f"unknown distance: {metric_name}")
    if k < 1:
        raise ValueError("k must be at least 1")
    norm, params, _ = s4.load_database()
    hist_metric = METRICS[metric_name]
    scales = s4.fit_distance_scales(norm, hist_metric=hist_metric)

    source = pd.read_csv(s4.FEATURES_CSV)
    query_class, query_file = str(features["class"]), str(features["file"])
    match = source[(source["class"].astype(str) == query_class) & (source["file"].astype(str) == query_file)]
    if match.empty:
        raise ValueError(f"{query_class}/{query_file} is not in {s4.FEATURES_CSV}")
    required = s4.GLOBALS + sum(s4.HIST.values(), [])
    missing = [c for c in required if c not in match.columns]
    if missing:
        raise ValueError(f"the feature table has no columns {missing}")
    q = s4.apply_normalization(match.iloc[[0]][["class", "file"] + required].copy(), params).iloc[0]

    distances = s4.combined_distance(s4.descriptor_distances(q, norm, hist_metric=hist_metric), scales)
    same_shape = ((norm["class"].astype(str) == query_class) & (norm["file"].astype(str) == query_file)).to_numpy()
    distances[same_shape] = np.inf                     # the query itself is not a result
    distances[~np.isfinite(distances)] = np.inf
    order = np.argsort(distances, kind="stable")
    order = order[np.isfinite(distances[order])][:min(k, len(norm))]
    return pd.DataFrame({"rank": np.arange(1, len(order) + 1), "class": norm["class"].iloc[order].to_numpy(),
                         "file": norm["file"].iloc[order].to_numpy(), "distance": distances[order]})



def compare_metrics(k=10):
    """
    Leave-one-out evaluation on the full feature table.
    Compares Euclidean, cosine distance and EMD using Precision@k.
    The query shape itself is excluded from its retrieved results.
    """
    table = pd.read_csv(s4.FEATURES_CSV)

    required = {"class", "file"}
    if not required.issubset(table.columns):
        raise ValueError("Feature table must contain class and file columns")

    if len(table) < 2:
        raise ValueError("At least two shapes are required")

    # Use every shape; do not remove the 10 held-out query shapes.
    table = table.reset_index(drop=True)

    # Fit normalization using the complete evaluation collection.
    params = s4.fit_normalization(table)
    norm = s4.apply_normalization(table, params)
    norm = norm.reset_index(drop=True)

    classes = norm["class"].astype(str).to_numpy()
    files = norm["file"].astype(str).to_numpy()
    k_eff = min(k, len(norm) - 1)

    metrics = {
        "euclidean": s4.euclidean,
        "cosine": cosine_distance,
        "emd": earth_movers_distance,
    }

    output_rows = []

    print(
        f"Evaluating {len(norm)} shapes with leave-one-out "
        f"retrieval, Precision@{k_eff}...",
        flush=True,
    )

    for metric_name, hist_metric in metrics.items():
        print(f"Evaluating {metric_name}...", flush=True)

        # Fit descriptor-distance scales consistently for this metric.
        scales = s4.fit_distance_scales(
            norm, hist_metric=hist_metric, seed=0
        )

        for i in range(len(norm)):
            q = norm.iloc[i]

            descriptor_d = s4.descriptor_distances(
                q, norm, hist_metric=hist_metric
            )
            distances = s4.combined_distance(descriptor_d, scales)

            # Exclude this exact database row, not every shape of its class.
            distances[i] = np.inf
            distances[~np.isfinite(distances)] = np.inf

            order = np.argsort(distances, kind="stable")
            order = order[np.isfinite(distances[order])][:k_eff]

            retrieved_classes = classes[order]
            relevant_count = int(
                np.sum(retrieved_classes == classes[i])
            )
            returned_count = len(order)

            output_rows.append({
                "query_class": classes[i],
                "query_file": files[i],
                "metric": metric_name,
                "k": k_eff,
                "relevant_count": relevant_count,
                "returned_count": returned_count,
                "precision_at_k": (
                    relevant_count / returned_count
                    if returned_count else 0.0
                ),
            })

    details = pd.DataFrame(output_rows)
    s4.OUT.mkdir(parents=True, exist_ok=True)

    details_path = s4.OUT / "metric_comparison_per_query.csv"
    details.to_csv(details_path, index=False)

    summary = (
        details.groupby(["metric", "k"])["precision_at_k"]
        .agg(["mean", "std", "count"])
        .reset_index()
        .rename(columns={
            "mean": "mean_precision_at_k",
            "std": "std_precision_at_k",
            "count": "query_count",
        })
    )

    summary_path = s4.OUT / "metric_comparison_summary.csv"
    summary.to_csv(summary_path, index=False)

    per_class = (
        details.groupby(["query_class", "metric", "k"])["precision_at_k"]
        .mean()
        .reset_index(name="mean_precision_at_k")
    )

    class_path = s4.OUT / "metric_comparison_by_class.csv"
    per_class.to_csv(class_path, index=False)

    print("\nFull-database Precision@k comparison:")
    print(summary.round(4).to_string(index=False))
    print(f"\nPer-query results: {details_path}")
    print(f"Overall summary:   {summary_path}")
    print(f"Class-wise results:{class_path}")

def main():
    ap = argparse.ArgumentParser(description="Step 4: cosine distance and Earth Mover's Distance")
    ap.add_argument("--query", help="shape of the feature table, e.g. Mug/D00685.obj")
    ap.add_argument("-k", type=int, default=5)
    ap.add_argument("--metric", choices=["cosine", "emd", "both"], default="both")
    ap.add_argument("--compare", action="store_true")
    a = ap.parse_args()
    if a.compare:
        compare_metrics()
        return
    if not a.query:
        ap.print_help()
        return
    features = s4.features_for(a.query)
    for metric_name in (list(METRICS) if a.metric == "both" else [a.metric]):
        print(f"\nthe {a.k} closest shapes with {metric_name.upper()}:", flush=True)
        print(query_with_metric(features, k=a.k, metric_name=metric_name).to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
