"""
Step 4 (second part): cosine distance and Earth Mover's Distance (EMD) for the histograms.

Run from the project folder:
    python run_step4_metrics.py --query Mug/D00685.obj -k 5                   both distances
    python run_step4_metrics.py --query Mug/D00685.obj -k 5 --metric emd     one distance
    python run_step4_metrics.py --compare                                     every database shape as a query

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
    """The k database shapes closest to a query (a row of features, not normalized; a shape of the feature table or
    a mesh file from outside the database), with the chosen histogram distance."""
    if metric_name not in METRICS:
        raise ValueError(f"unknown distance: {metric_name}")
    if k < 1:
        raise ValueError("k must be at least 1")
    norm, params, _ = s4.load_database()
    hist_metric = METRICS[metric_name]
    scales = s4.fit_distance_scales(norm, hist_metric=hist_metric)
    required = ["class", "file"] + s4.GLOBALS + sum(s4.HIST.values(), [])
    q = s4.apply_normalization(features[required].to_frame().T, params).iloc[0]
    distances = s4.combined_distance(s4.descriptor_distances(q, norm, hist_metric=hist_metric), scales)
    same_shape = ((norm["class"].astype(str) == str(q["class"])) & (norm["file"].astype(str) == str(q["file"]))).to_numpy()
    distances[same_shape] = np.inf                     # a database shape is not a result of its own query
    order = np.argsort(distances, kind="stable")[:min(k, len(norm))]
    return pd.DataFrame({"rank": np.arange(1, len(order) + 1), "class": norm["class"].iloc[order].to_numpy(),
                         "file": norm["file"].iloc[order].to_numpy(), "distance": distances[order]})


def compare_metrics(k=10):
    """Every database shape once as a query (the query itself left out): precision of the k closest shapes with the
    Euclidean distance, the cosine distance and the EMD for the histograms. Same database and normalization as
    run_step4.py --compare, so the Euclidean value equals the one of Step 4."""
    norm, params, _ = s4.load_database()
    classes = norm["class"].astype(str).to_numpy()
    files = norm["file"].astype(str).to_numpy()
    k = min(k, len(norm) - 1)
    metrics = {"euclidean": s4.euclidean, **METRICS}
    output_rows = []
    print(f"{len(norm)} database shapes as queries, precision of the {k} closest shapes ...", flush=True)
    for metric_name, hist_metric in metrics.items():
        print(f"  {metric_name}", flush=True)
        scales = s4.fit_distance_scales(norm, hist_metric=hist_metric)
        for i in range(len(norm)):
            distances = s4.combined_distance(s4.descriptor_distances(norm.iloc[i], norm, hist_metric=hist_metric),
                                             scales)
            distances[i] = np.inf                       # the query itself is not a result
            order = np.argsort(distances, kind="stable")[:k]
            relevant = int((classes[order] == classes[i]).sum())
            output_rows.append({"query_class": classes[i], "query_file": files[i], "metric": metric_name, "k": k,
                                "relevant_count": relevant, "returned_count": k, "precision_at_k": relevant / k})

    details = pd.DataFrame(output_rows)
    summary = (details.groupby(["metric", "k"])["precision_at_k"].agg(["mean", "std", "count"]).reset_index()
               .rename(columns={"mean": "mean_precision_at_k", "std": "std_precision_at_k", "count": "query_count"}))
    per_class = (details.groupby(["query_class", "metric", "k"])["precision_at_k"].mean()
                 .reset_index(name="mean_precision_at_k"))
    s4.OUT.mkdir(parents=True, exist_ok=True)
    details.to_csv(s4.OUT / "metric_comparison_per_query.csv", index=False)
    summary.to_csv(s4.OUT / "metric_comparison_summary.csv", index=False)
    per_class.to_csv(s4.OUT / "metric_comparison_by_class.csv", index=False)
    print("\nprecision of the closest shapes, every database shape as a query")
    print(summary.round(4).to_string(index=False))
    print(f"saved: {s4.OUT}/metric_comparison_*.csv")


def main():
    ap = argparse.ArgumentParser(description="Step 4: cosine distance and Earth Mover's Distance")
    ap.add_argument("--query", help="mesh file, or class/file of a shape of the feature table")
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
