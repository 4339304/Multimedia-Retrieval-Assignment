"""
Step 4 extension: cosine distance and Earth Mover's Distance (EMD).

Uses the existing run_step4.py feature extraction, normalization and database.
Does not modify run_step4.py.

Examples (run from the assignment project folder):
    python run_step4_metrics.py --query Mug/D00685.obj -k 5
    python run_step4_metrics.py --query Mug/D00685.obj -k 5 --metric cosine
    python run_step4_metrics.py --query Mug/D00685.obj -k 5 --metric emd
    python run_step4_metrics.py --compare
"""

import argparse
import numpy as np
import pandas as pd

import run_step4 as s4
from pathlib import Path

s4.PROJECT_ROOT = Path(__file__).resolve().parent
s4.OUT = s4.PROJECT_ROOT / "report" / "data" / "step4"
s4.NORM_CSV = s4.OUT / "features_normalized.csv"
s4.PARAMS_CSV = s4.OUT / "normalization_params.csv"
s4.QUERY_CSV = s4.OUT / "query_shapes.csv"


def cosine_distance(a, b):
    """Row-wise cosine distance: 1 - cosine similarity."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    dot = np.sum(a * b, axis=-1)
    norm_a = np.linalg.norm(a, axis=-1)
    norm_b = np.linalg.norm(b, axis=-1)
    denom = norm_a * norm_b

    similarity = np.divide(
        dot, denom, out=np.zeros_like(dot, dtype=float), where=denom > 0
    )
    both_zero = (norm_a == 0) & (norm_b == 0)
    similarity = np.where(both_zero, 1.0, similarity)
    return np.clip(1.0 - similarity, 0.0, 2.0)


def earth_movers_distance(a, b):
    """
    Normalized 1-D Earth Mover's Distance for equally spaced histogram bins.
    For normalized histograms, this sums absolute CDF differences and divides
    by (number of bins - 1), giving a distance relative to the histogram range.
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)

    if a.shape[-1] != b.shape[-1]:
        raise ValueError("EMD requires histograms with the same number of bins.")

    bins = a.shape[-1]
    if bins <= 1:
        return np.zeros(np.broadcast_shapes(a.shape[:-1], b.shape[:-1]), dtype=float)

    cdf_difference = np.cumsum(a - b, axis=-1)
    return np.sum(np.abs(cdf_difference[..., :-1]), axis=-1) / (bins - 1)


METRICS = {
    "cosine": cosine_distance,
    "emd": earth_movers_distance,
}

def query_with_metric(features, k=5, metric_name="cosine"):
    """Retrieve nearest shapes using a selected histogram distance."""
    if metric_name not in METRICS:
        raise ValueError(f"Unknown metric: {metric_name}")
    if k < 1:
        raise ValueError("k must be at least 1")

    norm, params, _ = s4.load_database()
    hist_metric = METRICS[metric_name]
    scales = s4.fit_distance_scales(norm, hist_metric=hist_metric)

    # Find the query's complete feature row in the original Step 3 table.
    source = pd.read_csv(s4.FEATURES_CSV)

    query_class = str(features["class"])
    query_file = str(features["file"])

    match = source[
        (source["class"].astype(str) == query_class) &
        (source["file"].astype(str) == query_file)
    ]

    if match.empty:
        raise ValueError(
            f"Query {query_class}/{query_file} was not found in "
            f"{s4.FEATURES_CSV}. Use a shape present in the Step 3 feature table."
        )

    query_df = match.iloc[[0]].copy()

    # Ensure the query uses the exact global and histogram schema.
    required = s4.GLOBALS + sum(s4.HIST.values(), [])
    missing = [col for col in required if col not in query_df.columns]
    if missing:
        raise ValueError(f"Query source is missing required features: {missing}")

    query_df = query_df[["class", "file"] + required]

    # Normalize the query with the fitted database parameters.
    q = s4.apply_normalization(query_df, params).iloc[0]

    descriptor_d = s4.descriptor_distances(
        q, norm, hist_metric=hist_metric
    )
    distances = s4.combined_distance(descriptor_d, scales)

    # Exclude the query itself from retrieval.
    same_shape = (
        (norm["class"].astype(str) == query_class) &
        (norm["file"].astype(str) == query_file)
    ).to_numpy()
    distances[same_shape] = np.inf

    # Do not return invalid distances.
    distances[~np.isfinite(distances)] = np.inf
    order = np.argsort(distances, kind="stable")
    order = order[np.isfinite(distances[order])][:min(k, len(norm))]

    return pd.DataFrame({
        "rank": np.arange(1, len(order) + 1),
        "class": norm["class"].iloc[order].to_numpy(),
        "file": norm["file"].iloc[order].to_numpy(),
        "distance": distances[order],
    })


def compare_metrics(k=10):
    """Evaluate cosine and EMD on the held-out query shapes."""
    query_path = s4.QUERY_CSV
    if not query_path.exists():
        raise FileNotFoundError(
            f"Query list not found: {query_path}. Run Step 4 build first."
        )

    queries = pd.read_csv(query_path)
    if not {"class", "file"}.issubset(queries.columns):
        raise ValueError("query_shapes.csv must contain class and file columns.")

    ks = sorted(set([5, k]))
    per_query = []

    print(f"Evaluating {len(queries)} held-out queries...", flush=True)

    for _, query in queries.iterrows():
        query_class = str(query["class"])
        query_file = str(query["file"])
        query_name = f"{query_class}/{query_file}"

        features = s4.features_for(query_name)

        for metric_name in METRICS:
            results = query_with_metric(
                features, k=max(ks), metric_name=metric_name
            )

            for current_k in ks:
                top_k = results.head(current_k)
                relevant = int(
                    (top_k["class"].astype(str) == query_class).sum()
                )
                returned = len(top_k)
                precision = relevant / returned if returned else 0.0

                per_query.append({
                    "query_class": query_class,
                    "query_file": query_file,
                    "metric": metric_name,
                    "k": current_k,
                    "relevant_count": relevant,
                    "returned_count": returned,
                    "precision_at_k": precision,
                })

    details = pd.DataFrame(per_query)
    out_dir = s4.OUT
    out_dir.mkdir(parents=True, exist_ok=True)

    details_path = out_dir / "metric_comparison_per_query.csv"
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
    summary_path = out_dir / "metric_comparison_summary.csv"
    summary.to_csv(summary_path, index=False)

    per_class = (
        details.groupby(["query_class", "metric", "k"])["precision_at_k"]
        .mean()
        .reset_index(name="mean_precision_at_k")
    )
    class_path = out_dir / "metric_comparison_by_class.csv"
    per_class.to_csv(class_path, index=False)

    print("\nOVERALL HELD-OUT QUERY RESULTS")
    print(summary.round(4).to_string(index=False))
    print(f"\nPer-query results: {details_path}")
    print(f"Overall summary:   {summary_path}")
    print(f"Per-class results: {class_path}")



def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Compare cosine distance and Earth Mover's Distance."
    )
    parser.add_argument(
        "--query",
        help="Query mesh, for example Mug/D00685.obj",
    )
    parser.add_argument("-k", type=int, default=5)
    parser.add_argument(
        "--metric",
        choices=["cosine", "emd", "both"],
        default="both",
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Run the metric comparison entry point.",
    )
    args = parser.parse_args()

    if args.compare:
        compare_metrics()
        return

    if not args.query:
        parser.print_help()
        return

    # Resolve the query using the existing Step 4 helper.
    features = s4.features_for(args.query)

    selected_metrics = (
        list(METRICS) if args.metric == "both" else [args.metric]
    )

    for metric_name in selected_metrics:
        print(f"\nTop {args.k} results using {metric_name.upper()}:", flush=True)
        results = query_with_metric(
            features,
            k=args.k,
            metric_name=metric_name,
        )
        print(results.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
