"""
Step 5: two-dimensional t-SNE map of the feature vectors, with the shape name and class shown on hover.

Run from the project folder:
    python run_step5_tsne.py --perplexity 30 --iterations 1000 --seed 42

Output: report/data/step4/step5/tsne_embedding.csv   2D position of every shape
        report/data/step4/step5/tsne_shapes.html     interactive scatterplot, colored by class
"""
import argparse
import time

import pandas as pd
import plotly.express as px

try:
    from sklearn.manifold import TSNE
except ImportError:
    raise SystemExit("scikit-learn is not installed, run: pip install -r requirements.txt")

from step5_common import EMBEDDING_PATH, PLOT_PATH, ensure_step5_dir, load_step4_data, vectorize_table


def main():
    ap = argparse.ArgumentParser(description="Step 5: t-SNE map of the shapes")
    ap.add_argument("--perplexity", type=float, default=30.0)
    ap.add_argument("--iterations", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    ensure_step5_dir()
    norm, params, scales = load_step4_data()
    X = vectorize_table(norm, scales)
    if not 0 < a.perplexity < len(X):
        raise ValueError(f"the perplexity must lie between 0 and {len(X)}")
    print(f"t-SNE of {len(X):,} shapes with {X.shape[1]} values each ...")
    t0 = time.perf_counter()
    model = TSNE(n_components=2, perplexity=a.perplexity, init="pca", learning_rate="auto",
                 max_iter=a.iterations, random_state=a.seed, method="barnes_hut")
    xy = model.fit_transform(X)
    elapsed = time.perf_counter() - t0

    emb = pd.DataFrame({"x": xy[:, 0], "y": xy[:, 1], "class": norm["class"].astype(str),
                        "file": norm["file"].astype(str)})
    emb["shape"] = emb["class"] + "/" + emb["file"]
    emb.to_csv(EMBEDDING_PATH, index=False)
    fig = px.scatter(emb, x="x", y="y", color="class", hover_name="shape",
                     hover_data={"class": True, "file": True, "x": False, "y": False, "shape": False},
                     title="t-SNE map of the shape database", render_mode="webgl")
    fig.update_traces(marker={"size": 6, "opacity": 0.8})
    fig.update_layout(template="plotly_white", legend_title_text="class",
                      xaxis_title="t-SNE dimension 1", yaxis_title="t-SNE dimension 2")
    fig.write_html(str(PLOT_PATH), include_plotlyjs=True, full_html=True)
    print(f"done in {elapsed:.1f} s")
    print(f"positions: {EMBEDDING_PATH}")
    print(f"interactive plot (open in a browser, hover over a point for the shape): {PLOT_PATH}")


if __name__ == "__main__":
    main()
