"""
Figures for Step 5, saved in report/figures/step5/.

    python run_step5_tsne.py --perplexity 30          first, the 2D positions (also --perplexity 40)
    python figures_step5.py

1. s5_tsne.png      the t-SNE maps for perplexity 30 and 40. The six most compact classes (with at least 20 shapes,
                    found in the map for perplexity 30) are colored in both maps, all other shapes are gray; the
                    interactive HTML of run_step5_tsne.py colors every class and shows the name of every shape
2. s5_queries.png   five query shapes from outside the database with their five closest shapes found by the
                    index, shown like the results of Step 4 (technical tips 5); the lists are compared with Step 4
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from figures_step4 import show
from mr_io import PROJECT_ROOT
from run_step4 import OUT, features_for
from run_step5 import load_index
from step5_common import STEP5_DIR, embedding_path, query_vector

FIG = PROJECT_ROOT / "report" / "figures" / "step5"
PERPLEXITIES = [30, 40]
QUERIES = ["Bicycle", "Car", "Chess", "Humanoid", "Mug"]          # five of the ten query shapes of Step 4


def compact_classes(emb, n=6, min_size=20):
    """Classes whose points lie closest together: median distance to the middle of the class."""
    spread = {}
    for cls, g in emb.groupby("class"):
        if len(g) >= min_size:
            mid = g[["x", "y"]].median().to_numpy()
            spread[cls] = np.median(np.linalg.norm(g[["x", "y"]].to_numpy() - mid, axis=1))
    return sorted(spread, key=spread.get)[:n]


def tsne_figure():
    maps = {p: pd.read_csv(embedding_path(p)) for p in PERPLEXITIES}
    shown = compact_classes(maps[PERPLEXITIES[0]])
    fig, axes = plt.subplots(1, len(maps), figsize=(13, 6))
    for ax, (p, emb) in zip(axes, maps.items()):
        rest = emb[~emb["class"].isin(shown)]
        ax.scatter(rest.x, rest.y, s=5, color="0.78", linewidths=0, label=f"other classes ({rest['class'].nunique()})")
        for i, cls in enumerate(shown):
            g = emb[emb["class"] == cls]
            ax.scatter(g.x, g.y, s=10, color=plt.cm.tab10(i), linewidths=0, label=f"{cls} ({len(g)})")
        ax.set_title(f"perplexity {p}", fontsize=11)
        ax.set_xlabel("t-SNE dimension 1")
        ax.set_ylabel("t-SNE dimension 2")
    axes[0].legend(loc="upper left", fontsize=8, markerscale=2, frameon=True)
    fig.tight_layout()
    fig.savefig(FIG / "s5_tsne.png", dpi=150)
    plt.close(fig)
    return shown


def query_figure(k=5):
    norm, params, scales, X, index, meta = load_index()
    queries = pd.read_csv(OUT / "query_shapes.csv").set_index("class").loc[QUERIES].reset_index()
    step4 = pd.read_csv(OUT / "s4_query_results.csv")
    same = 0
    fig, ax = plt.subplots(len(queries), k + 1, figsize=(10, 1.6 * len(queries)))
    for r, s in queries.iterrows():
        qv = query_vector(features_for(f"{s['class']}/{s['file']}"), params, scales).astype(np.float32)
        d2, labels = index.search(qv[None, :], k)
        found = [(meta["class"].iloc[j], meta["file"].iloc[j], float(np.sqrt(max(d, 0.0))))
                 for j, d in zip(labels[0], d2[0])]
        expected = step4[step4.query_class == s["class"]].sort_values("rank").head(k)
        same += [f for _, f, _ in found] == list(expected.file)
        show(ax[r, 0], s["class"], s["file"], f"query: {s['class']}\n{s['file'][:-4]}", "#1f77b4")
        for c, (cls, f, d) in enumerate(found, start=1):
            show(ax[r, c], cls, f, f"{cls}\nd = {d:.2f}", "#2ca02c" if cls == s["class"] else "#d62728")
    fig.tight_layout(h_pad=0.4, w_pad=0.2)
    fig.savefig(FIG / "s5_queries.png", dpi=170)
    plt.close(fig)
    return same, len(queries)


def main():
    FIG.mkdir(parents=True, exist_ok=True)
    shown = tsne_figure()
    same, n = query_figure()
    print("figures saved in", FIG)
    print("compact classes:", ", ".join(shown))
    print(f"index results identical to Step 4 for {same} of {n} queries")


if __name__ == "__main__":
    main()
