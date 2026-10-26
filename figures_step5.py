"""
Step 5: figure of the t-SNE map for the report (report/figures/step5/s5_tsne.png).

    python run_step5_tsne.py        first, writes the 2D positions
    python figures_step5.py

Every shape is a point. The six most compact classes with at least 20 shapes are colored, all other shapes
are grey; the interactive HTML of run_step5_tsne.py colors every class and shows the
name of every shape.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from mr_io import PROJECT_ROOT
from step5_common import EMBEDDING_PATH

FIG = PROJECT_ROOT / "report" / "figures" / "step5"


def compact_classes(emb, n=6, min_size=20):
    """Classes whose points lie closest together: median distance to the middle of the class."""
    spread = {}
    for cls, g in emb.groupby("class"):
        if len(g) >= min_size:
            mid = g[["x", "y"]].median().to_numpy()
            spread[cls] = np.median(np.linalg.norm(g[["x", "y"]].to_numpy() - mid, axis=1))
    return sorted(spread, key=spread.get)[:n]


def main():
    emb = pd.read_csv(EMBEDDING_PATH)
    FIG.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(9, 7))
    shown = compact_classes(emb)
    rest = emb[~emb["class"].isin(shown)]
    ax.scatter(rest.x, rest.y, s=6, color="0.78", linewidths=0, label=f"other classes ({rest['class'].nunique()})")
    for i, cls in enumerate(shown):
        g = emb[emb["class"] == cls]
        ax.scatter(g.x, g.y, s=12, color=plt.cm.tab10(i), linewidths=0, label=f"{cls} ({len(g)})")
    ax.legend(loc="upper left", fontsize=9, markerscale=2, frameon=True)
    ax.set_xlabel("t-SNE dimension 1")
    ax.set_ylabel("t-SNE dimension 2")
    fig.tight_layout()
    fig.savefig(FIG / "s5_tsne.png", dpi=150)
    print("saved", FIG / "s5_tsne.png", "| labeled:", ", ".join(compact_classes(emb)))


if __name__ == "__main__":
    main()
