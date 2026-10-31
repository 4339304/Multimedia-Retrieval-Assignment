"""Figures of the database statistics of Step 2.2 (report/figures/step2/).

    python figures_step2_2.py

Needs the statistics table of Step 2.1-2.2 (report/data/step2/shape_stats_raw.csv, written by run_step2.py).
1. hist_vertices_faces.png              histograms of the vertex and face counts (logarithmic axis)
2. class_sizes_and_median_vertices.png  shapes per class and median vertex count per class
3. gallery_outliers.png                 the three smallest, three median and three largest shapes in the viewer
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from mr_io import DATA_DIR, PROJECT_ROOT, mesh_arrays, render_png
from mr_resample import BAND

FIG = PROJECT_ROOT / "report" / "figures" / "step2"
STATS = PROJECT_ROOT / "report" / "data" / "step2" / "shape_stats_raw.csv"
LO, HI = BAND
plt.rcParams.update({"axes.grid": True, "grid.alpha": 0.3})


def hist_log(ax, s, label):
    bins = np.logspace(np.log10(s.min()), np.log10(s.max()), 45)
    ax.hist(s, bins=bins, color="steelblue", edgecolor="white")
    ax.set_xscale("log")
    ax.set_xlabel(label + " (log scale)")
    ax.set_ylabel("number of shapes")


def histograms(stats):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    hist_log(axes[0], stats["n_vertices"], "vertices")
    for x, c, t, ha in [(100, "crimson", "100", "right"), (LO, "darkgreen", "5,000", "right"),
                        (HI, "darkgreen", "10,000", "left")]:
        axes[0].axvline(x, color=c, ls="--", lw=1)
        axes[0].text(x, axes[0].get_ylim()[1] * .95, t, color=c, fontsize=8, ha=ha)
    axes[0].set_title("Vertices per shape")
    hist_log(axes[1], stats["n_faces"], "faces")
    axes[1].set_title("Faces per shape")
    plt.tight_layout()
    fig.savefig(FIG / "hist_vertices_faces.png", dpi=150)
    plt.close(fig)


def class_sizes(stats):
    cls = stats.groupby("class").agg(n_shapes=("file", "size"), median_vertices=("n_vertices", "median"))
    cls = cls.sort_values("n_shapes", ascending=False)
    fig, axes = plt.subplots(2, 1, figsize=(13, 8), sharex=True)
    axes[0].bar(cls.index, cls["n_shapes"], color="steelblue")
    axes[0].set_ylabel("shapes per class")
    axes[1].bar(cls.index, cls["median_vertices"], color="darkorange")
    axes[1].set_yscale("log")
    axes[1].set_ylabel("median vertices per shape (log)")
    for y in (LO, HI):
        axes[1].axhline(y, color="darkgreen", ls="--", lw=1)
    plt.xticks(rotation=90, fontsize=8)
    plt.tight_layout()
    fig.savefig(FIG / "class_sizes_and_median_vertices.png", dpi=150)
    plt.close(fig)


def gallery(stats):
    """The three smallest shapes, the three closest to the median vertex count and the three largest shapes."""
    med = stats["n_vertices"].median()
    rows = pd.concat([stats.nsmallest(3, "n_vertices"),
                      stats.iloc[(stats["n_vertices"] - med).abs().to_numpy().argsort()[:3]],
                      stats.nlargest(3, "n_vertices")])
    fig, axes = plt.subplots(3, 3, figsize=(4.2 * 3, 3.6 * 3))
    for ax, (_, r) in zip(axes.ravel(), rows.iterrows()):
        cls, f, nv, nf = r["class"], r["file"], int(r["n_vertices"]), int(r["n_faces"])
        V, F = mesh_arrays(DATA_DIR / cls / f)
        png = FIG / "outliers" / f"{cls}_{f[:-4]}.png"
        render_png(V, F, png, wireframe=True, title=f"{cls} {f}: {nv:,} V / {nf:,} F", width=600, height=480)
        ax.imshow(plt.imread(png))
        ax.set_title(f"{cls} {f}\n{nv:,} vertices", fontsize=9)
        ax.axis("off")
    plt.tight_layout()
    fig.savefig(FIG / "gallery_outliers.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    if not STATS.exists():
        raise SystemExit(f"{STATS} not found: run 'python run_step2.py' first (Steps 2.1-2.2)")
    FIG.mkdir(parents=True, exist_ok=True)
    stats = pd.read_csv(STATS)
    histograms(stats)
    class_sizes(stats)
    gallery(stats)
    print("saved in", FIG)


if __name__ == "__main__":
    main()
