"""Figures for Step 4 (first part), saved in report/figures/step4/.

    python figures_step4.py

1. the global descriptors before and after the normalization
2. every query shape with its five closest database shapes (Euclidean distance)
Needs run_step4.py --build first.
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from figures_step3_2 import RENDERS, render
from mr_io import PROJECT_ROOT
from run_s3_2 import FEATURES_CSV
from run_step4 import (GLOBALS, LOG_FEATURES, NORM_CSV, OUT, QUERY_CSV, query, query_features,
                       split_database)

FIG = PROJECT_ROOT / "report" / "figures" / "step4"
LABELS = {"surface_area": "surface area", "sphericity": "sphericity", "rectangularity_3d": "rectangularity",
          "diameter": "diameter", "convexity": "convexity", "eccentricity": "eccentricity"}


def normalization_figure():
    db, _ = split_database(pd.read_csv(FEATURES_CSV))
    norm = pd.read_csv(NORM_CSV)
    fig, ax = plt.subplots(2, len(GLOBALS), figsize=(14, 4.4))
    for k, c in enumerate(GLOBALS):
        ax[0, k].hist(db[c].dropna(), bins=40, color="#888888")
        ax[0, k].set_title(LABELS[c], fontsize=10)
        ax[1, k].hist(norm[c].dropna(), bins=40, color="#1f77b4")
        ax[1, k].set_xlabel("standardized" + (" (of log$_{10}$)" if c in LOG_FEATURES else ""), fontsize=8)
        for a in ax[:, k]:
            a.tick_params(labelsize=7)
    ax[0, 0].set_ylabel("before\nnumber of shapes", fontsize=9)
    ax[1, 0].set_ylabel("after\nnumber of shapes", fontsize=9)
    fig.tight_layout()
    fig.savefig(FIG / "s4_normalization.png", dpi=170)
    plt.close(fig)


def show(ax, cls, f, title, color="black"):
    png = RENDERS / f"{cls}_{Path(f).stem}.png"
    render(cls, f, png)
    ax.imshow(plt.imread(png))
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_title(title, fontsize=7.5, color=color)


def query_figure(k=5):
    q = pd.read_csv(QUERY_CSV)
    rows = []
    fig, ax = plt.subplots(len(q), k + 1, figsize=(10, 1.55 * len(q)))
    for r, (_, s) in enumerate(q.iterrows()):
        res = query(query_features(PROJECT_ROOT / "Data" / s["class"] / s["file"])[0], k)
        show(ax[r, 0], s["class"], s["file"], f"query: {s['class']}\n{Path(s['file']).stem}", "#1f77b4")
        for c, (_, m) in enumerate(res.iterrows(), start=1):
            ok = m["class"] == s["class"]
            show(ax[r, c], m["class"], m["file"], f"{m['class']}\nd = {m['distance']:.2f}",
                 "#2ca02c" if ok else "#d62728")
            rows.append({"query_class": s["class"], "query_file": s["file"], **m.to_dict()})
    fig.tight_layout(h_pad=0.4, w_pad=0.2)
    fig.savefig(FIG / "s4_queries.png", dpi=170)
    plt.close(fig)
    pd.DataFrame(rows).to_csv(OUT / "s4_query_results.csv", index=False)


if __name__ == "__main__":
    FIG.mkdir(parents=True, exist_ok=True)
    normalization_figure()
    query_figure()
    print("figures saved in", FIG)
