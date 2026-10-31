"""
Figures for Step 6, saved in report/figures/step6/ (needs run_step6.py first).

    python figures_step6.py

1. s6_class_precision.png   precision of the ten closest shapes for every class, with the random baseline
2. s6_precision_k.png       (a) precision and recall for K = 1 .. 50, (b) precision against the size of the class
3. s6_examples.png          one query of the two best and the two worst classes with its five closest shapes
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from figures_step4 import show
from mr_io import PROJECT_ROOT
from run_step4 import features_for, query
from run_step6 import OUT

FIG = PROJECT_ROOT / "report" / "figures" / "step6"


def class_figure(per_class, summary):
    pc = per_class.sort_values("precision")
    y = np.arange(len(pc))
    fig, ax = plt.subplots(figsize=(8, 12.5))
    ax.barh(y, pc.precision, color="#1f77b4", height=0.75, label="precision of the 10 closest shapes")
    ax.scatter(pc.baseline, y, color="#d62728", s=10, zorder=3, label="random order")
    for avg, style in zip(summary.itertuples(), ["-", "--"]):
        ax.axvline(avg.precision, color="0.3", lw=1, ls=style, label=f"average {avg.average} ({avg.precision:.2f})")
    ax.set_yticks(y)
    ax.set_yticklabels([f"{c} ({n})" for c, n in zip(pc["class"], pc.shapes)], fontsize=7)
    ax.set_ylim(-0.7, len(pc) - 0.3)
    ax.set_xlim(0, 1)
    ax.set_xlabel("precision (K = 10)")
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG / "s6_class_precision.png", dpi=170)
    plt.close(fig)


def k_figure(by_k, per_class):
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].plot(by_k.k, by_k.precision, color="#1f77b4", label="precision")
    ax[0].plot(by_k.k, by_k.recall, color="#2ca02c", label="recall")
    ax[0].axhline(by_k.baseline.iloc[0], color="#d62728", lw=1, ls=":", label="precision of a random order")
    ax[0].axvline(10, color="0.5", lw=0.8, ls="--")
    ax[0].set_xlabel("number of returned shapes K")
    ax[0].set_ylim(0, 1)
    ax[0].set_title("(a) average over all queries", fontsize=10)
    ax[0].legend(fontsize=8)
    ax[1].scatter(per_class.shapes, per_class.precision, s=14, color="#1f77b4", label="class")
    n = per_class.shapes.sum()
    size = np.linspace(per_class.shapes.min(), per_class.shapes.max(), 100)
    ax[1].plot(size, (size - 1) / (n - 1), color="#d62728", lw=1, ls=":", label="random order")
    for r in per_class.itertuples():
        if r.precision > 0.6 or r.shapes > 100:            # the worst classes are named in Figure s6_class_precision
            below = r._1 == "Humanoid"                       # Jet and Humanoid lie almost on the same point
            ax[1].annotate(r._1, (r.shapes, r.precision), fontsize=7, xytext=(-8, -10) if below else (3, 2),
                           textcoords="offset points", ha="right" if below else "left")
    ax[1].set_xlabel("number of shapes in the class")
    ax[1].set_ylabel("precision (K = 10)")
    ax[1].set_ylim(0, 1)
    ax[1].set_title("(b) per class", fontsize=10)
    ax[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG / "s6_precision_k.png", dpi=170)
    plt.close(fig)


def typical_query(q, cls):
    """The query of a class whose precision is closest to the mean of that class."""
    g = q[q["class"] == cls]
    return g.iloc[(g.precision - g.precision.mean()).abs().argsort().iloc[0]]


def example_figure(q, per_class, k=5):
    pc = per_class.sort_values("precision", ascending=False)
    picks = list(pc["class"].iloc[:2]) + list(pc["class"].iloc[-2:])
    fig, ax = plt.subplots(len(picks), k + 1, figsize=(10, 1.6 * len(picks)))
    for r, cls in enumerate(picks):
        s = typical_query(q, cls)
        res = query(features_for(f"{s['class']}/{s['file']}"), k)
        show(ax[r, 0], s["class"], s["file"], f"query: {s['class']}\n{s['file'][:-4]}", "#1f77b4")
        for c, (_, m) in enumerate(res.iterrows(), start=1):
            show(ax[r, c], m["class"], m["file"], f"{m['class']}\nd = {m['distance']:.2f}",
                 "#2ca02c" if m["class"] == s["class"] else "#d62728")
    fig.tight_layout(h_pad=0.4, w_pad=0.2)
    fig.savefig(FIG / "s6_examples.png", dpi=170)
    plt.close(fig)
    return picks


def main():
    FIG.mkdir(parents=True, exist_ok=True)
    q = pd.read_csv(OUT / "step6_per_query.csv")
    per_class = pd.read_csv(OUT / "step6_per_class.csv")
    summary = pd.read_csv(OUT / "step6_summary.csv")
    by_k = pd.read_csv(OUT / "step6_precision_by_k.csv")
    class_figure(per_class, summary)
    k_figure(by_k, per_class)
    picks = example_figure(q, per_class)
    print("figures saved in", FIG, "| example classes:", ", ".join(picks))


if __name__ == "__main__":
    main()
