"""Figures for Step 3.1 (report/figures/step3/).

    python figures_step3_1.py

1. before/after renders of a few shapes, with the coordinate axes drawn at the origin
2. histograms of the normalization measures before and after, over the whole database
Needs the log of run_step3_1.py.
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from mr_io import PROJECT_ROOT, mesh_arrays
from mr_normalize import STEP2_DIR, STEP3_DIR

FIG = PROJECT_ROOT / "report" / "figures" / "step3"
LOG = PROJECT_ROOT / "report" / "data" / "step3" / "normalize_log.csv"
EXAMPLES = [("Sword", "m702.obj"), ("Humanoid", "m213.obj"), ("Bicycle", "D00016.obj"), ("Chess", "m1608.obj")]
AXIS_COLORS = {"x": "#d62728", "y": "#2ca02c", "z": "#1f77b4"}


def render_with_axes(V, F, out_png, axis_len=None):
    """Smooth-shaded render with the x (red), y (green) and z (blue) axes drawn from the origin."""
    import plotly.graph_objects as go
    V = np.asarray(V, dtype=float)
    ext = (V.max(axis=0) - V.min(axis=0)).max()
    L = axis_len or 0.6 * ext
    data = [go.Mesh3d(x=V[:, 0], y=V[:, 1], z=V[:, 2], i=F[:, 0], j=F[:, 1], k=F[:, 2],
                      color="lightsteelblue", flatshading=False, opacity=0.85,
                      lighting=dict(ambient=0.5, diffuse=0.8, specular=0.2, roughness=0.6),
                      lightposition=dict(x=100, y=200, z=150))]
    for k, (name, col) in enumerate(AXIS_COLORS.items()):
        end = np.zeros(3)
        end[k] = L
        data.append(go.Scatter3d(x=[0, end[0]], y=[0, end[1]], z=[0, end[2]], mode="lines+text",
                                 text=["", name], textfont=dict(size=16, color=col),
                                 line=dict(color=col, width=8), showlegend=False))
    lo = np.minimum(V.min(axis=0), 0) - 0.05 * ext
    hi = np.maximum(V.max(axis=0), 0) + 0.15 * ext
    fig = go.Figure(data=data)
    fig.update_layout(
        scene=dict(aspectmode="data",
                   xaxis=dict(visible=False, range=[lo[0], hi[0]]),
                   yaxis=dict(visible=False, range=[lo[1], hi[1]]),
                   zaxis=dict(visible=False, range=[lo[2], hi[2]]),
                   # x to the right, y up, z towards the viewer (slightly from above and the side)
                   camera=dict(eye=dict(x=0.55, y=0.45, z=1.9), up=dict(x=0, y=1, z=0))),
        margin=dict(l=0, r=0, t=0, b=0), width=520, height=430)
    out_png = Path(out_png)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.write_image(str(out_png), scale=2)
    crop_white(out_png)


def crop_white(png, pad=12):
    """Cut the white border around a rendered image."""
    import matplotlib.image as mpimg
    img = mpimg.imread(png)
    mask = img[..., :3].min(axis=2) < 0.97
    ys, xs = mask.nonzero()
    y0, y1 = max(ys.min() - pad, 0), min(ys.max() + pad, img.shape[0])
    x0, x1 = max(xs.min() - pad, 0), min(xs.max() + pad, img.shape[1])
    plt.imsave(png, img[y0:y1, x0:x1])


def renders():
    for cls, f in EXAMPLES:
        stem = f"{cls}_{Path(f).stem}"
        V, F = mesh_arrays(STEP2_DIR / cls / f)
        render_with_axes(V, F, FIG / f"{stem}_before.png")
        V, F = mesh_arrays(STEP3_DIR / cls / f)
        render_with_axes(V, F, FIG / f"{stem}_after.png", axis_len=0.6)
        print("rendered", stem)


def histograms():
    d = pd.read_csv(LOG)
    d = d[d.status == "ok"]
    fig, axes = plt.subplots(2, 3, figsize=(12, 5.6))
    ax = axes.ravel()

    def pair(a, before, after, bins, title, xlabel, logx=False):
        a.hist(after, bins=bins, color="#1f77b4", label="after")
        a.hist(before, bins=bins, histtype="step", color="#555555", linewidth=1.4, label="before")
        a.set_title(title, fontsize=10)
        a.set_xlabel(xlabel, fontsize=9)
        if logx:
            a.set_xscale("log")
        a.tick_params(labelsize=8)

    pair(ax[0], d.bary_dist_before, d.bary_dist_after, np.logspace(-15, 9, 49),
         "(a) translation", r"$\|\bar v\|$: distance of the barycenter to the origin", logx=True)
    # alignment of all three axes; shapes with (almost) equal eigenvalues have no defined axis and are left out
    d1 = d[d.ratio_12_after > 1.01]
    d2 = d1[d1.ratio_23_after > 1.01]
    for k, (dd, e, axis) in enumerate([(d1, "e1", "x"), (d2, "e2", "y"), (d2, "e3", "z")], start=1):
        pair(ax[k], dd[f"angle_{e}_before"], dd[f"angle_{e}_after"], np.linspace(0, 90, 19),
             f"({'bcd'[k - 1]}) alignment, ${e[0]}_{e[1]}$", rf"angle between ${e[0]}_{e[1]}$ and the ${axis}$ axis (deg)")
    neg_b = (d[["f_x_before", "f_y_before", "f_z_before"]] < 0).sum(axis=1)
    neg_a = (d[["f_x_after", "f_y_after", "f_z_after"]] < 0).sum(axis=1)
    pair(ax[4], neg_b, neg_a, np.arange(-0.5, 4), "(e) flipping", r"number of axes with $f_k<0$")
    ax[4].set_xticks([0, 1, 2, 3])
    pair(ax[5], d.d_max_before, d.d_max_after, np.logspace(-2, 9, 45),
         "(f) scaling", r"$D_{\max}$: largest side of the bounding box", logx=True)
    ax[0].set_ylabel("number of shapes", fontsize=9)
    ax[3].set_ylabel("number of shapes", fontsize=9)
    ax[0].legend(fontsize=8)
    fig.tight_layout()
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / "normalization_histograms.png", dpi=200)
    print("histograms saved;", f"d_max before: {d.d_max_before.min():.3f}..{d.d_max_before.max():.3f}")


if __name__ == "__main__":
    histograms()
    renders()
