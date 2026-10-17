"""Figures and checks for Step 3.2 (report/figures/step3/, report/data/step3/).

    python figures_step3_2.py              figures (needs the feature table of run_s3_2.py)
    python figures_step3_2.py --ranges     largest descriptor values over the database (histogram ranges)
    python figures_step3_2.py --bins       sampling noise and class separation for several bin counts

1. demonstration: for each global descriptor a shape with a low and one with a high value
2. histograms of all shapes of a few classes, per descriptor (same class should look alike)
3. two shapes of the same class and one of another class with their histograms
4. one histogram with different bin counts
"""
import argparse
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from figures_step3_1 import crop_white
from mr_io import PROJECT_ROOT, mesh_arrays
from mr_normalize import STEP3_DIR
from run_s3_2 import DEFAULT_BINS, FEATURES_CSV, NAMES, RANGES, sample_distributions, seed_for

FIG = PROJECT_ROOT / "report" / "figures" / "step3"
DATA = PROJECT_ROOT / "report" / "data" / "step3"
RENDERS = FIG / "renders_3_2"
CLASSES = ["Mug", "Sword", "Humanoid", "Car"]
SIMILAR = [("Car", "m1521.obj"), ("Car", "m1526.obj"), ("Humanoid", "m213.obj")]
DEMO = [("sphericity", "sphericity"), ("rectangularity_3d", "3D rectangularity"), ("convexity", "convexity"),
        ("eccentricity", "eccentricity"), ("diameter", "diameter"), ("surface_area", "surface area")]
VOLUME_BASED = {"sphericity", "rectangularity_3d", "convexity"}
COLORS = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd"]


def render(cls, f, out_png):
    """Smooth-shaded render of a normalized shape, same view as in Step 3.1, no axes."""
    import plotly.graph_objects as go
    if out_png.exists():
        return
    V, F = mesh_arrays(STEP3_DIR / cls / f)
    fig = go.Figure(go.Mesh3d(x=V[:, 0], y=V[:, 1], z=V[:, 2], i=F[:, 0], j=F[:, 1], k=F[:, 2],
                              color="lightsteelblue", flatshading=False,
                              lighting=dict(ambient=0.5, diffuse=0.8, specular=0.2, roughness=0.6),
                              lightposition=dict(x=100, y=200, z=150)))
    fig.update_layout(scene=dict(aspectmode="data", xaxis_visible=False, yaxis_visible=False, zaxis_visible=False,
                                 camera=dict(eye=dict(x=0.55, y=0.45, z=1.9), up=dict(x=0, y=1, z=0))),
                      margin=dict(l=0, r=0, t=0, b=0), width=520, height=430)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.write_image(str(out_png), scale=2)
    crop_white(out_png)


def show(ax, cls, f):
    png = RENDERS / f"{cls}_{Path(f).stem}.png"
    render(cls, f, png)
    ax.imshow(plt.imread(png))
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)


def hist_cols(name):
    return [f"{name}_bin_{i:02d}" for i in range(DEFAULT_BINS)]


def bin_centers(name):
    lo, hi = RANGES[name]
    edges = np.linspace(lo, hi, DEFAULT_BINS + 1)
    return 0.5 * (edges[:-1] + edges[1:])


def pick(d, col, q):
    """The shape whose value is closest to quantile q of the column."""
    target = d[col].quantile(q)
    return d.loc[(d[col] - target).abs().idxmin()]


def demonstration(d):
    rows = []
    fig, ax = plt.subplots(len(DEMO), 2, figsize=(6.4, 2.25 * len(DEMO)))
    for r, (col, label) in enumerate(DEMO):
        dd = d[d.volume_source.isin(["closed", "filled"])] if col in VOLUME_BASED else d
        for c, q in enumerate([0.03, 0.97]):
            row = pick(dd, col, q)
            show(ax[r, c], row["class"], row.file)
            ax[r, c].set_title(f"{label} = {row[col]:.3g}\n{row['class']} {Path(row.file).stem}", fontsize=8.5)
            rows.append((label, "low" if c == 0 else "high", row["class"], row.file, row[col]))
    fig.tight_layout()
    fig.savefig(FIG / "s32_demo_descriptors.png", dpi=170)
    plt.close(fig)
    pd.DataFrame(rows, columns=["descriptor", "end", "class", "file", "value"]).to_csv(
        DATA / "s32_demo_shapes.csv", index=False)
    print(pd.DataFrame(rows).to_string())


def class_histograms(d):
    fig, ax = plt.subplots(len(CLASSES), len(NAMES), figsize=(13, 2.0 * len(CLASSES)), sharex="col")
    for r, cls in enumerate(CLASSES):
        g = d[d["class"] == cls]
        for c, name in enumerate(NAMES):
            x = bin_centers(name)
            for h in g[hist_cols(name)].values:
                ax[r, c].plot(x, h, color=COLORS[r], alpha=0.35, linewidth=0.8)
            ax[r, c].tick_params(labelsize=7)
            if r == 0:
                ax[r, c].set_title(name, fontsize=10)
            if c == 0:
                ax[r, c].set_ylabel(f"{cls} ({len(g)})", fontsize=9)
    fig.tight_layout()
    fig.savefig(FIG / "s32_class_histograms.png", dpi=170)
    plt.close(fig)


def similar_test(d):
    fig = plt.figure(figsize=(13, 2.6))
    gs = fig.add_gridspec(1, 3 + len(NAMES), width_ratios=[1, 1, 1] + [1.25] * len(NAMES))
    rows = [d[(d["class"] == c) & (d.file == f)].iloc[0] for c, f in SIMILAR]
    for k, (row, col) in enumerate(zip(rows, ["#1f77b4", "#17becf", "#d62728"])):
        a = fig.add_subplot(gs[0, k])
        show(a, row["class"], row.file)
        a.set_title(f"{row['class']} {Path(row.file).stem}", fontsize=9, color=col)
    for c, name in enumerate(NAMES):
        a = fig.add_subplot(gs[0, 3 + c])
        for row, col in zip(rows, ["#1f77b4", "#17becf", "#d62728"]):
            a.plot(bin_centers(name), row[hist_cols(name)].values.astype(float), color=col, linewidth=1.4)
        a.set_title(name, fontsize=10)
        a.tick_params(labelsize=7)
    fig.tight_layout()
    fig.savefig(FIG / "s32_similar_test.png", dpi=170)
    plt.close(fig)
    # L1 distances between the histograms, used in the report
    for name in NAMES:
        H = [row[hist_cols(name)].values.astype(float) for row in rows]
        print(f"{name}: car-car {np.abs(H[0]-H[1]).sum():.3f}, car1-human {np.abs(H[0]-H[2]).sum():.3f}, "
              f"car2-human {np.abs(H[1]-H[2]).sum():.3f}")


def bins_figure():
    cls, f = "Car", "m1518.obj"
    V, F = mesh_arrays(STEP3_DIR / cls / f)
    vals = sample_distributions(V, 1_000_000, np.random.default_rng(seed_for(cls, f)))["D2"]
    fig, ax = plt.subplots(1, 3, figsize=(10, 2.4), sharey=False)
    for a, b in zip(ax, [8, 20, 64]):
        lo, hi = RANGES["D2"]
        h, e = np.histogram(np.clip(vals, lo, hi), bins=b, range=(lo, hi))
        a.bar(e[:-1], h / h.sum(), width=e[1] - e[0], align="edge", color="#1f77b4", edgecolor="white", linewidth=0.4)
        a.set_title(f"B = {b}", fontsize=10)
        a.set_xlabel("D2", fontsize=9)
        a.tick_params(labelsize=7)
    ax[0].set_ylabel("frequency", fontsize=9)
    fig.tight_layout()
    fig.savefig(FIG / "s32_bins.png", dpi=170)
    plt.close(fig)


def _maxima(path):
    V, F = mesh_arrays(path)
    vals = sample_distributions(V, 200_000, np.random.default_rng(1))
    return [path.parent.name, path.name] + [float(np.max(vals[n])) for n in NAMES]


def ranges():
    paths = sorted(STEP3_DIR.glob("*/*.obj"))
    with ProcessPoolExecutor(8) as ex:
        rows = list(ex.map(_maxima, paths, chunksize=8))
    d = pd.DataFrame(rows, columns=["class", "file"] + NAMES)
    d.to_csv(DATA / "s32_descriptor_maxima.csv", index=False)
    print("largest values over the database:", d[NAMES].max().round(4).to_dict())


def _noise(path, bins_list=(8, 16, 20, 32, 64, 128), samples_list=(10_000, 1_000_000)):
    V, F = mesh_arrays(path)
    out = []
    for S in samples_list:
        a = sample_distributions(V, S, np.random.default_rng(1))
        b = sample_distributions(V, S, np.random.default_rng(2))
        for name in NAMES:
            lo, hi = RANGES[name]
            for B in bins_list:
                ha = np.histogram(np.clip(a[name], lo, hi), bins=B, range=(lo, hi))[0] / len(a[name])
                hb = np.histogram(np.clip(b[name], lo, hi), bins=B, range=(lo, hi))[0] / len(b[name])
                out.append((path.parent.name, path.name, S, name, B, 0.5 * np.abs(ha - hb).sum(), ha))
    return out


def bin_study():
    """Two random shapes per class: noise between two random seeds, and class separation, per B."""
    rng = np.random.default_rng(0)
    paths = []
    for c in sorted(p for p in STEP3_DIR.iterdir() if p.is_dir()):
        files = sorted(c.glob("*.obj"))
        paths += [files[i] for i in rng.choice(len(files), 2, replace=False)]
    with ProcessPoolExecutor(8) as ex:
        rows = [r for rr in ex.map(_noise, paths) for r in rr]
    d = pd.DataFrame(rows, columns=["class", "file", "samples", "name", "B", "noise", "h"])
    noise = d.pivot_table(index=["name", "samples"], columns="B", values="noise")
    sep = []
    for (name, B), g in d[d.samples == 1_000_000].groupby(["name", "B"]):
        H = np.stack(g.h.values)
        cl = np.asarray(g["class"], dtype=object)
        D = 0.5 * np.abs(H[:, None] - H[None]).sum(-1)
        same = cl[:, None] == cl[None]
        np.fill_diagonal(same, False)
        sep.append((name, B, D[same].mean(), D[cl[:, None] != cl[None]].mean()))
    sep = pd.DataFrame(sep, columns=["name", "B", "within", "between"])
    sep["ratio"] = sep.between / sep.within
    noise.to_csv(DATA / "s32_bins_noise.csv")
    sep.to_csv(DATA / "s32_bins_separation.csv", index=False)
    print(f"{len(paths)} shapes\nnoise:\n{noise.round(4)}\nseparation (between / within):")
    print(sep.pivot(index="name", columns="B", values="ratio").round(3))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--ranges", action="store_true")
    p.add_argument("--bins", action="store_true")
    a = p.parse_args()
    FIG.mkdir(parents=True, exist_ok=True)
    if a.ranges:
        ranges()
    if a.bins:
        bin_study()
    if not (a.ranges or a.bins):
        d = pd.read_csv(FEATURES_CSV)
        demonstration(d)
        class_histograms(d)
        similar_test(d)
        bins_figure()
        print("figures saved in", FIG)
