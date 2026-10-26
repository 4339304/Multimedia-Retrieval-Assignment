"""Large figures of the difficult decimation cases of Step 2.3 (report/figures/step2/).

    python figures_step2_3.py

1. the four Houses and the PlantIndoors / Tree for which quadric decimation alone fails:
   cleaned original, quadric decimation alone, final result (one shape per row, large)
2. the two shapes that the area check accepts with a larger Hausdorff distance
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from figures_step3_1 import crop_white
from mr_io import DATA_DIR, PROJECT_ROOT, mesh_arrays, read_obj_raw, render_png
from mr_resample import clean_mesh, cluster_to_vertices, decimate_to_vertices, fidelity, fix_orientation

FIG = PROJECT_ROOT / "report" / "figures" / "step2"
TMP = FIG / "renders_2_3"
LOG = PROJECT_ROOT / "report" / "data" / "step2" / "resample_log_full.csv"
HOUSES = [("House", "D00286.obj"), ("House", "D00901.obj"), ("House", "D00574.obj"), ("House", "D00903.obj")]
PLANTS = [("PlantIndoors", "D00080.obj"), ("Tree", "D00765.obj")]
SHAPES_2 = [("Skyscraper", "D00058.obj"), ("Tree", "D00479.obj")]
METHOD = {"quadric": "quadric decimation", "quadric_bw100": "quadric, boundary weight",
          "vertex_clustering": "vertex clustering"}


def cleaned(cls, f):
    v, s, i = read_obj_raw(DATA_DIR / cls / f)
    V, F, _ = clean_mesh(v, s, i)
    F, _ = fix_orientation(V, F)
    return V, F


def panel(ax, V, F, png, title):
    render_png(V, F, png, width=900, height=700, max_faces=10**7)   # full mesh, no display decimation
    crop_white(png)
    ax.imshow(plt.imread(png))
    ax.set_title(title, fontsize=11)
    ax.axis("off")


def comparison(shapes, out, final_label, middle="quadric"):
    """One shape per row: cleaned original, an alternative result (quadric alone, or vertex clustering),
    and the final result of Step 2.3."""
    log = pd.read_csv(LOG).set_index(["class", "file"])
    TMP.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(len(shapes), 3, figsize=(15, 4.4 * len(shapes)), squeeze=False)
    for r, (cls, f) in enumerate(shapes):
        rec = log.loc[(cls, f)]
        V, F = cleaned(cls, f)
        if middle == "quadric":
            Vq, Fq, _ = decimate_to_vertices(V, F)
            label = f"quadric decimation alone ({len(Vq):,} vertices)"
        else:
            Vq, Fq, _ = cluster_to_vertices(V, F)
            label = f"vertex clustering ({len(Vq):,} vertices, $\\delta_H$ = {100 * fidelity(V, F, Vq, Fq)['hausdorff_rel']:.1f} %)"
        Vf, Ff = mesh_arrays(PROJECT_ROOT / "processed" / "step2_full" / cls / f)
        stem = f"{cls}_{Path(f).stem}"
        panel(ax[r, 0], V, F, TMP / f"{stem}_0.png", f"{cls} {Path(f).stem}: cleaned original ({len(V):,} vertices)")
        panel(ax[r, 1], Vq, Fq, TMP / f"{stem}_1{middle[0]}.png", label)
        panel(ax[r, 2], Vf, Ff, TMP / f"{stem}_2.png",
              f"{final_label(rec)} ({len(Vf):,} vertices, $\\delta_H$ = {100 * rec.hausdorff_rel:.1f} %)")
    fig.tight_layout()
    fig.savefig(FIG / out, dpi=150)
    plt.close(fig)
    print("saved", FIG / out)


def conflict_faces(F):
    """Faces that share an edge with exactly one neighbor that lists the edge in the same direction
    (the same test as the orientation conflicts of Step 2.1; edges of three or more faces are not counted)."""
    import numpy as np
    he = np.concatenate([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]])
    face = np.tile(np.arange(len(F)), 3)
    key = np.minimum(he[:, 0], he[:, 1]).astype(np.int64) * (int(F.max()) + 1) + np.maximum(he[:, 0], he[:, 1])
    _, inv, cnt = np.unique(key, return_inverse=True, return_counts=True)
    two = cnt[inv] == 2                                           # edges shared by exactly two faces
    order = np.argsort(key, kind="stable")
    k, h, fc, t = key[order], he[order], face[order], two[order]
    same = (k[1:] == k[:-1]) & (h[1:, 0] == h[:-1, 0]) & t[1:]   # both faces list the edge in the same direction
    bad = np.zeros(len(F), dtype=bool)
    bad[fc[1:][same]] = True
    bad[fc[:-1][same]] = True
    return bad


def orientation_render(V, F, png, title):
    """Shaded mesh with the faces that have an orientation conflict in red."""
    import numpy as np
    import plotly.graph_objects as go
    bad = conflict_faces(F)
    colors = np.where(bad, "#d62728", "#b0c4de")
    fig = go.Figure(go.Mesh3d(x=V[:, 0], y=V[:, 1], z=V[:, 2], i=F[:, 0], j=F[:, 1], k=F[:, 2],
                              facecolor=colors, flatshading=True,
                              lighting=dict(ambient=0.6, diffuse=0.7, specular=0.1)))
    fig.update_layout(scene=dict(aspectmode="data", xaxis=dict(visible=False), yaxis=dict(visible=False),
                                 zaxis=dict(visible=False), camera=dict(eye=dict(x=1.7, y=1.7, z=1.1))),
                      margin=dict(l=0, r=0, t=0, b=0), width=900, height=700)
    fig.write_image(str(png), scale=2)
    return int(bad.sum())


def orientation_figure(cls="Car", f="m1558.obj"):
    """Faces with orientation conflicts (red) before and after the orientation step of Step 2.3."""
    v, s, i = read_obj_raw(DATA_DIR / cls / f)
    V, F, _ = clean_mesh(v, s, i)
    TMP.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(1, 2, figsize=(13, 5))
    for k, (VV, FF, when) in enumerate([(V, F, "after cleaning, before the orientation step"),
                                        (*mesh_arrays(PROJECT_ROOT / "processed" / "step2_full" / cls / f),
                                         "after Step 2.3")]):
        png = TMP / f"orientation_{cls}_{Path(f).stem}_{k}.png"
        n = orientation_render(VV, FF, png, when)
        ax[k].imshow(plt.imread(png))
        ax[k].set_title(f"{cls} {Path(f).stem}, {when}: {n:,} faces with a conflict", fontsize=11)
        ax[k].axis("off")
    fig.tight_layout()
    fig.savefig(FIG / "orientation_conflicts.png", dpi=150)
    plt.close(fig)
    print("saved", FIG / "orientation_conflicts.png")


if __name__ == "__main__":
    orientation_figure()
    final = lambda rec: f"final: {METHOD.get(rec.decim_method, rec.decim_method)}"
    comparison(HOUSES[:2], "exceptions_houses_a.png", final)
    comparison(HOUSES[2:], "exceptions_houses_b.png", final)
    comparison(PLANTS, "exceptions_plant_tree.png", final)
    comparison(SHAPES_2, "area_check_two_shapes.png", final, middle="clustering")
