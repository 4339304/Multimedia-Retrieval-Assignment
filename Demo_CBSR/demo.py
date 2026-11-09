"""
Demo of the content-based 3D shape retrieval system.

Two shapes from outside the database (one Jet and one Car, taken out of the database before it was built)
are run through the whole pipeline, one step at a time:

    Step 1   read the OBJ file and show the shape
    Step 2   clean, reorient and resample the mesh
    Step 3   normalize (translation, PCA alignment, flipping, scaling) and compute the descriptors
    Step 4   compare with all 2,473 database shapes and show the 10 closest ones
    Step 5   the same query through the HNSW index (FAISS)

Every 3D view and chart opens in the web browser. After each step the program waits for Enter.

Usage:
    python demo.py                 both queries (Jet, then Car)
    python demo.py --only Car      one query
    python demo.py --no-pause      no waiting for Enter
    python demo.py --no-show       no browser windows (the pages are still saved in demo_output/)
"""
import argparse
import sys
import time
import webbrowser
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from mr_io import DATA_DIR, read_obj_raw, triangulate
from mr_normalize import STEP3_DIR, align, center, flip, moments, principal_axes, scale
from mr_resample import area_cv, clean_mesh, count_conflicts, fix_orientation, resample_mesh
from run_s3_2 import NAMES, RANGES, features_of
import run_step4 as s4
from step5_common import load_step4_data, query_vector, vectorize_table

QUERIES = [("Jet", "m1161.obj"), ("Car", "m1512.obj")]
K = 10
OUT = HERE / "demo_output"
LIGHT = dict(ambient=0.5, diffuse=0.8, specular=0.2, roughness=0.6)
LIGHT_POS = dict(x=100, y=200, z=150)
SHOW = True
PAUSE = True


# ------------------------------------------------------------------ helpers
def say(text=""):
    print(text, flush=True)


def title(text):
    say("\n" + "=" * 78)
    say(text)
    say("=" * 78)


def wait():
    if PAUSE:
        input("\n  [press Enter to continue] ")


def open_page(fig, name, page_title):
    """Save a figure as an HTML page in demo_output/ and open it in the browser."""
    OUT.mkdir(exist_ok=True)
    path = OUT / f"{name}.html"
    fig.update_layout(title=dict(text=page_title, x=0.5, font=dict(size=20)))
    fig.write_html(path, include_plotlyjs=True)
    if SHOW:
        webbrowser.open(path.resolve().as_uri())


def mesh_traces(V, F, wireframe=False, flat=False):
    traces = [go.Mesh3d(x=V[:, 0], y=V[:, 1], z=V[:, 2], i=F[:, 0], j=F[:, 1], k=F[:, 2],
                        color="lightsteelblue", flatshading=flat, lighting=LIGHT, lightposition=LIGHT_POS,
                        hoverinfo="skip", showscale=False)]
    if wireframe:
        e = np.unique(np.sort(np.concatenate([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]]), axis=1), axis=0)
        a, b = V[e[:, 0]], V[e[:, 1]]
        nan = np.full((len(e), 1), np.nan)
        xs, ys, zs = (np.hstack([a[:, k:k + 1], b[:, k:k + 1], nan]).ravel() for k in range(3))
        traces.append(go.Scatter3d(x=xs, y=ys, z=zs, mode="lines", line=dict(color="#34476b", width=1),
                                   hoverinfo="skip", showlegend=False))
    return traces


def axes_traces(r):
    out = []
    for vec, col, name in (((1, 0, 0), "red", "x"), ((0, 1, 0), "green", "y"), ((0, 0, 1), "blue", "z")):
        p = np.array(vec, float) * r
        out.append(go.Scatter3d(x=[-p[0], p[0]], y=[-p[1], p[1]], z=[-p[2], p[2]], mode="lines+text",
                                line=dict(color=col, width=5), text=["", name], textfont=dict(size=16, color=col),
                                hoverinfo="skip", showlegend=False))
    return out


def scene(cube=None):
    ax = dict(visible=False)
    if cube is not None:
        ax = dict(visible=False, range=[-cube, cube])
        return dict(xaxis=ax, yaxis=ax, zaxis=ax, aspectmode="cube")
    return dict(xaxis=ax, yaxis=ax, zaxis=ax, aspectmode="data")


def side_by_side(panels, name, page_title, cube=None):
    """panels: list of (subtitle, traces)."""
    fig = make_subplots(rows=1, cols=len(panels), specs=[[{"type": "scene"}] * len(panels)],
                        subplot_titles=[p[0] for p in panels], horizontal_spacing=0.02)
    for c, (_, traces) in enumerate(panels, start=1):
        for t in traces:
            fig.add_trace(t, row=1, col=c)
        fig.update_layout(**{f"scene{'' if c == 1 else c}": scene(cube)})
    fig.update_layout(height=650, margin=dict(l=10, r=10, t=90, b=10), showlegend=False)
    open_page(fig, name, page_title)


def load_result_mesh(cls, file):
    """A database shape after Step 3.1 (stored in processed/step3_normalized)."""
    verts, sizes, idx = read_obj_raw(STEP3_DIR / cls / file)
    return verts, triangulate(sizes, idx)


# ------------------------------------------------------------------ the steps
def step1(cls, file):
    title(f"STEP 1  Reading and viewing  -  query {cls} {file}")
    path = DATA_DIR / cls / file
    t0 = time.perf_counter()
    verts, sizes, idx = read_obj_raw(path)
    F = triangulate(sizes, idx)
    say(f"  file:          {path.relative_to(HERE)}")
    say(f"  vertices:      {len(verts):,}")
    say(f"  faces:         {len(sizes):,}   face type: "
        f"{'triangles only' if (sizes == 3).all() else 'mixed polygons'}")
    say(f"  read in {1000 * (time.perf_counter() - t0):.0f} ms")
    side_by_side([("smooth shading", mesh_traces(verts, F)),
                  ("flat shading + wireframe", mesh_traces(verts, F, wireframe=True, flat=True))],
                 f"{cls}_1_original", f"Step 1: {cls} {file} as stored ({len(verts):,} vertices)")
    wait()
    return verts, sizes, idx, F


def step2(cls, file, verts, sizes, idx, F_raw):
    title("STEP 2  Cleaning and resampling")
    t0 = time.perf_counter()
    V, F, info = clean_mesh(verts, sizes, idx)
    say("  cleaning:")
    say(f"    duplicate vertices merged:   {info['merged_duplicates']:,}")
    say(f"    unused vertices removed:     {info['unused_removed']:,}")
    say(f"    degenerate faces removed:    {info['degenerate_removed']:,}")
    say(f"    duplicated faces removed:    {info['duplicate_faces_removed']:,}")
    F, o1 = fix_orientation(V, F)
    say(f"  orientation: conflicting edges {o1['conflicts_before']:,} -> {o1['conflicts_after']:,}, "
        f"parts turned inside-out: {o1['inverted_components']}")
    cv_before = area_cv(V, F)
    n_before = len(V)
    V, F, log = resample_mesh(V, F)
    F, o2 = fix_orientation(V, F)
    say(f"  resampling (target 7,500, accepted 5,000 - 10,000): {log['action']}")
    if log["action"] == "refined":
        say(f"    refinement steps (L / 1.1 per step): {log['subdiv_iters']}")
    if log.get("decim_method"):
        say(f"    decimation method: {log['decim_method']}")
    say(f"    vertices: {n_before:,} -> {len(V):,}")
    say(f"    triangle sizes (CV): {cv_before:.2f} -> {area_cv(V, F):.2f}")
    say(f"  remaining orientation conflicts: {count_conflicts(F)}")
    say(f"  done in {time.perf_counter() - t0:.2f} s")
    side_by_side([(f"before: {len(verts):,} vertices", mesh_traces(verts, F_raw, wireframe=True)),
                  (f"after: {len(V):,} vertices", mesh_traces(V, F, wireframe=True))],
                 f"{cls}_2_resampled", f"Step 2: {cls} cleaned and resampled")
    wait()
    return V, F


def step3_normalize(cls, V, F):
    title("STEP 3.1  Normalizing")
    V0 = np.asarray(V, float)
    V1 = center(V0)
    say(f"  translation: barycenter moved by {np.linalg.norm(V0.mean(axis=0)):.4f}  -> now at the origin")
    V2 = align(V1)
    lam, _ = principal_axes(V1)
    say(f"  alignment (PCA): eigenvalues {lam[0]:.4f} >= {lam[1]:.4f} >= {lam[2]:.4f};  e1 -> x, e2 -> y, e1 x e2 -> z")
    m = moments(V2, F)
    V3, F3 = flip(V2, F)
    mirrored = [ax for ax, f in zip("xyz", m) if f < 0]
    say(f"  flipping (moment test): mirrored axes: {', '.join(mirrored) if mirrored else 'none'}"
        f"{'  (odd number -> faces reversed)' if len(mirrored) % 2 else ''}")
    V4 = scale(V3)
    size = np.ptp(V4, axis=0)
    say(f"  scaling: bounding box {size[0]:.3f} x {size[1]:.3f} x {size[2]:.3f}  (largest side = 1)")
    _, E = principal_axes(V4)
    ang = [np.degrees(np.arccos(min(1.0, abs(E[k, k])))) for k in range(3)]
    say(f"  check: angles of e1, e2, e3 with x, y, z after normalizing: "
        f"{ang[0]:.1e}, {ang[1]:.1e}, {ang[2]:.1e} degrees")
    r0 = 0.6 * np.ptp(V1, axis=0).max()
    side_by_side([("before: stored pose", mesh_traces(V0, F) + axes_traces(r0)),
                  ("after: centered, aligned, flipped, scaled", mesh_traces(V4, F3) + axes_traces(0.75))],
                 f"{cls}_3_normalized", f"Step 3.1: {cls} normalized (x red, y green, z blue)")
    wait()
    return V4, F3


def step3_describe(cls, file, V, F):
    title("STEP 3.2  Describing")
    t0 = time.perf_counter()
    row = features_of(V, F, cls, file)
    say(f"  computed in {time.perf_counter() - t0:.1f} s (1,000,000 samples per histogram, D1 from all vertices)")
    say("  global descriptors:")
    for key, label in (("surface_area", "surface area"), ("compactness", "compactness"),
                       ("sphericity", "sphericity"), ("rectangularity_3d", "rectangularity"),
                       ("diameter", "diameter"), ("convexity", "convexity"), ("eccentricity", "eccentricity")):
        say(f"    {label:<16}{row[key]:.4f}")
    say(f"    (volume {row['mesh_volume']:.5f}, obtained from: {row['volume_source']})")
    fig = make_subplots(rows=1, cols=5, subplot_titles=NAMES, horizontal_spacing=0.04)
    for c, n in enumerate(NAMES, start=1):
        lo, hi = RANGES[n]
        h = np.array([row[f"{n}_bin_{j:02d}"] for j in range(20)], float)
        x = lo + (np.arange(20) + 0.5) * (hi - lo) / 20
        fig.add_trace(go.Bar(x=x, y=h, width=(hi - lo) / 20, marker=dict(color="#B0C4DE", line=dict(color="black", width=1)),
                             showlegend=False), row=1, col=c)
    fig.update_layout(height=420, margin=dict(l=30, r=30, t=100, b=40), plot_bgcolor="white")
    open_page(fig, f"{cls}_4_histograms", f"Step 3.2: the five histograms of the {cls} (20 bins each)")
    wait()
    return pd.Series(row)


def step4(cls, file, features):
    title("STEP 4  Querying: the 10 closest database shapes")
    t0 = time.perf_counter()
    res = s4.query(features, k=K)
    say(f"  compared with all {len(s4.load_database()[0]):,} database shapes in {1000 * (time.perf_counter() - t0):.0f} ms "
        f"(weighted Euclidean distance)")
    hits = int((res["class"] == cls).sum())
    say(f"  {hits} of {K} have the class {cls}\n")
    say(res.assign(distance=res["distance"].round(3)).to_string(index=False))
    show_results(cls, file, res, f"{cls}_5_results",
                 f"Step 4: the {K} closest shapes to the {cls} query  ({hits} of {K} correct;  green = {cls}, red = other class)")
    wait()
    return res


def show_results(cls, file, res, name, page_title):
    cols = 6
    rows = 2
    titles = [f"<b>QUERY</b><br>{cls} {Path(file).stem}"] + [
        f"{r['rank']}. <span style='color:{'green' if r['class'] == cls else 'red'}'>{r['class']}</span>"
        f"<br>d = {r['distance']:.2f}" for _, r in res.iterrows()]
    titles += [""] * (rows * cols - len(titles))
    fig = make_subplots(rows=rows, cols=cols, specs=[[{"type": "scene"}] * cols] * rows, subplot_titles=titles,
                        horizontal_spacing=0.01, vertical_spacing=0.08)
    meshes = [load_result_mesh(r["class"], r["file"]) for _, r in res.iterrows()]
    qV, qF = QUERY_MESH[cls]
    meshes = [(qV, qF)] + meshes
    for i, (V, F) in enumerate(meshes):
        r, c = divmod(i, cols)
        for t in mesh_traces(V, F):
            fig.add_trace(t, row=r + 1, col=c + 1)
        fig.update_layout(**{f"scene{'' if i == 0 else i + 1}": dict(scene(0.62), camera=dict(eye=dict(x=0.9, y=-1.3, z=0.9)))})
    fig.update_layout(height=760, margin=dict(l=5, r=5, t=110, b=5), showlegend=False)
    open_page(fig, name, page_title)


def step5(cls, features, exact):
    title("STEP 5  The same query through the HNSW index")
    try:
        import faiss
    except ImportError:
        say("  FAISS is not installed, so this step is skipped (pip install faiss-cpu).")
        wait()
        return
    norm, params, scales = load_step4_data()
    X = vectorize_table(norm, scales)
    t0 = time.perf_counter()
    index = faiss.IndexHNSWFlat(X.shape[1], 16)
    index.hnsw.efConstruction = 200
    index.add(X)
    index.hnsw.efSearch = 100
    build_ms = 1000 * (time.perf_counter() - t0)
    q = query_vector(features, params, scales)[None, :].astype(np.float32)
    t0 = time.perf_counter()
    d2, lab = index.search(q, K)
    ann_ms = 1000 * (time.perf_counter() - t0)
    t0 = time.perf_counter()
    d_all = np.sqrt(((X - q) ** 2).sum(axis=1))
    exact_order = np.argsort(d_all)[:K]
    exact_ms = 1000 * (time.perf_counter() - t0)
    ann = pd.DataFrame({"rank": np.arange(1, K + 1), "class": norm["class"].iloc[lab[0]].values,
                        "file": norm["file"].iloc[lab[0]].values, "distance": np.sqrt(np.maximum(d2[0], 0)).round(3)})
    say(f"  index of {len(X):,} vectors with {X.shape[1]} values each, built in {build_ms:.0f} ms")
    say(f"  search with the index: {ann_ms:.3f} ms   |   exact search over all vectors: {exact_ms:.3f} ms\n")
    say(ann.to_string(index=False))
    same = list(ann["file"]) == list(exact["file"]) and list(ann["class"]) == list(exact["class"])
    same_exact = set(lab[0]) == set(exact_order)
    say(f"\n  same 10 shapes as Step 4: {'yes' if same else 'no'}   |   same as exact search: {'yes' if same_exact else 'no'}")
    wait()


# ------------------------------------------------------------------ main
QUERY_MESH = {}


def run(cls, file):
    verts, sizes, idx, F_raw = step1(cls, file)
    V, F = step2(cls, file, verts, sizes, idx, F_raw)
    V, F = step3_normalize(cls, V, F)
    QUERY_MESH[cls] = (V, F)
    features = step3_describe(cls, file, V, F)
    exact = step4(cls, file, features)
    step5(cls, features, exact)


def main():
    global SHOW, PAUSE
    ap = argparse.ArgumentParser(description="Demo: two outside queries through the whole retrieval pipeline")
    ap.add_argument("--only", choices=[c for c, _ in QUERIES], help="run only this query")
    ap.add_argument("--no-pause", action="store_true", help="do not wait for Enter after each step")
    ap.add_argument("--no-show", action="store_true", help="do not open browser windows")
    a = ap.parse_args()
    SHOW, PAUSE = not a.no_show, not a.no_pause
    say("Content-based 3D shape retrieval: demo with two shapes from outside the database")
    say(f"database: {len(s4.load_database()[0]):,} shapes;  queries: " + ", ".join(f"{c} {f}" for c, f in QUERIES))
    for cls, file in QUERIES:
        if a.only and cls != a.only:
            continue
        run(cls, file)
    title("End of the demo")


if __name__ == "__main__":
    main()
