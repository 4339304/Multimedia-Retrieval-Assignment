"""Step 2.4: check the resampling of Step 2.3.

    python src/check_resampling.py            # uses cached statistics if they exist
    python src/check_resampling.py --force    # re-read all resampled meshes (about 2 minutes)

Needs the output of Step 2.3 (python src/run_step2.py):
    report/data/step2/resample_log_full.csv   log of the resampling
    report/data/step2/shape_stats_raw.csv     statistics of the original database (Step 2.2)
    processed/step2_full/                     resampled meshes

Writes
    report/data/step2/step2_4_refined_database_statistics.csv   statistics of every resampled mesh
    report/data/step2/step2_4_vertex_summary.csv                vertex counts before / after cleaning / after resampling
    report/data/step2/step2_4_face_summary.csv                  face counts after resampling
    report/data/step2/step2_4_printout.txt                      the printed summary below
    report/figures/step2/step2_4_histograms.png                 vertex, face and face-area histograms before / after
    report/figures/step2/<name>_raw_*.png, <name>_refined_*.png viewer images before / after for three shapes
"""
import argparse
import io
import sys
from contextlib import redirect_stdout
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from mr_io import DATA_DIR, PROJECT_ROOT, mesh_arrays, render_png
from mr_stats import list_shapes, scan_database

PROCESSED = PROJECT_ROOT / "processed"
DATA_OUT = PROJECT_ROOT / "report" / "data" / "step2"
FIG = PROJECT_ROOT / "report" / "figures" / "step2"

LOWER_LIMIT = 5000
UPPER_LIMIT = 10000

# viewer check: one shape that was refined, one that was decimated and one that was kept
EXAMPLES = {"Bird": ("Bird", "D00089.obj"),            # refined (233 vertices before)
            "Humanoid": ("Humanoid", "m253.obj"),      # decimated (11,806 vertices before)
            "Kept": None}                              # first shape that was kept unchanged


def check(force=False):
    resample_log_path = DATA_OUT / "resample_log_full.csv"
    refined_dir = PROCESSED / "step2_full"
    if not resample_log_path.exists():
        raise FileNotFoundError("The resampling log is missing. Please run Step 2.3 first (python src/run_step2.py).")
    if not refined_dir.exists():
        raise FileNotFoundError("The processed mesh directory is missing. Please run Step 2.3 first.")

    full = pd.read_csv(resample_log_path)
    refined_files = sorted(refined_dir.rglob("*.obj"))

    print("STEP 2.4: RESAMPLING CHECK")
    print("-" * 50)
    print(f"Shapes in processing log: {len(full)}")
    print(f"Refined OBJ files found:  {len(refined_files)}")

    in_log = set(zip(full["class"], full["file"]))
    on_disk = {(p.parent.name, p.name) for p in refined_files}
    print(f"Shapes in the log without a file: {len(in_log - on_disk)}")
    print(f"Files that are not in the log:    {len(on_disk - in_log)}")

    print("\nProcessing actions:")
    print(full["action"].value_counts().to_frame("Number of shapes").to_string())
    print("\nProcessing status:")
    print(full["status"].value_counts(dropna=False).to_frame("Number of shapes").to_string())

    # --- statistics of the resampled meshes, computed from the files themselves with the Step 2.2 tool
    refined_stats = scan_database(list_shapes(refined_dir), cache_csv=DATA_OUT / "step2_4_refined_database_statistics.csv",
                                  force=force, verbose=False)
    refined_stats["path"] = [str(refined_dir / c / f) for c, f in zip(refined_stats["class"], refined_stats["file"])]
    raw_stats = pd.read_csv(DATA_OUT / "shape_stats_raw.csv")

    merged = full.merge(refined_stats[["class", "file", "n_vertices", "n_faces"]], on=["class", "file"], how="left")
    same = (merged["v_out"] == merged["n_vertices"]) & (merged["f_out"] == merged["n_faces"])
    print(f"\nVertex and face counts in the files equal to the log: {same.sum()} / {len(full)}")

    vertex_summary = pd.DataFrame({"original": raw_stats["n_vertices"].describe(),
                                   "after cleaning": full["v_clean"].describe(),
                                   "after resampling": refined_stats["n_vertices"].describe()}).round(2)
    print("\nVertex count statistics:")
    print(vertex_summary.to_string())

    inside_range = refined_stats["n_vertices"].between(LOWER_LIMIT, UPPER_LIMIT)
    print(f"\nTarget-range check ({LOWER_LIMIT:,}-{UPPER_LIMIT:,} vertices):")
    print(f"Shapes within range:  {inside_range.sum()} / {len(refined_stats)}")
    print(f"Shapes outside range: {(~inside_range).sum()}")

    face_summary = pd.DataFrame({"original": raw_stats["n_faces"].describe(),
                                 "after resampling": refined_stats["n_faces"].describe()}).round(2)
    print("\nFace count statistics:")
    print(face_summary.to_string())

    # uniformity of the triangle sizes: coefficient of variation of the face areas (0 = all equal)
    cv = full[["area_cv_before", "area_cv_after"]].describe().round(3)
    print("\nFace-area coefficient of variation (per shape):")
    print(cv.to_string())

    print("\nMesh quality after resampling:")
    print(f"Shapes without orientation conflicts: {(refined_stats['n_conflicting_edges'] == 0).sum()} / {len(refined_stats)}")
    print(f"Shapes without degenerate faces:      {(refined_stats['n_degenerate_faces'] == 0).sum()} / {len(refined_stats)}")
    print(f"Shapes without duplicate vertices:    {(refined_stats['n_duplicate_vertices'] == 0).sum()} / {len(refined_stats)}")

    vertex_summary.to_csv(DATA_OUT / "step2_4_vertex_summary.csv")
    face_summary.to_csv(DATA_OUT / "step2_4_face_summary.csv")
    return full, raw_stats, refined_stats


def histograms(full, raw_stats, refined_stats):
    fig, ax = plt.subplots(1, 3, figsize=(13, 3.2))
    bins = np.logspace(1, 5.2, 50)
    for a, col, label in ((ax[0], "n_vertices", "number of vertices"), (ax[1], "n_faces", "number of faces")):
        a.hist(raw_stats[col], bins=bins, histtype="step", color="#555555", linewidth=1.4, label="original")
        a.hist(refined_stats[col], bins=bins, color="#1f77b4", label="resampled")
        a.set_xscale("log")
        a.set_xlabel(label, fontsize=9)
    ax[0].axvspan(LOWER_LIMIT, UPPER_LIMIT, color="#2ca02c", alpha=0.12, label="target range")
    ax[0].set_ylabel("number of shapes", fontsize=9)
    ax[0].legend(fontsize=8)
    cbins = np.linspace(0, 6, 61)
    ax[2].hist(full["area_cv_before"].clip(upper=6), bins=cbins, histtype="step", color="#555555", linewidth=1.4,
               label="original")
    ax[2].hist(full["area_cv_after"].clip(upper=6), bins=cbins, color="#1f77b4", label="resampled")
    ax[2].set_xlabel("coefficient of variation of the face areas", fontsize=9)
    ax[2].set_xticks([0, 1, 2, 3, 4, 5, 6], ["0", "1", "2", "3", "4", "5", r"$\geq$6"])
    ax[2].legend(fontsize=8)
    for a, t in zip(ax, ("(a) vertices", "(b) faces", "(c) uniformity of the face sizes")):
        a.set_title(t, fontsize=10)
        a.tick_params(labelsize=8)
    fig.tight_layout()
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / "step2_4_histograms.png", dpi=200)
    plt.close(fig)


def viewer_images(full):
    kept = full[full["action"] == "kept"].iloc[0]
    examples = dict(EXAMPLES, Kept=(kept["class"], kept["file"]))
    print("\nViewer check (images in report/figures/step2/):")
    for name, (cls, f) in examples.items():
        row = full[(full["class"] == cls) & (full["file"] == f)].iloc[0]
        for stage, path in (("raw", DATA_DIR / cls / f), ("refined", PROCESSED / "step2_full" / cls / f)):
            V, F = mesh_arrays(path)
            for kind, wire in (("shaded", False), ("wireframe", True)):
                png = FIG / f"{name}_{stage}_{kind}.png"
                render_png(V, F, png, wireframe=wire, title=None)
                crop_white(png)
        print(f"  {name:8s} = {cls} {f}: {row['action']}, {int(row['v_raw']):,} -> {int(row['v_out']):,} vertices")


def crop_white(png, pad=12):
    """Cut the white border around a rendered image."""
    img = plt.imread(png)
    ys, xs = (img[..., :3].min(axis=2) < 0.97).nonzero()
    img = img[max(ys.min() - pad, 0):ys.max() + pad, max(xs.min() - pad, 0):xs.max() + pad]
    plt.imsave(png, img)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true", help="re-read all resampled meshes")
    args = ap.parse_args()

    buf = io.StringIO()
    with redirect_stdout(buf):
        full, raw_stats, refined_stats = check(force=args.force)
        histograms(full, raw_stats, refined_stats)
        viewer_images(full)
    text = buf.getvalue()
    print(text)
    (DATA_OUT / "step2_4_printout.txt").write_text(text, encoding="utf-8")
    print("Statistics saved in report/data/step2/, figures in report/figures/step2/.")


if __name__ == "__main__":
    main()
