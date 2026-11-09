"""
Step 3.2: global and histogram-based 3D shape descriptors.

Run from the project folder:
    python run_s3_2.py --limit 5         test on the first 5 shapes
    python run_s3_2.py                   whole database (about 17 minutes with 6 worker processes)
    python run_s3_2.py --shape Data/Bird/D00089.obj
                                         print the features of one shape (read from the feature
                                         table if the shape is in it, else computed)

Input:  processed/step3_normalized/<class>/<file>.obj
Output: report/data/step3/features_step3_2.csv
        report/data/step3/features_step3_2_summary.csv
        report/data/step3/features_step3_2_errors.csv (only when errors occur)

Volume: meshes that are not closed get their holes filled with a triangle fan around the center
of each hole (technical tips 3b); the volume is then computed with the barycenter as reference
point (Module 4). If this volume is impossible (not positive, larger than the convex hull volume or
more compact than a sphere), a rough estimate is used instead and the shape is marked as "estimate".
The distribution descriptors A3, D2, D3, D4 use random vertex tuples without repeated vertices,
with a reproducible per-shape random seed; D1 uses all vertices. Histograms store normalized
frequencies.
"""

import argparse
import hashlib
import math
import os
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import trimesh

from mr_io import PROJECT_ROOT, mesh_arrays
from mr_normalize import STEP3_DIR

OUT = PROJECT_ROOT / "report" / "data" / "step3"
FEATURES_CSV = OUT / "features_step3_2.csv"
SUMMARY_CSV = OUT / "features_step3_2_summary.csv"
ERRORS_CSV = OUT / "features_step3_2_errors.csv"

DEFAULT_SAMPLES = 1_000_000
DEFAULT_BINS = 20
EPS = 1e-12
BLOCK = 256
SAMPLE_BLOCK = 100_000
HULL_TOLERANCE = 1.01     # a volume up to 1 % above the hull volume is still accepted (rounding)
NAMES = ["A3", "D1", "D2", "D3", "D4"]
# Fixed ranges: the largest value of each descriptor over the whole normalized database,
# rounded up (see figures_step3_2.py, ranges()). Values outside are moved to the nearest end.
RANGES = {
    "A3": (0.0, 180.0),
    "D1": (0.0, 1.15),
    "D2": (0.0, 1.60),
    "D3": (0.0, 0.85),
    "D4": (0.0, 0.60),
}
GLOBAL_COLUMNS = ["surface_area", "compactness", "sphericity", "rectangularity_3d", "diameter",
                  "convexity", "eccentricity"]


def seed_for(class_name, filename):
    key = f"{class_name}/{filename}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(key).digest()[:4], "little")


def surface_area(V, F):
    T = V[F]
    cross = np.cross(T[:, 1] - T[:, 0], T[:, 2] - T[:, 0])
    return float(0.5 * np.linalg.norm(cross, axis=1).sum())


def signed_volume(V, F, o):
    """Sum of the signed volumes of the tetrahedra (o, v1, v2, v3) over all triangles."""
    T = V[F] - o
    return float(np.einsum("ij,ij->i", T[:, 0], np.cross(T[:, 1], T[:, 2])).sum() / 6.0)


def hole_loops(F):
    """Closed loops of boundary edges, in the direction of the faces they belong to."""
    half_edges = np.concatenate([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]])
    present = set(map(tuple, half_edges.tolist()))
    nxt = defaultdict(list)
    for a, b in half_edges.tolist():
        if (b, a) not in present:          # no neighbor face on the other side: boundary edge
            nxt[a].append(b)
    loops = []
    for start in list(nxt):
        while nxt[start]:
            loop, a = [start], nxt[start].pop()
            while a != start and nxt[a]:
                loop.append(a)
                a = nxt[a].pop()
            if a == start and len(loop) >= 3:
                loops.append(loop)
    return loops


def fill_holes(V, F):
    """Close every hole with a fan of triangles around the mean of its boundary vertices."""
    loops = hole_loops(F)
    if not loops:
        return V, F, 0
    new_V, new_F = [V], [F]
    k = len(V)
    for loop in loops:
        L = np.asarray(loop)
        new_V.append(V[L].mean(axis=0)[None])
        # boundary edge (a, b) of a face becomes (b, a) in the fan triangle: same orientation
        new_F.append(np.c_[np.roll(L, -1), L, np.full(len(L), k)])
        k += 1
    return np.vstack(new_V), np.vstack(new_F), len(loops)


def mesh_volume(V, F):
    """Volume of the shape, area of the surface that encloses it (with the filled holes),
    and how the volume was obtained: closed, filled or open (approximate)."""
    mesh = trimesh.Trimesh(vertices=V, faces=F, process=False)
    if mesh.is_watertight:
        return signed_volume(V, F, V.mean(axis=0)), surface_area(V, F), "closed", 0
    V2, F2, n_holes = fill_holes(V, F)
    closed = trimesh.Trimesh(vertices=V2, faces=F2, process=False).is_watertight
    return (signed_volume(V2, F2, V.mean(axis=0)), surface_area(V2, F2),
            "filled" if closed else "open", n_holes)


def exact_diameter(points):
    """Exact maximum pairwise distance among convex-hull vertices, using blocks."""
    P = np.asarray(points, dtype=np.float64)
    best_sq = 0.0
    for i in range(0, len(P), BLOCK):
        A = P[i:i + BLOCK]
        aa = np.sum(A * A, axis=1)[:, None]
        for j in range(i, len(P), BLOCK):
            B = P[j:j + BLOCK]
            d2 = aa + np.sum(B * B, axis=1)[None, :] - 2.0 * (A @ B.T)
            np.maximum(d2, 0.0, out=d2)
            best_sq = max(best_sq, float(d2.max(initial=0.0)))
    return math.sqrt(best_sq)


def global_descriptors(V, F):
    mesh = trimesh.Trimesh(vertices=V, faces=F, process=False)
    area = surface_area(V, F)
    volume, closed_area, source, n_holes = mesh_volume(V, F)
    hull = mesh.convex_hull
    hull_volume = abs(float(hull.volume))
    # After Step 3.1 the shape is aligned with its principal axes, so its oriented bounding box
    # (computed with PCA, Module 4) is the axis-aligned bounding box.
    obb_volume = float(np.prod(V.max(axis=0) - V.min(axis=0)))
    # a volume is impossible if it is larger than the hull volume, or larger than the volume of a
    # sphere with the same surface area (no shape is more compact than a sphere)
    if not (EPS < volume <= HULL_TOLERANCE * hull_volume and 36.0 * math.pi * volume**2 <= closed_area**3):
        # rough estimate instead of no value: the size of the computed volume, limited to what is possible
        # (at most the hull volume and the volume of a sphere with the same surface area)
        volume = min(max(abs(volume), EPS), hull_volume, closed_area**1.5 / math.sqrt(36.0 * math.pi))
        source = "estimate"

    out = {
        "surface_area": area,
        "compactness": np.nan,
        "sphericity": np.nan,
        "rectangularity_3d": np.nan,
        "diameter": exact_diameter(np.asarray(hull.vertices, dtype=np.float64)),
        "convexity": np.nan,
        "eccentricity": np.nan,
        "mesh_volume": volume,
        "volume_source": source,
        "holes_filled": n_holes,
        "convex_hull_volume": hull_volume,
        "obb_volume": obb_volume,
    }
    if np.isfinite(volume):
        # compactness with respect to a sphere (Module 4): 1 for a sphere, larger otherwise;
        # area and volume of the same closed surface (with the filled holes)
        out["compactness"] = closed_area**3 / (36.0 * math.pi * volume**2)
        out["sphericity"] = 1.0 / out["compactness"]
        out["rectangularity_3d"] = volume / obb_volume
        out["convexity"] = min(volume / hull_volume, 1.0)

    P = V - V.mean(axis=0)
    cov = P.T @ P / (len(V) - 1)
    eigenvalues = np.sort(np.maximum(np.linalg.eigvalsh(cov), 0.0))
    if eigenvalues[0] > EPS:
        out["eccentricity"] = float(eigenvalues[-1] / eigenvalues[0])
    return out


def random_tuples(n, k, size, rng):
    """size random k-tuples of vertex indices, without a vertex repeated inside a tuple."""
    I = rng.integers(n, size=(size, k))
    while True:
        bad = np.zeros(size, dtype=bool)
        for a in range(k):
            for b in range(a + 1, k):
                bad |= I[:, a] == I[:, b]
        if not bad.any():
            return I
        I[bad] = rng.integers(n, size=(int(bad.sum()), k))


def _measures(V, n, size, rng):
    """size samples of A3, D2, D3 and D4."""
    I = random_tuples(n, 3, size, rng)
    u, w = V[I[:, 0]] - V[I[:, 1]], V[I[:, 2]] - V[I[:, 1]]
    denom = np.linalg.norm(u, axis=1) * np.linalg.norm(w, axis=1)
    cosine = np.einsum("ij,ij->i", u, w) / np.maximum(denom, EPS)
    a3 = np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0)))   # angle at the middle vertex

    I = random_tuples(n, 2, size, rng)
    d2 = np.linalg.norm(V[I[:, 0]] - V[I[:, 1]], axis=1)

    I = random_tuples(n, 3, size, rng)
    T = V[I]
    d3 = np.sqrt(0.5 * np.linalg.norm(np.cross(T[:, 1] - T[:, 0], T[:, 2] - T[:, 0]), axis=1))

    I = random_tuples(n, 4, size, rng)
    T = V[I]
    det = np.einsum("ij,ij->i", T[:, 1] - T[:, 0], np.cross(T[:, 2] - T[:, 0], T[:, 3] - T[:, 0]))
    d4 = np.cbrt(np.abs(det) / 6.0)
    return a3, d2, d3, d4


def sample_distributions(V, n_samples, rng):
    """A3, D2, D3, D4 from n_samples random vertex tuples; D1 from all vertices.
    The samples are made in blocks, so that the memory use stays small."""
    n = len(V)
    if n < 4:
        raise ValueError("At least four vertices are needed for A3/D1-D4.")
    blocks = [_measures(V, n, min(SAMPLE_BLOCK, n_samples - k), rng) for k in range(0, n_samples, SAMPLE_BLOCK)]
    a3, d2, d3, d4 = (np.concatenate(x) for x in zip(*blocks))
    d1 = np.linalg.norm(V - V.mean(axis=0), axis=1)
    return {"A3": a3, "D1": d1, "D2": d2, "D3": d3, "D4": d4}


def make_histograms(values, bins):
    out = {}
    for name, vals in values.items():
        lo, hi = RANGES[name]
        vals = np.clip(np.asarray(vals)[np.isfinite(vals)], lo, hi)
        counts, _ = np.histogram(vals, bins=bins, range=(lo, hi))
        hist = counts / counts.sum() if counts.sum() else np.zeros(bins)
        for i, val in enumerate(hist):
            out[f"{name}_bin_{i:02d}"] = float(val)
    return out


def features_of(V, F, class_name, filename, samples=DEFAULT_SAMPLES, bins=DEFAULT_BINS):
    """All Step 3.2 features of one (normalized) mesh, as one table row."""
    V, F = np.asarray(V, dtype=np.float64), np.asarray(F, dtype=np.int64)
    if V.ndim != 2 or V.shape[1] != 3 or len(V) < 4:
        raise ValueError(f"Invalid vertex array shape: {V.shape}")
    if F.ndim != 2 or F.shape[1] != 3 or len(F) == 0:
        raise ValueError(f"Expected triangle faces, got {F.shape}")
    if F.min() < 0 or F.max() >= len(V):
        raise ValueError("Face index outside vertex array")

    row = {"class": class_name, "file": filename, "n_vertices": len(V), "n_faces": len(F)}
    row.update(global_descriptors(V, F))
    rng = np.random.default_rng(seed_for(class_name, filename))
    row.update(make_histograms(sample_distributions(V, samples, rng), bins))
    row["status"] = "ok"
    return row


def process(args):
    path, samples, bins = args
    try:
        V, F = mesh_arrays(path)
        return features_of(V, F, path.parent.name, path.name, samples, bins), None
    except Exception as exc:
        return None, {"class": path.parent.name, "file": path.name, "error": repr(exc)}


def print_features(row):
    print(f"\n{row['class']}/{row['file']}  ({row['n_vertices']} vertices, {row['n_faces']} faces)")
    print("Global descriptors:")
    for c in GLOBAL_COLUMNS + ["mesh_volume", "volume_source"]:
        v = row[c]
        print(f"  {c:20s} {v:.5g}" if isinstance(v, (float, np.floating)) else f"  {c:20s} {v}")
    for name in NAMES:
        cols = sorted(c for c in row.index if c.startswith(name + "_bin_")) if isinstance(row, pd.Series) \
            else sorted(c for c in row if c.startswith(name + "_bin_"))
        lo, hi = RANGES[name]
        print(f"{name} histogram ({len(cols)} bins, range {lo:g}..{hi:g}):")
        print("  " + " ".join(f"{row[c]:.3f}" for c in cols))


def show_shape(path, samples, bins):
    """Print the features of one shape: from the feature table if it is there, else computed."""
    path = Path(path)
    if FEATURES_CSV.exists():
        table = pd.read_csv(FEATURES_CSV)
        hit = table[(table["class"] == path.parent.name) & (table["file"] == path.name)]
        if len(hit):
            print(f"Read from {FEATURES_CSV}")
            print_features(hit.iloc[0])
            return
    V, F = mesh_arrays(path)
    ext = (V.max(axis=0) - V.min(axis=0)).max()
    if abs(ext - 1.0) > 1e-6 or np.linalg.norm(V.mean(axis=0)) > 1e-6:
        print("Note: this shape is not normalized (Step 3.1); the features are computed as it is.")
    print("Computed now (not in the feature table)")
    print_features(features_of(V, F, path.parent.name, path.name, samples, bins))


def main():
    parser = argparse.ArgumentParser(description="Step 3.2 3D shape descriptors")
    parser.add_argument("--samples", type=int, default=DEFAULT_SAMPLES,
                        help="random vertex tuples per distribution descriptor (default 1,000,000)")
    parser.add_argument("--bins", type=int, default=DEFAULT_BINS,
                        help="histogram bins per descriptor (default 20)")
    parser.add_argument("--limit", type=int, default=0,
                        help="test only the first N meshes; 0 processes all")
    parser.add_argument("--shape", help="print the features of this one mesh file and stop")
    parser.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 1))
    args = parser.parse_args()
    if args.samples < 1 or args.bins < 1:
        parser.error("--samples and --bins must be positive")
    if args.shape:
        show_shape(args.shape, args.samples, args.bins)
        return
    if not STEP3_DIR.exists():
        raise FileNotFoundError(
            f"Missing {STEP3_DIR}. Run Step 3.1 first: python run_step3_1.py"
        )

    paths = sorted(STEP3_DIR.glob("*/*.obj"))
    if args.limit > 0:
        paths = paths[:args.limit]
    if not paths:
        raise FileNotFoundError(f"No OBJ files found in {STEP3_DIR}")

    OUT.mkdir(parents=True, exist_ok=True)
    rows, errors = [], []
    start = time.time()
    print(f"Step 3.2: {len(paths)} shapes; {args.samples} samples per distribution; {args.bins} bins")
    print(f"Input: {STEP3_DIR}")

    with ProcessPoolExecutor(args.workers) as ex:
        jobs = [(p, args.samples, args.bins) for p in paths]
        for k, (row, err) in enumerate(ex.map(process, jobs, chunksize=4), start=1):
            rows.append(row) if row else errors.append(err)
            if k % 100 == 0 or k == len(paths):
                print(f"  {k}/{len(paths)} processed; success={len(rows)}, failed={len(errors)}")

    if rows:
        df = pd.DataFrame(rows).sort_values(["class", "file"]).reset_index(drop=True)
        df.to_csv(FEATURES_CSV, index=False)
        df.select_dtypes(include=[np.number]).describe().T.to_csv(SUMMARY_CSV)

    if errors:
        pd.DataFrame(errors).to_csv(ERRORS_CSV, index=False)
    elif ERRORS_CSV.exists():
        ERRORS_CSV.unlink()

    print("\nSTEP 3.2 SUMMARY")
    print(f"Input shapes: {len(paths)}")
    print(f"Successful: {len(rows)}")
    print(f"Failed: {len(errors)}")
    print(f"Elapsed: {time.time() - start:.1f} seconds")
    if rows:
        print("Volume:", ", ".join(f"{k} {v}" for k, v in df.volume_source.value_counts().items()))
        print(f"Features: {FEATURES_CSV}")
        print(f"Summary:  {SUMMARY_CSV}")
        print(f"Histogram features: 5 x {args.bins} = {5 * args.bins} columns")
    if errors:
        print(f"Errors: {ERRORS_CSV}; these shapes have no features.")


if __name__ == "__main__":
    main()
