"""
Step 3.2: global and histogram-based 3D shape descriptors.

Run from Multimedia-Retrieval-Assignment:
    python .\run_step3_2.py --limit 5
    python .\run_step3_2.py

Input:  processed/step3_normalized/<class>/<file>.obj
Output: report/data/step3/features_step3_2.csv
        report/data/step3/features_step3_2_summary.csv
        report/data/step3/features_step3_2_errors.csv (only when errors occur)

The five distribution descriptors are sampled from vertices with a reproducible
per-shape random seed. Histogram bins store normalized frequencies.
Volume-based descriptors are NaN for non-watertight meshes.
"""

import argparse
import hashlib
import math
import time
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

DEFAULT_SAMPLES = 10000
DEFAULT_BINS = 32
EPS = 1e-12
BLOCK = 256
RANGES = {
    "A3": (0.0, 180.0),
    "D1": (0.0, math.sqrt(3.0)),
    "D2": (0.0, math.sqrt(3.0)),
    "D3": (0.0, math.sqrt(math.sqrt(3.0) / 2.0)),
    "D4": (0.0, (1.0 / 3.0) ** (1.0 / 3.0)),
}


def seed_for(class_name, filename):
    key = f"{class_name}/{filename}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(key).digest()[:4], "little")


def surface_area(V, F):
    T = V[F]
    cross = np.cross(T[:, 1] - T[:, 0], T[:, 2] - T[:, 0])
    return float(0.5 * np.linalg.norm(cross, axis=1).sum())


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
    volume = abs(float(mesh.volume)) if mesh.is_watertight else np.nan
    out = {
        "surface_area": area,
        "compactness_sphere": np.nan,
        "rectangularity_3d": np.nan,
        "diameter": np.nan,
        "convexity": np.nan,
        "eccentricity": np.nan,
        "mesh_watertight": bool(mesh.is_watertight),
        "mesh_volume": volume,
        "convex_hull_volume": np.nan,
        "obb_volume": np.nan,
    }

    # Spherical compactness: 36*pi*volume^2 / area^3; a perfect sphere is 1.
    if area > EPS and np.isfinite(volume):
        out["compactness_sphere"] = 36.0 * math.pi * volume**2 / area**3

    try:
        obb_volume = abs(float(mesh.bounding_box_oriented.volume))
        out["obb_volume"] = obb_volume
        if obb_volume > EPS and np.isfinite(volume):
            out["rectangularity_3d"] = volume / obb_volume
    except Exception:
        pass

    try:
        hull = mesh.convex_hull
        hull_volume = abs(float(hull.volume))
        out["convex_hull_volume"] = hull_volume
        if hull_volume > EPS and np.isfinite(volume):
            out["convexity"] = volume / hull_volume
        # A set's maximum distance is attained at vertices of its convex hull.
        out["diameter"] = exact_diameter(np.asarray(hull.vertices, dtype=np.float64))
    except Exception:
        pass

    if len(V) > 1:
        P = V - V.mean(axis=0)
        cov = P.T @ P / (len(V) - 1)
        eigenvalues = np.sort(np.maximum(np.linalg.eigvalsh(cov), 0.0))
        if eigenvalues[0] > EPS:
            out["eccentricity"] = float(eigenvalues[-1] / eigenvalues[0])
    return out


def sample_distributions(V, n_samples, rng):
    """Estimate A3, D1, D2, D3, D4 from random vertices."""
    n = len(V)
    if n < 4:
        raise ValueError("At least four vertices are needed for A3/D1-D4.")

    bary = V.mean(axis=0)
    d1 = np.linalg.norm(V[rng.integers(n, size=n_samples)] - bary, axis=1)

    i, j = rng.integers(n, size=n_samples), rng.integers(n, size=n_samples)
    d2 = np.linalg.norm(V[i] - V[j], axis=1)

    ia, ib, ic = (rng.integers(n, size=n_samples) for _ in range(3))
    u, v = V[ib] - V[ia], V[ic] - V[ia]
    denom = np.linalg.norm(u, axis=1) * np.linalg.norm(v, axis=1)
    cosine = np.zeros(n_samples)
    valid = denom > EPS
    cosine[valid] = np.einsum("ij,ij->i", u[valid], v[valid]) / denom[valid]
    a3 = np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0)))
    a3[~valid] = 0.0

    tri = V[rng.integers(n, size=(n_samples, 3))]
    area = 0.5 * np.linalg.norm(
        np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1
    )
    d3 = np.sqrt(area)

    tet = V[rng.integers(n, size=(n_samples, 4))]
    det = np.einsum(
        "ij,ij->i",
        tet[:, 1] - tet[:, 0],
        np.cross(tet[:, 2] - tet[:, 0], tet[:, 3] - tet[:, 0]),
    )
    d4 = np.cbrt(np.abs(det) / 6.0)
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


def process(path, samples, bins):
    V, F = mesh_arrays(path)
    V, F = np.asarray(V, dtype=np.float64), np.asarray(F, dtype=np.int64)
    if V.ndim != 2 or V.shape[1] != 3 or len(V) < 4:
        raise ValueError(f"Invalid vertex array shape: {V.shape}")
    if F.ndim != 2 or F.shape[1] != 3 or len(F) == 0:
        raise ValueError(f"Expected triangle faces, got {F.shape}")
    if F.min() < 0 or F.max() >= len(V):
        raise ValueError("Face index outside vertex array")

    row = {
        "class": path.parent.name,
        "file": path.name,
        "n_vertices": len(V),
        "n_faces": len(F),
    }
    row.update(global_descriptors(V, F))
    rng = np.random.default_rng(seed_for(path.parent.name, path.name))
    row.update(make_histograms(sample_distributions(V, samples, rng), bins))
    row["status"] = "ok"
    return row


def main():
    parser = argparse.ArgumentParser(description="Step 3.2 3D shape descriptors")
    parser.add_argument("--samples", type=int, default=DEFAULT_SAMPLES,
                        help="samples per distribution descriptor (default 10000)")
    parser.add_argument("--bins", type=int, default=DEFAULT_BINS,
                        help="histogram bins per descriptor (default 32)")
    parser.add_argument("--limit", type=int, default=0,
                        help="test only the first N meshes; 0 processes all")
    args = parser.parse_args()
    if args.samples < 1 or args.bins < 1:
        parser.error("--samples and --bins must be positive")
    if not STEP3_DIR.exists():
        raise FileNotFoundError(
            f"Missing {STEP3_DIR}. Run Step 3.1 first: python .\\run_step3_1.py"
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

    for k, path in enumerate(paths, start=1):
        try:
            rows.append(process(path, args.samples, args.bins))
        except Exception as exc:
            errors.append({"class": path.parent.name, "file": path.name, "error": repr(exc)})
        if k % 25 == 0 or k == len(paths):
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
        print(f"Features: {FEATURES_CSV}")
        print(f"Summary:  {SUMMARY_CSV}")
        print(f"Histogram features: 5 x {args.bins} = {5 * args.bins} columns")
    if errors:
        print(f"Errors: {ERRORS_CSV}; review before claiming all shapes succeeded.")


if __name__ == "__main__":
    main()
