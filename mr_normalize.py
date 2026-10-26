"""Step 3.1: full (four-step) normalization, continuing from Step 2.5.

The four steps of Module 4, in the order of the technical tips:

    Step 2.5 (normalize_shapes.py)       translation  ->  scaling
    Step 3.1 (this file)                 alignment    ->  flipping  ->  scaling again

Step 3.1 reads the output of Step 2.5 (processed/step2_normalized), which is already centered at the
vertex barycenter and scaled to a largest bounding-box side of 1. It adds the alignment and the
flipping. The scaling is repeated at the end, because the rotation changes the bounding box.

    center(V)    -> V          vertex barycenter to the origin    (same formula as Step 2.5)
    align(V)     -> V          major/medium eigenvector onto x/y  (Step 3.1)
    flip(V, F)   -> (V, F)     most of the mass on the positive side of each axis (Step 3.1)
    scale(V)     -> V          largest bounding-box side becomes 1 (same formula as Step 2.5)

center() and scale() use exactly the formulas of Step 2.5, so normalize() gives the same result for a
shape that did not go through Step 2.5 (e.g. a query shape). All functions take NumPy arrays
(V: (n, 3) float, F: (m, 3) int) and return new arrays.
"""
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from mr_io import PROJECT_ROOT, mesh_arrays, write_obj

STEP2_DIR = PROJECT_ROOT / "processed" / "step2_full"             # resampled shapes (Step 2.3)
STEP2_NORM_DIR = PROJECT_ROOT / "processed" / "step2_normalized"  # centered and scaled (Step 2.5)
STEP3_DIR = PROJECT_ROOT / "processed" / "step3_normalized"       # fully normalized (Step 3.1)


# ------------------------------------------------------------------------- basic quantities
def triangle_centers(V, F):
    return V[F].mean(axis=1)


def barycenter(V):
    """Barycenter as in Step 2.5 and Module 4: the mean of the vertex positions."""
    return V.mean(axis=0)


def principal_axes(V):
    """Eigenvalues (descending) and unit eigenvectors (columns) of the vertex covariance matrix
    (Module 4): Sigma = 1/(N_V - 1) sum_i (v_i - m)(v_i - m)^T, with m the mean of the vertices."""
    P = V - V.mean(axis=0)
    cov = P.T @ P / (len(V) - 1)
    lam, E = np.linalg.eigh(cov)            # ascending order
    order = np.argsort(lam)[::-1]
    return lam[order], E[:, order]


def moments(V, F):
    """Moment test (Module 4, slide 21): f_k = sum_t sign(C_t,k) C_t,k^2, k = x, y, z."""
    C = triangle_centers(V, F)
    return (np.sign(C) * C ** 2).sum(axis=0)


def signed_volume(V, F):
    return float(np.einsum("ij,ij->i", V[F[:, 0]], np.cross(V[F[:, 1]], V[F[:, 2]])).sum() / 6.0)


# ------------------------------------------------------------------------- the four steps
def center(V):
    """Translation (Step 2.5): v_i <- v_i - m, with m the mean of the vertices."""
    return V - barycenter(V)


def align(V):
    """Alignment of a centered shape: project onto e1, e2 and e1 x e2.

    x_i = v_i . e1,  y_i = v_i . e2,  z_i = v_i . (e1 x e2).
    Using e1 x e2 instead of e3 makes the new frame right-handed, so this is a pure rotation
    (determinant +1) and the face orientation is preserved. Because the shape is centered at the
    vertex mean, the projection is done around the same point as the covariance matrix."""
    _, E = principal_axes(V)
    e1, e2 = E[:, 0], E[:, 1]
    R = np.stack([e1, e2, np.cross(e1, e2)], axis=1)
    return V @ R


def flip(V, F):
    """Moment test: mirror axis k if f_k < 0 (f_k = 0 is treated as positive).

    A mirror with an odd number of flipped axes has determinant -1 and would turn every face
    inside-out, so the face winding is reversed in that case to keep the normals outwards."""
    s = np.where(moments(V, F) < 0, -1.0, 1.0)
    V2 = V * s
    F2 = F[:, ::-1].copy() if np.prod(s) < 0 else F.copy()
    return V2, F2


def scale(V):
    """Uniform scaling (Step 2.5): the largest side of the axis-aligned bounding box becomes 1."""
    return V / np.ptp(V, axis=0).max()


def normalize(V, F):
    """All four steps. For a shape from Step 2.5, center() changes nothing (already centered)."""
    V = np.asarray(V, dtype=np.float64)
    F = np.asarray(F, dtype=np.int64)
    V = center(V)
    V = align(V)
    V, F = flip(V, F)
    V = scale(V)
    return V, F


# ------------------------------------------------------------------------- verification
def measures(V, F):
    """The quantities each normalization step is supposed to fix (plus two ambiguity indicators).

    bary_dist   |m|: distance of the vertex barycenter from the origin    -> 0 after translation
    angle_e1    angle between the major eigenvector and the x axis (deg)  -> 0 after alignment
    angle_e2    angle between the medium eigenvector and the y axis (deg) -> 0 after alignment
    angle_e3    angle between the minor eigenvector and the z axis (deg)  -> 0 after alignment
    f_x/f_y/f_z moment test values (around the origin)                     -> >= 0 after flipping
    d_max       largest side of the axis-aligned bounding box             -> 1 after scaling
    ratio_12, ratio_23   lambda1/lambda2, lambda2/lambda3: close to 1 means the axes are ambiguous
    """
    V = np.asarray(V, dtype=np.float64)
    F = np.asarray(F, dtype=np.int64)
    lam, E = principal_axes(V)
    f = moments(V, F)
    eps = 1e-300
    return {
        "bary_dist": float(np.linalg.norm(barycenter(V))),
        "angle_e1": float(np.degrees(np.arccos(np.clip(abs(E[0, 0]), 0, 1)))),
        "angle_e2": float(np.degrees(np.arccos(np.clip(abs(E[1, 1]), 0, 1)))),
        "angle_e3": float(np.degrees(np.arccos(np.clip(abs(E[2, 2]), 0, 1)))),
        "f_x": float(f[0]), "f_y": float(f[1]), "f_z": float(f[2]),
        "f_rel_min": float((f / max(np.abs(f).sum(), eps)).min()),
        "d_max": float(np.ptp(V, axis=0).max()),
        "lambda1": float(lam[0]), "lambda2": float(lam[1]), "lambda3": float(lam[2]),
        "ratio_12": float(lam[0] / max(lam[1], eps)),
        "ratio_23": float(lam[1] / max(lam[2], eps)),
        "signed_volume": signed_volume(V, F),
    }


def face_normals(V, F):
    return np.cross(V[F[:, 1]] - V[F[:, 0]], V[F[:, 2]] - V[F[:, 0]])


def normal_check(V, F, V2, F2):
    """Check that no face was turned inside-out, independently of the origin.

    The whole normalization (Step 2.5 and 3.1) maps every vertex by v' = (v - m) L with a linear map
    L (rotation, mirror and uniform scale). A face normal n then has to point along n L afterwards
    (for a rotation times a uniform scale, L^{-T} is a positive multiple of L). L is recovered by
    least squares from the two vertex sets, so the check does not trust the normalization code.
    Slivers with less than 1e-4 of the median triangle area are skipped: their corners are stored
    with 7 digits, so the direction of their normal is rounding noise."""
    L = np.linalg.lstsq(V - barycenter(V), V2, rcond=None)[0]
    n1, n2 = face_normals(V, F), face_normals(V2, F2)
    expected = n1 @ L
    a1, a2 = np.linalg.norm(n1, axis=1), np.linalg.norm(n2, axis=1)
    valid = (a1 > 1e-4 * np.median(a1)) & (a2 > 1e-4 * np.median(a2))
    dots = np.einsum("ij,ij->i", n2[valid], expected[valid])
    return {"faces_checked": int(valid.sum()), "faces_skipped": int((~valid).sum()),
            "faces_turned": int((dots < 0).sum()), "det_L": float(np.linalg.det(L))}


def process_file(args):
    """Worker: Step 2.5 OBJ -> fully normalized OBJ. Never raises; failures are logged.

    'before' = the resampled shape of Step 2.3 (start of the four-step normalization),
    'input'  = the output of Step 2.5 that this step reads (handover check),
    'after'  = the result of Step 3.1 as written to disk."""
    in_path, raw_path, out_path = args
    t0 = time.time()
    p = Path(in_path)
    rec = {"class": p.parent.name, "file": p.name}
    try:
        V0, F0 = mesh_arrays(raw_path)
        rec.update({f"{k}_before": v for k, v in measures(V0, F0).items()})
        V, F = mesh_arrays(p)
        if len(V) != len(V0) or not np.array_equal(F, F0):
            raise ValueError("Step 2.5 output does not have the same mesh as the Step 2.3 output")
        rec["input_bary_dist"] = float(np.linalg.norm(barycenter(V)))
        rec["input_d_max"] = float(np.ptp(V, axis=0).max())
        V2, F2 = normalize(V, F)
        if out_path is not None:
            write_obj(out_path, V2, F2, digits=10)  # enough digits that no two vertices coincide
            V2, F2 = mesh_arrays(out_path)          # verify what was actually written
        rec.update({f"{k}_after": v for k, v in measures(V2, F2).items()})
        rec.update(normal_check(V0, F0, V2, F2))
        rec["status"] = "ok"
    except Exception as exc:
        rec["status"] = "failed"
        rec["error"] = repr(exc)
    rec["seconds"] = round(time.time() - t0, 3)
    return rec


def run_normalization(in_dir=STEP2_NORM_DIR, out_dir=STEP3_DIR, raw_dir=STEP2_DIR, workers=4, cache_csv=None,
                      force=False, verbose=True):
    """Normalize every shape of in_dir (class sub-folders, output of Step 2.5) into out_dir."""
    if cache_csv is not None and Path(cache_csv).exists() and not force:
        return pd.read_csv(cache_csv)
    if not Path(in_dir).exists():
        raise FileNotFoundError(f"{in_dir} not found. Run Step 2.5 first (python src/normalize_shapes.py).")
    paths = sorted(Path(in_dir).glob("*/*.obj"))
    jobs = [(str(p), str(Path(raw_dir) / p.parent.name / p.name),
             None if out_dir is None else str(Path(out_dir) / p.parent.name / p.name)) for p in paths]
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=workers) as ex:
        results = list(ex.map(process_file, jobs, chunksize=8))
    df = pd.DataFrame(results).sort_values(["class", "file"]).reset_index(drop=True)
    if verbose:
        print(f"normalized {len(df)} shapes in {time.time() - t0:.0f}s")
    if cache_csv is not None:
        Path(cache_csv).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(cache_csv, index=False)
    return df
