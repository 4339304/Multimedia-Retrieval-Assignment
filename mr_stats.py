"""Step 2.1 (single-shape analysis) and 2.2 (database statistics)."""
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from mr_io import DATA_DIR, read_obj_raw, triangulate
from mr_resample import fix_orientation


def list_shapes(data_dir=DATA_DIR) -> pd.DataFrame:
    rows = []
    for class_dir in sorted(Path(data_dir).iterdir()):
        if class_dir.is_dir():
            for p in sorted(class_dir.glob("*.obj")):
                rows.append({"class": class_dir.name, "file": p.name, "path": str(p)})
    return pd.DataFrame(rows)


def analyze_obj(path) -> dict:
    """2.1: everything we want to know about one shape, from the *raw* OBJ file.

    Topology checks are done on a copy where duplicate vertices (identical coordinates) are
    merged, because edges can only be paired between faces that share vertex indices.
    """
    path = Path(path)
    rec = {"class": path.parent.name, "file": path.name}
    try:
        verts, sizes, idx = read_obj_raw(path)
    except Exception as exc:  # pragma: no cover - reported in the table
        rec["error"] = repr(exc)
        return rec

    nv, nf = len(verts), len(sizes)
    rec["n_vertices"], rec["n_faces"] = nv, nf
    rec["n_tri"] = int((sizes == 3).sum())
    rec["n_quad"] = int((sizes == 4).sum())
    rec["n_other"] = nf - rec["n_tri"] - rec["n_quad"]
    kinds = np.unique(sizes)
    if nf == 0:
        rec["face_type"] = "none"
    elif len(kinds) == 1:
        rec["face_type"] = {3: "triangles", 4: "quads"}.get(int(kinds[0]), f"{kinds[0]}-gons")
    else:
        rec["face_type"] = "mixed"

    if nv == 0:
        return rec
    mn, mx = verts.min(axis=0), verts.max(axis=0)
    for k, ax in enumerate("xyz"):
        rec[f"bbox_min_{ax}"], rec[f"bbox_max_{ax}"] = mn[k], mx[k]
    diag = float(np.linalg.norm(mx - mn))
    rec["bbox_diag"] = diag
    if nf == 0:
        return rec

    # --- vertex usage / duplicates -------------------------------------------------------
    rec["n_unused_vertices"] = nv - int(np.unique(idx).size)
    uv, inv = np.unique(verts, axis=0, return_inverse=True)
    inv = inv.reshape(-1)
    fm = inv[idx]                                  # merged vertex index of every face corner
    rec["n_duplicate_vertices"] = int(np.unique(idx).size - np.unique(fm).size)

    # --- degenerate triangles (after fan triangulation) -----------------------------------
    T = triangulate(sizes, fm)
    rep = (T[:, 0] == T[:, 1]) | (T[:, 1] == T[:, 2]) | (T[:, 2] == T[:, 0])
    a, b, c = uv[T[:, 0]], uv[T[:, 1]], uv[T[:, 2]]
    cr = np.cross(b - a, c - a)
    area = 0.5 * np.linalg.norm(cr, axis=1)
    rec["n_degenerate_faces"] = int((rep | (area <= 1e-12 * max(diag, 1e-30) ** 2)).sum())

    # --- edges: boundary / non-manifold / winding consistency (on polygon edges) ----------
    starts = np.cumsum(sizes) - sizes
    nxt = np.roll(fm, -1)
    nxt[starts + sizes - 1] = fm[starts]
    keep = fm != nxt
    ea, eb = fm[keep].astype(np.int64), nxt[keep].astype(np.int64)
    M = uv.shape[0] + 1
    ukeys, uinv, ucnt = np.unique(np.minimum(ea, eb) * M + np.maximum(ea, eb),
                                  return_inverse=True, return_counts=True)
    uinv = uinv.reshape(-1)
    rec["n_edges"] = int(ukeys.size)
    rec["n_boundary_edges"] = int((ucnt == 1).sum())
    rec["n_nonmanifold_edges"] = int((ucnt > 2).sum())
    # orientation conflict = an edge shared by exactly two faces that list it in the SAME direction
    fwd = np.bincount(uinv, weights=(ea < eb).astype(np.float64), minlength=ukeys.size)
    rec["n_conflicting_edges"] = int(((ucnt == 2) & (fwd != 1)).sum())
    rec["winding_consistent"] = rec["n_conflicting_edges"] == 0
    rec["closed"] = rec["n_boundary_edges"] == 0 and rec["n_nonmanifold_edges"] == 0
    rec["euler_characteristic"] = int(np.unique(fm).size - ukeys.size + nf)
    rec["signed_volume"] = float(np.einsum("ij,ij->i", a, np.cross(b, c)).sum() / 6.0)

    # --- connected components and components that are inside-out (invisible to the edge test,
    #     because two components with opposite orientation share no edges) -----------------
    _, oinfo = fix_orientation(uv, T[~(rep | (area <= 1e-12 * max(diag, 1e-30) ** 2))])
    rec["n_components"] = oinfo["components"]
    rec["n_inverted_components"] = oinfo["inverted_components"]
    rec["n_open_components"] = oinfo["open_components"]
    return rec


def scan_database(shapes: pd.DataFrame, workers=4, cache_csv=None, force=False, verbose=True):
    """Run analyze_obj over all shapes in parallel (largest files first). Cached as CSV."""
    if cache_csv is not None and Path(cache_csv).exists() and not force:
        return pd.read_csv(cache_csv)
    paths = shapes["path"].tolist()
    sizes = [Path(p).stat().st_size for p in paths]
    order = np.argsort(sizes)[::-1]
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=workers) as ex:
        results = list(ex.map(analyze_obj, [paths[i] for i in order], chunksize=4))
    df = pd.DataFrame(results).sort_values(["class", "file"]).reset_index(drop=True)
    df.insert(2, "path", [str(DATA_DIR / r["class"] / r["file"]) for r in df.to_dict("records")])
    if verbose:
        print(f"scanned {len(df)} shapes in {time.time() - t0:.0f}s")
    if cache_csv is not None:
        Path(cache_csv).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(cache_csv, index=False)
    return df


def tukey_outliers(values, k=1.5):
    """Tukey fences on log10(values): returns boolean masks (low, high)."""
    lv = np.log10(np.asarray(values, dtype=float))
    q1, q3 = np.percentile(lv, [25, 75])
    lo, hi = q1 - k * (q3 - q1), q3 + k * (q3 - q1)
    return lv < lo, lv > hi, (10 ** lo, 10 ** hi)


def select_reduced(df, per_class=3, seed=42, n_stress=5):
    """Reduced database for developing the pipeline: `per_class` random shapes from every class
    (balanced) + the `n_stress` smallest and largest shapes overall as flagged stress tests."""
    rng = np.random.default_rng(seed)
    parts = []
    for cls, g in df.groupby("class"):
        take = min(per_class, len(g))
        parts.append(g.iloc[rng.choice(len(g), size=take, replace=False)].assign(group="balanced"))
    red = pd.concat(parts)
    extreme = pd.concat([df.nsmallest(n_stress, "n_vertices"), df.nlargest(n_stress, "n_vertices")])
    extreme = extreme[~extreme["path"].isin(red["path"])].assign(group="stress")
    return pd.concat([red, extreme]).sort_values(["class", "file"]).reset_index(drop=True)
