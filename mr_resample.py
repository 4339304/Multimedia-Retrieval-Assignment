"""Step 2.3: cleaning, orientation repair and resampling to a target vertex count.

Pipeline per shape (draft, parameters still to be agreed):
    raw OBJ -> clean_mesh -> fix_orientation -> resample_mesh -> OBJ
"""
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import trimesh

from mr_io import read_obj_raw, triangulate, write_obj

V_TARGET = 7500
BAND = (5000, 10000)
TOL = 0.10          # accept results within +-10% of V_TARGET
MAX_SUBDIV = 8
MAX_OPEN_FRACTION = 0.01   # a component counts as "closed" if <=1% of its faces touch a bad edge
MAX_DEVIATION = 0.05       # decimation may move the surface by at most 5% of the bbox diagonal (Hausdorff)


# ----------------------------------------------------------------------------- cleaning
def clean_mesh(verts, sizes, idx):
    """Merge duplicate vertices, triangulate, drop degenerate / duplicate faces and unused vertices.

    Returns V (n,3), F (m,3) and a dict with what was removed.
    """
    info = {"v_raw": len(verts), "f_raw": len(sizes)}
    used_raw = int(np.unique(idx).size)
    uv, inv = np.unique(verts, axis=0, return_inverse=True)
    inv = inv.reshape(-1)
    fm = inv[idx]
    info["merged_duplicates"] = used_raw - int(np.unique(fm).size)
    info["unused_removed"] = len(verts) - used_raw

    T = triangulate(sizes, fm)
    info["triangles_after_triangulation"] = len(T)
    mn, mx = uv.min(axis=0), uv.max(axis=0)
    diag = float(np.linalg.norm(mx - mn))
    a, b, c = uv[T[:, 0]], uv[T[:, 1]], uv[T[:, 2]]
    rep = (T[:, 0] == T[:, 1]) | (T[:, 1] == T[:, 2]) | (T[:, 2] == T[:, 0])
    area = 0.5 * np.linalg.norm(np.cross(b - a, c - a), axis=1)
    bad = rep | (area <= 1e-12 * max(diag, 1e-30) ** 2)
    info["degenerate_removed"] = int(bad.sum())
    T = T[~bad]

    _, first = np.unique(np.sort(T, axis=1), axis=0, return_index=True)
    first.sort()
    info["duplicate_faces_removed"] = len(T) - len(first)
    T = T[first]

    used = np.unique(T)
    remap = np.full(len(uv), -1, dtype=np.int64)
    remap[used] = np.arange(len(used))
    V, F = uv[used], remap[T]
    info["v_clean"], info["f_clean"] = len(V), len(F)
    return V, F, info


# ------------------------------------------------------------------------ orientation
def _directed_edges(F):
    return np.stack([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]], axis=1).reshape(-1, 2).astype(np.int64)


def count_conflicts(F):
    """Edges shared by exactly two faces that list the edge in the same direction.

    0 means consistently oriented (edges shared by 3+ faces are non-manifold and not counted)."""
    if len(F) == 0:
        return 0
    e = _directed_edges(F)
    n = int(F.max()) + 1
    key = np.minimum(e[:, 0], e[:, 1]) * n + np.maximum(e[:, 0], e[:, 1])
    ukeys, uinv, ucnt = np.unique(key, return_inverse=True, return_counts=True)
    fwd = np.bincount(uinv.reshape(-1), weights=(e[:, 0] < e[:, 1]).astype(np.float64),
                      minlength=ukeys.size)
    return int(((ucnt == 2) & (fwd != 1)).sum())


def fix_orientation(V, F):
    """Make triangle winding consistent, then point closed components outwards.

    Step 1 (consistency): faces sharing an edge must list it in opposite directions. Propagate a
    flip flag through the face-adjacency graph: flip[v] = flip[u] XOR (edge listed in the same
    direction in u and v). Only edges shared by exactly two faces are followed.
    Step 2 (global): a closed component with negative signed volume is turned inside-out.
    """
    F = np.asarray(F, dtype=np.int64)
    m = len(F)
    info = {"conflicts_before": count_conflicts(F)}
    if m == 0:
        return F, {**info, "conflicts_after": 0, "components": 0, "inverted_components": 0,
                   "open_components": 0}

    e = _directed_edges(F)
    n = int(F.max()) + 1
    key = np.minimum(e[:, 0], e[:, 1]) * n + np.maximum(e[:, 0], e[:, 1])
    order = np.argsort(key, kind="stable")
    ks = key[order]
    new = np.empty(len(ks), dtype=bool)
    new[0] = True
    new[1:] = ks[1:] != ks[:-1]
    first = np.flatnonzero(new)
    counts = np.diff(np.r_[first, len(ks)])
    two = counts == 2
    s0, s1 = order[first[two]], order[first[two] + 1]
    same = e[s0, 0] == e[s1, 0]
    fa, fb = s0 // 3, s1 // 3

    src = np.r_[fa, fb]
    dst = np.r_[fb, fa]
    w = np.r_[same, same].astype(np.int8)
    o2 = np.argsort(src, kind="stable")
    src, dst, w = src[o2], dst[o2], w[o2]
    indptr = np.searchsorted(src, np.arange(m + 1)).tolist()
    dst_l, w_l = dst.tolist(), w.tolist()

    comp = [-1] * m
    flip = [0] * m
    ncomp = 0
    for s in range(m):
        if comp[s] != -1:
            continue
        comp[s] = ncomp
        stack = [s]
        while stack:
            u = stack.pop()
            fu = flip[u]
            for k in range(indptr[u], indptr[u + 1]):
                v = dst_l[k]
                if comp[v] == -1:
                    comp[v] = ncomp
                    flip[v] = fu ^ w_l[k]
                    stack.append(v)
        ncomp += 1
    comp = np.asarray(comp)
    flip = np.asarray(flip, dtype=bool)
    F2 = F.copy()
    F2[flip] = F2[flip][:, ::-1]

    vol = np.einsum("ij,ij->i", V[F2[:, 0]], np.cross(V[F2[:, 1]], V[F2[:, 2]])) / 6.0
    comp_vol = np.bincount(comp, weights=vol, minlength=ncomp)
    gid = np.cumsum(new) - 1
    bad_faces = np.unique(order[(counts != 2)[gid]] // 3)
    comp_size = np.bincount(comp, minlength=ncomp)
    comp_open = np.bincount(comp[bad_faces], minlength=ncomp) > MAX_OPEN_FRACTION * comp_size
    invert = (~comp_open) & (comp_vol < 0)
    flipc = invert[comp]
    F2[flipc] = F2[flipc][:, ::-1]

    info.update(conflicts_after=count_conflicts(F2), components=int(ncomp),
                inverted_components=int(invert.sum()), open_components=int(comp_open.sum()))
    return F2, info


# ---------------------------------------------------------------------------- resampling
def _o3d_mesh(V, F):
    import open3d as o3d
    return o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(np.asarray(V, dtype=np.float64)),
                                     o3d.utility.Vector3iVector(np.asarray(F, dtype=np.int32)))


def decimate_to_vertices(V, F, v_target=V_TARGET, tol=TOL, max_iter=5, boundary_weight=1.0):
    """Quadric-error decimation (Garland-Heckbert, Open3D) aiming at v_target *vertices*.

    Open3D is driven by a triangle count, so start from f_t = v_target / (V/F) of the input
    (about 2*v_target for a closed surface) and correct f_t with the measured vertex count.
    """
    ratio = float(np.clip(len(V) / max(len(F), 1), 0.3, 0.7))
    ft = int(v_target / ratio)
    best, it = None, 0
    for it in range(1, max_iter + 1):
        m = _o3d_mesh(V, F).simplify_quadric_decimation(target_number_of_triangles=max(ft, 4),
                                                        boundary_weight=boundary_weight)
        m.remove_unreferenced_vertices()
        v_out = len(m.vertices)
        if best is None or abs(v_out - v_target) < abs(len(best.vertices) - v_target):
            best = m
        if abs(v_out - v_target) <= tol * v_target:
            break
        ft = int(ft * v_target / max(v_out, 1))
    return np.asarray(best.vertices), np.asarray(best.triangles, dtype=np.int64), it


def cluster_to_vertices(V, F, v_target=V_TARGET, tol=0.05, iters=30):
    """Vertex-clustering simplification (Open3D): vertices falling in the same voxel are merged.

    The voxel size is found by bisection (the vertex count decreases with the voxel size). Every part of
    the mesh stays in place within about one voxel, so thin or loose parts cannot disappear."""
    ext = float((V.max(axis=0) - V.min(axis=0)).max())
    lo, hi = ext / 2000.0, ext / 2.0
    best = None
    for it in range(1, iters + 1):
        vs = np.sqrt(lo * hi)
        m = _o3d_mesh(V, F).simplify_vertex_clustering(voxel_size=vs)
        m.remove_degenerate_triangles()
        m.remove_unreferenced_vertices()
        v_out = len(m.vertices)
        if best is None or abs(v_out - v_target) < abs(len(best.vertices) - v_target):
            best = m
        if abs(v_out - v_target) <= tol * v_target:
            break
        lo, hi = (vs, hi) if v_out > v_target else (lo, vs)
    return np.asarray(best.vertices), np.asarray(best.triangles, dtype=np.int64), it


def decimate_robust(V, F, v_target=V_TARGET, band=BAND, tol=TOL, max_deviation=MAX_DEVIATION):
    """Decimate to v_target; if the result is outside the band or deviates from the input surface by more
    than max_deviation (Hausdorff, relative to the bbox diagonal), try the fall-backs and keep the best.

    Order: quadric (default) -> quadric with a strong boundary weight -> vertex clustering."""
    strategies = [("quadric", lambda: decimate_to_vertices(V, F, v_target, tol)),
                  ("quadric_bw100", lambda: decimate_to_vertices(V, F, v_target, tol, boundary_weight=100.0)),
                  ("vertex_clustering", lambda: cluster_to_vertices(V, F, v_target))]
    tried = []
    for name, run in strategies:
        V2, F2, iters = run()
        dev = fidelity(V, F, V2, F2)["hausdorff_rel"]
        ok = band[0] <= len(V2) <= band[1]
        tried.append((not ok, dev, name, V2, F2, iters))
        if ok and dev <= max_deviation:
            break
    _, _, name, V2, F2, iters = min(tried, key=lambda t: (t[0], t[1]))
    return V2, F2, iters, name, len(tried) > 1


def triangle_areas(V, F):
    return 0.5 * np.linalg.norm(np.cross(V[F[:, 1]] - V[F[:, 0]], V[F[:, 2]] - V[F[:, 0]]), axis=1)


def area_cv(V, F):
    """Coefficient of variation of the triangle areas (0 = all triangles equally large)."""
    a = triangle_areas(V, F)
    return float(a.std() / a.mean()) if len(a) and a.mean() > 0 else float("nan")


def split_long_edges(V, F, L):
    """One conforming refinement pass: every edge longer than L is split at its midpoint.

    The midpoint of an edge is shared by both faces that contain the edge, so the mesh stays
    watertight (no T-junctions). A triangle with k split edges becomes 1, 2, 3 or 4 triangles,
    all keeping the original winding order.
    """
    m, nV = len(F), len(V)
    e = np.stack([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]], axis=1)             # (m, 3, 2)
    length = np.linalg.norm(V[e[:, :, 0]] - V[e[:, :, 1]], axis=2)                # (m, 3)
    mark = length > L
    if not mark.any():
        return V, F, False
    key = np.minimum(e[:, :, 0], e[:, :, 1]).astype(np.int64) * nV + np.maximum(e[:, :, 0], e[:, :, 1])
    uk, inv = np.unique(key[mark], return_inverse=True)
    inv = inv.reshape(-1)
    V = np.vstack([V, 0.5 * (V[uk // nV] + V[uk % nV])])
    mid = np.full((m, 3), -1, dtype=np.int64)
    mid[mark] = nV + inv
    a, b, c = F[:, 0], F[:, 1], F[:, 2]
    m0, m1, m2 = mid[:, 0], mid[:, 1], mid[:, 2]
    k = mark.sum(axis=1)
    t = lambda x, y, z, s: np.stack([x, y, z], axis=1)[s]
    s0 = k == 0
    s3 = k == 3
    s1a, s1b, s1c = (k == 1) & mark[:, 0], (k == 1) & mark[:, 1], (k == 1) & mark[:, 2]
    s2a = (k == 2) & mark[:, 0] & mark[:, 1]      # edges ab, bc split (shared corner b)
    s2b = (k == 2) & mark[:, 1] & mark[:, 2]      # edges bc, ca split (shared corner c)
    s2c = (k == 2) & mark[:, 2] & mark[:, 0]      # edges ca, ab split (shared corner a)
    parts = [F[s0],
             t(a, m0, m2, s3), t(m0, b, m1, s3), t(m2, m1, c, s3), t(m0, m1, m2, s3),
             t(a, m0, c, s1a), t(m0, b, c, s1a),
             t(a, b, m1, s1b), t(a, m1, c, s1b),
             t(a, b, m2, s1c), t(m2, b, c, s1c),
             t(m0, b, m1, s2a), t(a, m0, m1, s2a), t(a, m1, c, s2a),
             t(m1, c, m2, s2b), t(a, b, m1, s2b), t(a, m1, m2, s2b),
             t(m2, a, m0, s2c), t(m0, b, c, s2c), t(m0, c, m2, s2c)]
    return V, np.concatenate(parts).astype(np.int64), True


def refine_adaptive(V, F, v_target=V_TARGET, tol=TOL, max_iter=10, max_attempts=8, band=BAND):
    """Edge-length based (= area based) refinement towards v_target vertices.

    Only edges longer than L are split, so large triangles are refined and small ones are left alone,
    which makes triangle sizes more uniform. For equilateral triangles of edge L covering a surface of
    area A there are N_F = 4A / (sqrt(3) L^2) faces and about N_F / 2 vertices; L is derived from
    v_target and then corrected (N_V ~ 1/L^2) by up to `max_attempts` restarts.
    """
    area = triangle_areas(V, F).sum()
    L = 1.4 * np.sqrt(4.0 * area / (np.sqrt(3.0) * 2.0 * v_target))
    best = None
    for attempt in range(1, max_attempts + 1):
        V2, F2 = V, F
        for _ in range(max_iter):
            V2, F2, changed = split_long_edges(V2, F2, L)
            if not changed or len(F2) > 12 * v_target:
                break
        # prefer candidates inside the acceptance band, then the one closest to the target
        score = abs(np.log(len(V2) / v_target)) + (0.0 if band[0] <= len(V2) <= band[1] else 10.0)
        if best is None or score < best[2]:
            best = (V2, F2, score, attempt)
        if (1 - tol) * v_target <= len(V2) <= (1 + tol) * v_target:
            break
        L *= np.sqrt(len(V2) / v_target)          # N_V is roughly proportional to 1 / L^2
    return best[0], best[1], best[3]


def resample_mesh(V, F, v_target=V_TARGET, band=BAND, tol=TOL, refine="adaptive"):
    """Bring a cleaned mesh towards v_target vertices; shapes already inside `band` are kept.

    refine = "adaptive": edge-length based splitting (default);  "uniform": midpoint subdivision
    of every triangle (kept for comparison).
    """
    lo, hi = band
    log = {"v_in": len(V), "subdiv_iters": 0, "decim_iters": 0}
    if lo <= len(V) <= hi:
        log["action"] = "kept"
        return V, F, log
    if len(V) < lo:
        log["action"] = "refined"
        if refine == "adaptive":
            V, F, log["subdiv_iters"] = refine_adaptive(V, F, v_target, tol, band=band)
            if len(V) < lo:                      # safety net: still below the band -> uniform subdivision
                V, F = trimesh.remesh.subdivide(V, F)
        else:
            while len(V) < v_target and log["subdiv_iters"] < MAX_SUBDIV:
                V, F = trimesh.remesh.subdivide(V, F)
                log["subdiv_iters"] += 1
    else:
        log["action"] = "decimated"
    # adaptive refinement already lands near the target; decimating afterwards would re-create large
    # triangles, so only shapes that ended above the band are decimated
    needs_decimation = (len(V) > hi or len(V) < lo) if (log["action"] == "refined" and refine == "adaptive") \
        else abs(len(V) - v_target) > tol * v_target
    log["decim_method"], log["decim_retried"] = "", False
    if needs_decimation:
        V, F, log["decim_iters"], log["decim_method"], log["decim_retried"] = decimate_robust(
            V, F, v_target, band, tol)
    return np.asarray(V), np.asarray(F, dtype=np.int64), log


def surface_samples(V, F, n, rng):
    """n points drawn uniformly (area-weighted) from the surface of a triangle mesh."""
    a = triangle_areas(V, F)
    tri = rng.choice(len(F), size=n, p=a / a.sum())
    u, v = rng.random(n), rng.random(n)
    flip = u + v > 1
    u[flip], v[flip] = 1 - u[flip], 1 - v[flip]
    p0, p1, p2 = V[F[tri, 0]], V[F[tri, 1]], V[F[tri, 2]]
    return p0 + u[:, None] * (p1 - p0) + v[:, None] * (p2 - p0)


def _distance_to_mesh(P, V, F):
    """Exact unsigned distance from points P to the surface of a triangle mesh (Open3D ray-casting scene)."""
    import open3d as o3d
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(o3d.core.Tensor(np.ascontiguousarray(V, dtype=np.float32)),
                        o3d.core.Tensor(np.ascontiguousarray(F, dtype=np.uint32)))
    return scene.compute_distance(o3d.core.Tensor(np.ascontiguousarray(P, dtype=np.float32))).numpy()


def fidelity(V1, F1, V2, F2, n=20000, seed=0):
    """How far mesh 2 is from mesh 1: n area-weighted points are sampled on each surface and their
    exact distance to the *other* surface is measured, relative to the bbox diagonal of mesh 1.
    chamfer_rel = mean over both directions, hausdorff_rel = maximum over both directions (sampled)."""
    rng = np.random.default_rng(seed)
    P1, P2 = surface_samples(V1, F1, n, rng), surface_samples(V2, F2, n, rng)
    d12 = _distance_to_mesh(P1, V2, F2)
    d21 = _distance_to_mesh(P2, V1, F1)
    diag = float(np.linalg.norm(V1.max(axis=0) - V1.min(axis=0))) or 1.0
    return {"chamfer_rel": float((d12.mean() + d21.mean()) / 2 / diag),
            "hausdorff_rel": float(max(d12.max(), d21.max()) / diag)}


# ------------------------------------------------------------------------------ pipeline
def process_file(args):
    """Worker: raw OBJ -> cleaned, re-oriented, resampled OBJ. Never raises; failures are logged."""
    path, out_path, v_target, band, refine = args
    t0 = time.time()
    p = Path(path)
    rec = {"class": p.parent.name, "file": p.name}
    try:
        verts, sizes, idx = read_obj_raw(p)
        V, F, cinfo = clean_mesh(verts, sizes, idx)
        rec.update(cinfo)
        if len(F) == 0:
            raise ValueError("no faces left after cleaning")
        F, oinfo = fix_orientation(V, F)
        rec.update({f"orient_{k}": v for k, v in oinfo.items()})
        V2, F2, rlog = resample_mesh(V, F, v_target, band, refine=refine)
        rec.update({k: v for k, v in rlog.items()})
        rec["conflicts_after_resample"] = count_conflicts(F2)     # decimation can re-introduce some
        F2, ofinal = fix_orientation(V2, F2)                       # final orientation pass
        rec["v_out"], rec["f_out"] = len(V2), len(F2)
        rec["conflicts_final"] = ofinal["conflicts_after"]
        rec["orientation_ok"] = ofinal["conflicts_after"] == 0
        rec["inverted_final"] = ofinal["inverted_components"]
        rec["area_cv_before"], rec["area_cv_after"] = area_cv(V, F), area_cv(V2, F2)
        if rec["action"] == "kept":
            rec["chamfer_rel"], rec["hausdorff_rel"] = 0.0, 0.0
        else:
            rec.update(fidelity(V, F, V2, F2))
        rec["status"] = "ok" if band[0] <= len(V2) <= band[1] else "out_of_band"
        if out_path is not None:
            write_obj(out_path, V2, F2)
    except Exception as exc:
        rec["status"] = "failed"
        rec["error"] = repr(exc)
    rec["seconds"] = round(time.time() - t0, 2)
    return rec


def run_pipeline(shapes, out_dir, v_target=V_TARGET, band=BAND, workers=8, cache_csv=None,
                 force=False, verbose=True, refine="adaptive"):
    """Run process_file on every row of `shapes` (needs a `path` column); log cached as CSV.

    out_dir=None runs the pipeline without writing meshes (used for method comparisons)."""
    if cache_csv is not None and Path(cache_csv).exists() and not force:
        return pd.read_csv(cache_csv)
    jobs = [(r["path"], None if out_dir is None else str(Path(out_dir) / r["class"] / r["file"]),
             v_target, band, refine) for r in shapes.to_dict("records")]
    jobs.sort(key=lambda j: -Path(j[0]).stat().st_size)
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=workers) as ex:
        results = list(ex.map(process_file, jobs, chunksize=1))
    df = pd.DataFrame(results).sort_values(["class", "file"]).reset_index(drop=True)
    if verbose:
        print(f"processed {len(df)} shapes in {time.time() - t0:.0f}s")
    if cache_csv is not None:
        Path(cache_csv).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(cache_csv, index=False)
    return df
