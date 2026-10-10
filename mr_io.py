"""Shared I/O + rendering helpers (Step 2 onward).

load_mesh / the Plotly viewer are copied from the Step 1 notebook so Step 1 stays untouched.
read_obj_raw parses the OBJ text directly so the *original* face sizes (tri/quad/n-gon) are kept,
which trimesh's force-triangulation (used in Step 1) hides.
"""
from pathlib import Path

import numpy as np
import trimesh

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "Data"


def load_mesh(path) -> trimesh.Trimesh:
    """Step 1 loader: triangulated trimesh, no automatic processing."""
    mesh = trimesh.load(path, force="mesh", process=False)
    if not isinstance(mesh, trimesh.Trimesh):
        raise ValueError(f"Could not load a triangle mesh from {path}")
    return mesh


def read_obj_raw(path):
    """Parse an OBJ file without triangulating or merging anything.

    Returns
    -------
    verts : (n, 3) float64     all `v` lines, in file order
    sizes : (m,) int64         number of vertices of every `f` line (3 = triangle, 4 = quad, ...)
    idx   : (sum(sizes),) int64  0-based vertex indices of all faces, concatenated
    """
    verts, sizes, idx = [], [], []
    nv = 0
    with open(path, "r", errors="ignore") as fh:
        for line in fh:
            c = line[:2]
            if c == "v ":
                p = line.split()
                verts.append((float(p[1]), float(p[2]), float(p[3])))
                nv += 1
            elif c == "f ":
                toks = line.split()[1:]
                if "/" in line:
                    toks = [t.split("/", 1)[0] for t in toks]
                face = [int(t) for t in toks]
                face = [i - 1 if i > 0 else nv + i for i in face]
                sizes.append(len(face))
                idx.extend(face)
    verts = np.asarray(verts, dtype=np.float64).reshape(-1, 3)
    return verts, np.asarray(sizes, dtype=np.int64), np.asarray(idx, dtype=np.int64)


def triangulate(sizes, idx):
    """Fan-triangulate polygon faces: (a0,a1,...,ak-1) -> (a0,ai,ai+1). Returns (T, 3) int64."""
    sizes = np.asarray(sizes)
    idx = np.asarray(idx)
    if sizes.size == 0:
        return np.empty((0, 3), dtype=np.int64)
    if (sizes == 3).all():
        return idx.reshape(-1, 3)
    starts = np.cumsum(sizes) - sizes
    reps = np.clip(sizes - 2, 0, None)
    total = int(reps.sum())
    face_id = np.repeat(np.arange(sizes.size), reps)
    i = np.arange(total) - np.repeat(np.cumsum(reps) - reps, reps) + 1
    s = starts[face_id]
    return np.stack([idx[s], idx[s + i], idx[s + i + 1]], axis=1)


def write_obj(path, V, F):
    """Write a triangle mesh as a plain OBJ (v / f lines only)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as fh:
        np.savetxt(fh, np.asarray(V), fmt="v %.7g %.7g %.7g")
        np.savetxt(fh, np.asarray(F, dtype=np.int64) + 1, fmt="f %d %d %d")


def mesh_arrays(path):
    """Raw vertices + triangulated faces of an OBJ file (no cleaning) - for display."""
    verts, sizes, idx = read_obj_raw(path)
    return verts, triangulate(sizes, idx)


def render_png(V, F, out_png, wireframe=False, title=None, max_faces=40000,
               max_wire_faces=25000, width=700, height=550, flatshading=False):
    """Static render (Plotly + kaleido) of a triangle mesh, same look as the Step 1 viewer.

    Smooth shading is the default: Plotly's flat shading produces dark/light blotches on dense
    meshes (the Hand in Step 1), independent of face orientation. Large meshes are decimated
    *for display only*; the title says so.
    """
    import plotly.graph_objects as go

    V = np.asarray(V, dtype=np.float64)
    F = np.asarray(F, dtype=np.int64)
    note = ""
    if len(F) > max_faces:
        import open3d as o3d
        m = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(V), o3d.utility.Vector3iVector(F))
        m = m.simplify_quadric_decimation(max_faces)
        V, F = np.asarray(m.vertices), np.asarray(m.triangles)
        note = f" [display decimated to {len(F):,} faces]"

    data = [go.Mesh3d(
        x=V[:, 0], y=V[:, 1], z=V[:, 2], i=F[:, 0], j=F[:, 1], k=F[:, 2],
        color="lightsteelblue", flatshading=flatshading, opacity=1.0,
        lighting=dict(ambient=0.5, diffuse=0.8, specular=0.2, roughness=0.6),
        lightposition=dict(x=100, y=200, z=150),
    )]
    if wireframe and len(F) <= max_wire_faces:
        tri = np.concatenate([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]])
        a, b = V[tri[:, 0]], V[tri[:, 1]]
        nan = np.full((len(tri), 1), np.nan)
        xs = np.hstack([a[:, :1], b[:, :1], nan]).ravel()
        ys = np.hstack([a[:, 1:2], b[:, 1:2], nan]).ravel()
        zs = np.hstack([a[:, 2:], b[:, 2:], nan]).ravel()
        data.append(go.Scatter3d(x=xs, y=ys, z=zs, mode="lines",
                                 line=dict(color="black", width=1), showlegend=False))
    elif wireframe:
        note += " [wireframe omitted: too many faces]"

    fig = go.Figure(data=data)
    fig.update_layout(
        title=dict(text=(title or "") + note, font=dict(size=13)),
        scene=dict(aspectmode="data", xaxis=dict(visible=False),
                   yaxis=dict(visible=False), zaxis=dict(visible=False),
                   camera=dict(eye=dict(x=1.7, y=1.7, z=1.1))),
        margin=dict(l=0, r=0, t=40, b=0), width=width, height=height,
    )
    out_png = Path(out_png)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.write_image(str(out_png), scale=2)
    return out_png
