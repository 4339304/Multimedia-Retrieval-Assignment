"""
Step 2.5: translation and scale normalization of the resampled shapes.

Run from the project folder (after Step 2.3):
    python normalize_shapes.py

Input:  processed/step2_full/                      resampled shapes (Step 2.3)
Output: processed/step2_normalized/                centered and scaled shapes
        report/data/step2/step2_5_normalization_verification.csv   check of every saved shape
        report/data/step2/step2_5_normalization_summary.csv        summary of the check
        report/figures/step2/Bird_normalized_*.png                 viewer images of the normalized Bird

Every shape is moved so that the mean of its vertices is at the origin, and scaled uniformly so that the
largest side of its axis-aligned bounding box is 1. Only the vertex lines of the OBJ file are rewritten, so
the faces and all other lines stay as they were. The saved files are then read again and checked.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from mr_io import PROJECT_ROOT, mesh_arrays, render_png

PROCESSED = PROJECT_ROOT / "processed"
DATA_OUT = PROJECT_ROOT / "report" / "data" / "step2"
FIG = PROJECT_ROOT / "report" / "figures"
REFINED_DIR = PROCESSED / "step2_full"
NORMALIZED_DIR = PROCESSED / "step2_normalized"


# ------------------------------------------------------------------ normalization
def read_obj_vertices_and_face_count(path):
    """Vertex positions and number of faces of an OBJ file."""
    vertices = []
    face_count = 0
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            content = line.split("#", 1)[0].strip()
            if not content:
                continue
            tokens = content.split()
            if tokens[0] == "v":
                vertices.append([float(tokens[1]), float(tokens[2]), float(tokens[3])])
            elif tokens[0] == "f":
                face_count += 1
    return np.asarray(vertices, dtype=np.float64).reshape(-1, 3), face_count


def line_ending(line):
    for ending in ("\r\n", "\n", "\r"):
        if line.endswith(ending):
            return ending
    return ""


def normalize_obj(source_path, destination_path):
    """Center (barycenter = mean of the vertices) and scale (largest side = 1) one OBJ file."""
    lines = source_path.read_text(encoding="utf-8", errors="replace").splitlines(keepends=True)
    vertices, vertex_lines = [], []
    for line_index, line in enumerate(lines):
        content = line.split("#", 1)[0].strip()
        if content:
            tokens = content.split()
            if tokens[0] == "v":
                vertices.append([float(tokens[1]), float(tokens[2]), float(tokens[3])])
                vertex_lines.append(line_index)
    V = np.asarray(vertices, dtype=np.float64).reshape(-1, 3)
    if len(V) == 0 or not np.isfinite(V).all():
        raise ValueError(f"no valid vertices: {source_path}")

    centered = V - V.mean(axis=0)                       # translation
    largest_extent = np.ptp(centered, axis=0).max()
    if largest_extent <= 0:
        raise ValueError(f"shape has zero size: {source_path}")
    normalized = centered / largest_extent              # scaling

    # rewrite only the coordinates of the vertex lines; extra values and comments are kept
    for i, line_index in enumerate(vertex_lines):
        old_line = lines[line_index]
        body, separator, comment = old_line.partition("#")
        tokens = body.split()
        new_line = "v " + " ".join(f"{v:.12g}" for v in normalized[i])
        if len(tokens) > 4:
            new_line += " " + " ".join(tokens[4:])
        if separator:
            new_line += " #" + comment.rstrip("\r\n")
        lines[line_index] = new_line + line_ending(old_line)
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    destination_path.write_text("".join(lines), encoding="utf-8")


def normalize_all():
    """Normalize every resampled shape; returns the list of input files and the errors."""
    source_files = sorted(REFINED_DIR.rglob("*.obj"))
    errors = []
    print(f"Step 2.5: {len(source_files)} input meshes")
    for index, source in enumerate(source_files, start=1):
        relative = source.relative_to(REFINED_DIR)
        try:
            normalize_obj(source, NORMALIZED_DIR / relative)
        except Exception as exc:
            errors.append({"file": str(relative), "error": str(exc)})
        if index % 100 == 0 or index == len(source_files):
            print(f"processed {index}/{len(source_files)}")
    return source_files, errors


# ------------------------------------------------------------------ verification
def verify():
    """Read every saved shape again and check its barycenter, its largest side and its number of faces."""
    normalized_files = sorted(NORMALIZED_DIR.rglob("*.obj"))
    rows, errors = [], []
    for path in normalized_files:
        relative = path.relative_to(NORMALIZED_DIR)
        try:
            V, face_count = read_obj_vertices_and_face_count(path)
            _, original_face_count = read_obj_vertices_and_face_count(REFINED_DIR / relative)
            if len(V) == 0 or not np.isfinite(V).all():
                raise ValueError("no valid vertices")
            barycenter = V.mean(axis=0)
            extents = np.ptp(V, axis=0)
            center_error = np.linalg.norm(barycenter)
            rows.append({
                "class": relative.parent.name, "file": relative.name, "n_vertices": len(V), "n_faces": face_count,
                "barycenter_x": barycenter[0], "barycenter_y": barycenter[1], "barycenter_z": barycenter[2],
                "center_error": center_error,
                "extent_x": extents[0], "extent_y": extents[1], "extent_z": extents[2], "max_extent": extents.max(),
                "face_count_preserved": face_count == original_face_count,
                "center_ok": center_error < 1e-8,
                "size_ok": np.isclose(extents.max(), 1.0, atol=1e-8),
            })
        except Exception as exc:
            errors.append({"file": str(relative), "error": str(exc)})
    verification = pd.DataFrame(rows)
    if verification.empty:
        raise RuntimeError("no normalized mesh could be checked")
    verification["passed"] = verification["center_ok"] & verification["size_ok"] & verification["face_count_preserved"]
    return normalized_files, verification, errors


def save_results(source_files, errors, normalized_files, verification, verification_errors):
    verification.to_csv(DATA_OUT / "step2_5_normalization_verification.csv", index=False)
    v = verification
    summary = pd.DataFrame({
        "measurement": ["Input shapes", "Normalized OBJ files", "Shapes verified", "Shapes passing", "Shapes failing",
                        "Mean center error", "Maximum center error", "Minimum largest dimension",
                        "Maximum largest dimension", "All face counts preserved", "Normalization errors",
                        "Verification errors"],
        "value": [len(source_files), len(normalized_files), len(v), int(v["passed"].sum()), int((~v["passed"]).sum()),
                  v["center_error"].mean(), v["center_error"].max(), v["max_extent"].min(), v["max_extent"].max(),
                  bool(v["face_count_preserved"].all()), len(errors), len(verification_errors)],
    })
    summary.to_csv(DATA_OUT / "step2_5_normalization_summary.csv", index=False)
    if errors:
        pd.DataFrame(errors).to_csv(DATA_OUT / "step2_5_normalization_errors.csv", index=False)
    if verification_errors:
        pd.DataFrame(verification_errors).to_csv(DATA_OUT / "step2_5_verification_errors.csv", index=False)

    print("\nverification of the normalized shapes")
    print(f"normalized files: {len(normalized_files)}, checked: {len(v)}")
    print(f"passing: {int(v['passed'].sum())}, failing: {int((~v['passed']).sum())}")
    print(f"distance of the barycenter to the origin: mean {v['center_error'].mean():.3e}, "
          f"max {v['center_error'].max():.3e}")
    print(f"largest side: {v['max_extent'].min():.8f} to {v['max_extent'].max():.8f}")
    print(f"all face counts unchanged: {v['face_count_preserved'].all()}")
    print(f"errors: {len(errors)} in the normalization, {len(verification_errors)} in the check")
    print(f"\nnormalized meshes: {NORMALIZED_DIR}")
    print(f"statistics: {DATA_OUT}")


# ------------------------------------------------------------------ viewer check
def crop_white(png, pad=12):
    img = plt.imread(png)
    ys, xs = (img[..., :3].min(axis=2) < 0.97).nonzero()
    plt.imsave(png, img[max(ys.min() - pad, 0):ys.max() + pad, max(xs.min() - pad, 0):xs.max() + pad])


def bird_images():
    """Viewer images of the Bird of Step 2.4 after the normalization (file names used in the report)."""
    V, F = mesh_arrays(NORMALIZED_DIR / "Bird" / "D00089.obj")
    for kind, wire in (("shaded", False), ("wireframe", True)):
        png = FIG / "step2" / f"Bird_normalized_{kind}.png"
        render_png(V, F, png, wireframe=wire, title=None)
        crop_white(png)
    print(f"viewer images of the normalized Bird: {FIG / 'step2'}")


def main():
    if not REFINED_DIR.exists():
        raise FileNotFoundError(f"resampled shapes not found: {REFINED_DIR} (run Step 2.3 first)")
    DATA_OUT.mkdir(parents=True, exist_ok=True)
    NORMALIZED_DIR.mkdir(parents=True, exist_ok=True)
    source_files, errors = normalize_all()
    normalized_files, verification, verification_errors = verify()
    save_results(source_files, errors, normalized_files, verification, verification_errors)
    bird_images()


if __name__ == "__main__":
    main()
