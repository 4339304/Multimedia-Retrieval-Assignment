from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from mr_io import PROJECT_ROOT, mesh_arrays, render_png

PROCESSED = PROJECT_ROOT / "processed"
DATA_OUT = PROJECT_ROOT / "report" / "data" / "step2"
FIG = PROJECT_ROOT / "report" / "figures"

refined_dir = PROCESSED / "step2_full"
NORMALIZED_DIR = PROCESSED / "step2_normalized"

def read_obj_vertices_and_face_count(path):
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


def normalize_obj(source_path, destination_path):
    lines = source_path.read_text(encoding="utf-8", errors="replace").splitlines(keepends=True)
    vertices = []
    vertex_lines = []

    for line_index, line in enumerate(lines):
        content = line.split("#", 1)[0].strip()
        if content:
            tokens = content.split()
            if tokens[0] == "v":
                vertices.append([float(tokens[1]), float(tokens[2]), float(tokens[3])])
                vertex_lines.append(line_index)

    V = np.asarray(vertices, dtype=np.float64).reshape(-1, 3)
    if len(V) == 0 or not np.isfinite(V).all():
        raise ValueError(f"Empty or invalid vertices: {source_path}")

    # Center the mean vertex position at the origin.
    centered = V - V.mean(axis=0)

    # Uniform scaling: largest bounding-box dimension becomes 1.
    largest_extent = np.ptp(centered, axis=0).max()
    if largest_extent <= 0:
        raise ValueError(f"Mesh has zero size: {source_path}")
    normalized = centered / largest_extent

    # Replace only vertex coordinates, preserving faces and other OBJ records.
    for i, line_index in enumerate(vertex_lines):
        old_line = lines[line_index]
        body, separator, comment = old_line.partition("#")
        tokens = body.split()
        xyz = " ".join(f"{v:.12g}" for v in normalized[i])
        new_line = "v " + xyz
        if len(tokens) > 4:
            new_line += " " + " ".join(tokens[4:])
        if separator:
            new_line += " #" + comment.rstrip("\r\n")
        if old_line.endswith("\r\n"):
            ending = "\r\n"
        elif old_line.endswith("\n"):
            ending = "\n"
        elif old_line.endswith("\r"):
            ending = "\r"
        else:
            ending = ""
        lines[line_index] = new_line + ending

    destination_path.parent.mkdir(parents=True, exist_ok=True)
    destination_path.write_text("".join(lines), encoding="utf-8")


if not refined_dir.exists():
    raise FileNotFoundError(f"Resampled input directory not found: {refined_dir}")

DATA_OUT.mkdir(parents=True, exist_ok=True)
NORMALIZED_DIR.mkdir(parents=True, exist_ok=True)

source_files = sorted(refined_dir.rglob("*.obj"))
errors = []

print("STEP 2.5: NORMALIZING SHAPES")
print("=" * 50)
print(f"Input meshes: {len(source_files)}")

for index, source in enumerate(source_files, start=1):
    relative = source.relative_to(refined_dir)
    destination = NORMALIZED_DIR / relative
    try:
        normalize_obj(source, destination)
    except Exception as exc:
        errors.append({"file": str(relative), "error": str(exc)})

    if index % 100 == 0 or index == len(source_files):
        print(f"Processed {index}/{len(source_files)}")

normalized_files = sorted(NORMALIZED_DIR.rglob("*.obj"))
verification_rows = []
verification_errors = []

for path in normalized_files:
    relative = path.relative_to(NORMALIZED_DIR)
    original = refined_dir / relative
    try:
        V, face_count = read_obj_vertices_and_face_count(path)
        _, original_face_count = read_obj_vertices_and_face_count(original)

        if len(V) == 0 or not np.isfinite(V).all():
            raise ValueError("Invalid or empty vertex array")

        barycenter = V.mean(axis=0)
        extents = np.ptp(V, axis=0)
        max_extent = extents.max()
        center_error = np.linalg.norm(barycenter)

        verification_rows.append({
            "class": relative.parent.name,
            "file": relative.name,
            "n_vertices": len(V),
            "n_faces": face_count,
            "barycenter_x": barycenter[0],
            "barycenter_y": barycenter[1],
            "barycenter_z": barycenter[2],
            "center_error": center_error,
            "extent_x": extents[0],
            "extent_y": extents[1],
            "extent_z": extents[2],
            "max_extent": max_extent,
            "face_count_preserved": face_count == original_face_count,
            "center_ok": center_error < 1e-8,
            "size_ok": np.isclose(max_extent, 1.0, atol=1e-8),
        })
    except Exception as exc:
        verification_errors.append({"file": str(relative), "error": str(exc)})

verification = pd.DataFrame(verification_rows)
if verification.empty:
    raise RuntimeError("No normalized meshes could be verified.")

verification["passed"] = (
    verification["center_ok"]
    & verification["size_ok"]
    & verification["face_count_preserved"]
)

verification.to_csv(DATA_OUT / "step2_5_normalization_verification.csv", index=False)

summary = pd.DataFrame({
    "measurement": [
        "Input shapes",
        "Normalized OBJ files",
        "Shapes verified",
        "Shapes passing",
        "Shapes failing",
        "Mean center error",
        "Maximum center error",
        "Minimum largest dimension",
        "Maximum largest dimension",
        "All face counts preserved",
        "Normalization errors",
        "Verification errors",
    ],
    "value": [
        len(source_files),
        len(normalized_files),
        len(verification),
        int(verification["passed"].sum()),
        int((~verification["passed"]).sum()),
        verification["center_error"].mean(),
        verification["center_error"].max(),
        verification["max_extent"].min(),
        verification["max_extent"].max(),
        bool(verification["face_count_preserved"].all()),
        len(errors),
        len(verification_errors),
    ],
})
summary.to_csv(DATA_OUT / "step2_5_normalization_summary.csv", index=False)

if errors:
    pd.DataFrame(errors).to_csv(DATA_OUT / "step2_5_normalization_errors.csv", index=False)
if verification_errors:
    pd.DataFrame(verification_errors).to_csv(DATA_OUT / "step2_5_verification_errors.csv", index=False)

print("\nNORMALIZATION VERIFICATION")
print("=" * 50)
print(f"Normalized files: {len(normalized_files)}")
print(f"Shapes verified: {len(verification)}")
print(f"Shapes passing: {int(verification['passed'].sum())}")
print(f"Shapes failing: {int((~verification['passed']).sum())}")
print(f"Mean center error: {verification['center_error'].mean():.3e}")
print(f"Maximum center error: {verification['center_error'].max():.3e}")
print(f"Largest dimension range: {verification['max_extent'].min():.8f} to {verification['max_extent'].max():.8f}")
print(f"All face counts preserved: {verification['face_count_preserved'].all()}")
print(f"Normalization errors: {len(errors)}")
print(f"Verification errors: {len(verification_errors)}")
print(f"\nNormalized meshes saved in: {NORMALIZED_DIR}")
print(f"Statistics saved in: {DATA_OUT}")


# Viewer check: the Bird of Step 2.4 after normalization (same names as in the report).
def crop_white(png, pad=12):
    img = plt.imread(png)
    ys, xs = (img[..., :3].min(axis=2) < 0.97).nonzero()
    plt.imsave(png, img[max(ys.min() - pad, 0):ys.max() + pad, max(xs.min() - pad, 0):xs.max() + pad])


bird = NORMALIZED_DIR / "Bird" / "D00089.obj"
V, F = mesh_arrays(bird)
for kind, wire in (("shaded", False), ("wireframe", True)):
    png = FIG / "step2" / f"Bird_normalized_{kind}.png"
    render_png(V, F, png, wireframe=wire, title=None)
    crop_white(png)
print(f"Viewer images of the normalized Bird saved in: {FIG / 'step2'}")
