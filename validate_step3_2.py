"""
Step 3.2 check on simple shapes whose descriptor values are known.

    python validate_step3_2.py

Shapes: unit cube, sphere (diameter 1), box 1 x 0.5 x 0.25 and the same box with one side
removed (an open mesh, to test the hole filling). Writes report/data/step3/step3_2_validation.txt.
"""
import math

import numpy as np
import trimesh

from mr_io import PROJECT_ROOT
from run_s3_2 import DEFAULT_BINS, NAMES, global_descriptors, make_histograms, sample_distributions

SAMPLES = 100_000
OUT = PROJECT_ROOT / "report" / "data" / "step3" / "step3_2_validation.txt"
lines = []


def say(text=""):
    print(text)
    lines.append(text)


def check(name, value, expected, tol):
    ok = abs(value - expected) <= tol
    say(f"  {name:20s} {value:10.4f}   expected {expected:8.4f}   {'ok' if ok else 'FAILED'}")
    assert ok, f"{name}: {value} instead of {expected}"


def test_shape(name, mesh, expected):
    V = np.asarray(mesh.vertices, dtype=np.float64)
    F = np.asarray(mesh.faces, dtype=np.int64)
    say(f"\n{name} ({len(V)} vertices, {len(F)} faces, closed: {mesh.is_watertight})")
    desc = global_descriptors(V, F)
    say(f"  volume source: {desc['volume_source']}, holes filled: {desc['holes_filled']}")
    for key, (value, tol) in expected.items():
        check(key, desc[key], value, tol)

    hist = make_histograms(sample_distributions(V, SAMPLES, np.random.default_rng(42)), DEFAULT_BINS)
    sums = [sum(hist[f"{n}_bin_{i:02d}"] for i in range(DEFAULT_BINS)) for n in NAMES]
    say("  histogram sums: " + ", ".join(f"{n} {s:.6f}" for n, s in zip(NAMES, sums)))
    assert np.allclose(sums, 1.0)


def main():
    cube = trimesh.creation.box(extents=[1.0, 1.0, 1.0])
    sphere = trimesh.creation.icosphere(subdivisions=4, radius=0.5)
    box = trimesh.creation.box(extents=[1.0, 0.5, 0.25])
    open_box = box.copy()
    open_box.update_faces(open_box.face_normals[:, 0] < 0.5)   # remove the two triangles of side x = +0.5

    s_vol, s_area = sphere.volume, sphere.area
    test_shape("Cube", cube, {
        "surface_area": (6.0, 1e-9), "mesh_volume": (1.0, 1e-9),
        "compactness": (6.0**3 / (36 * math.pi), 1e-6), "sphericity": (math.pi / 6, 1e-6),
        "rectangularity_3d": (1.0, 1e-9), "convexity": (1.0, 1e-9),
        "diameter": (math.sqrt(3), 1e-9), "eccentricity": (1.0, 1e-9)})
    # the icosphere is slightly smaller than the exact sphere, so compare with pi/6 loosely
    test_shape("Sphere", sphere, {
        "mesh_volume": (s_vol, 1e-9), "sphericity": (1.0, 0.01),
        "rectangularity_3d": (math.pi / 6, 0.01), "convexity": (1.0, 1e-9),
        "diameter": (1.0, 1e-9), "eccentricity": (1.0, 0.01)})
    # box corners: the variance along an axis is a quarter of the side squared (1/4, 1/16, 1/64),
    # so the eccentricity is (1/0.25)^2 = 16
    test_shape("Box 1 x 0.5 x 0.25", box, {
        "surface_area": (1.75, 1e-9), "mesh_volume": (0.125, 1e-9), "rectangularity_3d": (1.0, 1e-9),
        "convexity": (1.0, 1e-9), "diameter": (math.sqrt(1 + 0.25 + 0.0625), 1e-9),
        "eccentricity": (16.0, 1e-6)})
    test_shape("Open box (one side removed)", open_box, {
        "surface_area": (1.75 - 0.125, 1e-9), "mesh_volume": (0.125, 1e-9),
        "rectangularity_3d": (1.0, 1e-9), "convexity": (1.0, 1e-9),
        "sphericity": (36 * math.pi * 0.125**2 / 1.75**3, 1e-9)})   # same as the closed box
    say(f"\n(sphere: area {s_area:.4f}, volume {s_vol:.4f})")
    say("All Step 3.2 validation tests passed.")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
