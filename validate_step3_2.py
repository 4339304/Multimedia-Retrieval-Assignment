
import numpy as np
import trimesh

from run_s3_2 import global_descriptors, sample_distributions, make_histograms

SAMPLES = 10000
BINS = 32


def test_shape(name, mesh):
    V = np.asarray(mesh.vertices, dtype=np.float64)
    F = np.asarray(mesh.faces, dtype=np.int64)

    print(f"\n{'=' * 45}")
    print(f"{name.upper()} VALIDATION")
    print(f"{'=' * 45}")

    print("Watertight:", mesh.is_watertight)
    assert mesh.is_watertight, f"{name} must be watertight"

    # Test the actual global descriptor implementation.
    desc = global_descriptors(V, F)

    print("\nGlobal descriptors:")
    for key, value in desc.items():
        print(f"  {key}: {value}")

    # Both test shapes have a largest bounding-box dimension of 1.
    assert np.isclose(np.ptp(V, axis=0).max(), 1.0)
    assert np.isfinite(desc["surface_area"])
    assert desc["surface_area"] > 0
    assert np.isfinite(desc["diameter"])
    assert np.isclose(desc["compactness_sphere"],
                      36 * np.pi * desc["mesh_volume"]**2
                      / desc["surface_area"]**3)

    # Convex, closed test shapes should have convexity near 1.
    assert np.isclose(desc["convexity"], 1.0, atol=1e-5)

    # Check the five distribution descriptors and histograms.
    rng = np.random.default_rng(42)
    distributions = sample_distributions(V, SAMPLES, rng)
    histograms = make_histograms(distributions, BINS)

    print("\nHistogram validation:")
    for name_, values in distributions.items():
        hist = np.array([
            histograms[f"{name_}_bin_{i:02d}"]
            for i in range(BINS)
        ])
        print(
            f"  {name_}: samples={len(values)}, "
            f"bins={len(hist)}, histogram_sum={hist.sum():.6f}"
        )
        assert len(values) == SAMPLES
        assert np.isclose(hist.sum(), 1.0), (
            f"{name_} histogram does not sum to 1"
        )

    print(f"\n{ name } basic validation: PASSED")


def main():
    # Unit cube: side length 1.
    cube = trimesh.creation.box(extents=[1.0, 1.0, 1.0])

    # Closed triangular approximation of a sphere, diameter 1.
    sphere = trimesh.creation.icosphere(
        subdivisions=3,
        radius=0.5
    )

    test_shape("Cube", cube)
    test_shape("Sphere", sphere)

    print("\nAll Step 3.2 basic validation tests passed.")


if __name__ == "__main__":
    main()
