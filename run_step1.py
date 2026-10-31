"""
Step 1: load test of the whole database and figures of the six test shapes (report/figures/step1/).

    python run_step1.py               load test, Table 2.1 and all figures
    python run_step1.py --no-figures  only the load test and Table 2.1

The interactive viewer is view_mesh.py. This script uses the same loading function (trimesh, no processing) and
writes static images with the same lighting as the viewer (mr_io.render_png):
1. <Class>_shaded.png, <Class>_wireframe.png  the six test shapes (smooth shading; the coarse Bird flat)
2. bird_shading.png                            the Bird with smooth and with flat shading
3. hand_flat_crop.png, hand_flat_reversed_crop.png, hand_smooth_crop.png
                                               the Hand with flat shading, flat shading with all faces reversed,
                                               and smooth shading (the image does not change when the faces are
                                               reversed, because the renderer lights both sides of a face)
"""
import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from figures_step3_1 import crop_white
from mr_io import DATA_DIR, PROJECT_ROOT, render_png
from view_mesh import load_mesh

FIG = PROJECT_ROOT / "report" / "figures" / "step1"
SHAPES = [("Bicycle", "D00016.obj"), ("Bird", "D00089.obj"), ("Chess", "D01017.obj"),
          ("Hand", "D01006.obj"), ("House", "D00001.obj"), ("Tree", "D00096.obj")]
FLAT = {"Bird"}                     # very coarse mesh: flat shading shows it better (Section 2.5 of the report)


def load_test():
    """Load every OBJ file of the database and count failures and empty meshes."""
    files = sorted(DATA_DIR.rglob("*.obj"))
    if not files:
        raise SystemExit(f"no OBJ files in {DATA_DIR}: put the shape database there (see README.txt)")
    failed, empty = [], []
    for f in files:
        try:
            m = load_mesh(f)
            if len(m.vertices) == 0 or len(m.faces) == 0:
                empty.append(f)
        except Exception as e:                       # report every file that cannot be read
            failed.append((f, e))
    classes = len({f.parent.name for f in files})
    print(f"load test: {len(files)} OBJ files in {classes} classes, {len(failed)} failed, {len(empty)} empty")
    for f, e in failed:
        print("  failed:", f, e)


def six_shapes():
    rows = []
    for cls, f in SHAPES:
        m = load_mesh(DATA_DIR / cls / f)
        rows.append({"class": cls, "file": f[:-4], "N_V": len(m.vertices), "N_F": len(m.faces)})
    print(pd.DataFrame(rows).to_string(index=False))


def render(V, F, png, **kw):
    render_png(V, F, png, max_faces=10**7, max_wire_faces=10**7, **kw)     # full mesh, no display decimation
    crop_white(png)


def figures():
    FIG.mkdir(parents=True, exist_ok=True)
    for cls, f in SHAPES:
        m = load_mesh(DATA_DIR / cls / f)
        V, F = m.vertices, m.faces
        render(V, F, FIG / f"{cls}_shaded.png", flatshading=cls in FLAT)
        render(V, F, FIG / f"{cls}_wireframe.png", wireframe=True, flatshading=cls in FLAT)

    bird = load_mesh(DATA_DIR / "Bird" / "D00089.obj")
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.6))
    for k, (flat, title) in enumerate([(False, "(a) smooth shading (averaged vertex normals)"),
                                       (True, "(b) flat shading (one normal per triangle)")]):
        png = FIG / f"bird_{'flat' if flat else 'smooth'}.png"
        render_png(bird.vertices, bird.faces, png, width=900, height=600, flatshading=flat)
        crop_white(png)
        ax[k].imshow(plt.imread(png))
        ax[k].set_title(title, fontsize=11)
        ax[k].axis("off")
    fig.tight_layout()
    fig.savefig(FIG / "bird_shading.png", dpi=150)
    plt.close(fig)

    hand = load_mesh(DATA_DIR / "Hand" / "D01006.obj")
    V, F = hand.vertices, hand.faces
    render(V, F, FIG / "hand_flat_crop.png", flatshading=True)
    render(V, F[:, ::-1], FIG / "hand_flat_reversed_crop.png", flatshading=True)    # every face reversed
    render(V, F, FIG / "hand_smooth_crop.png")
    print("figures saved in", FIG)


def main():
    ap = argparse.ArgumentParser(description="Step 1: load test and figures of the six test shapes")
    ap.add_argument("--no-figures", action="store_true", help="only the load test and the table of the six shapes")
    a = ap.parse_args()
    load_test()
    six_shapes()
    if not a.no_figures:
        figures()


if __name__ == "__main__":
    main()
