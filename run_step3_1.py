"""Step 3.1: alignment and flipping of the database normalized in Step 2.5, with the verification
of the whole four-step normalization.

    python src/normalize_shapes.py       # Step 2.5 first: centering and scaling
    python src/run_step3_1.py            # uses the cached log if it exists
    python src/run_step3_1.py --force    # recompute (a few minutes with 8 cores)

Input : processed/step2_normalized/<class>/<file>.obj  (output of Step 2.5)
        processed/step2_full/<class>/<file>.obj        (output of Step 2.3, for the 'before' measures)
Output: processed/step3_normalized/<class>/<file>.obj  (input for Step 3.2)
Logs  : report/data/step3/normalize_log.csv            (measures before and after, per shape)
        report/data/step3/stats_step2_resampled.csv    (Step 2.2 statistics tool, before)
        report/data/step3/stats_step3_normalized.csv   (Step 2.2 statistics tool, after)
"""
import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from mr_io import PROJECT_ROOT
from mr_normalize import STEP2_DIR, STEP2_NORM_DIR, STEP3_DIR, run_normalization
from mr_stats import list_shapes, scan_database

AMBIGUOUS = 1.01    # eigenvalue ratio below which two principal axes are treated as interchangeable
ZERO_MOMENT = 1e-6  # |f_k| / sum_k |f_k| below which a moment counts as zero (symmetric axis)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true", help="recompute even if a cached csv exists")
    args = ap.parse_args()

    # all four steps (translation, alignment, flipping, scaling) are applied to the resampled shapes of
    # Step 2.3, with the centering and scaling functions of Step 2.5; the scaling is done once, at the end
    log = run_normalization(STEP2_DIR, STEP3_DIR, STEP2_DIR, force=args.force,
                            cache_csv=PROJECT_ROOT / "report" / "data" / "step3" / "normalize_log.csv")
    ok = log[log.status == "ok"]
    print(f"3.1: {len(log)} shapes, status {log.status.value_counts().to_dict()}")
    print(f"  translation  max |m| after           : {ok.bary_dist_after.max():.2e}")
    # an eigenvector is only defined if its eigenvalue differs from the others (ratio > AMBIGUOUS)
    d1 = ok.ratio_12_after > AMBIGUOUS
    d2 = d1 & (ok.ratio_23_after > AMBIGUOUS)
    print(f"  alignment    max angle(e1, x) after   : {ok.angle_e1_after[d1].max():.2e} deg "
          f"({d1.sum()} shapes with lambda1/lambda2 > {AMBIGUOUS}; {(~d1).sum()} ambiguous)")
    print(f"               max angle(e2, y) after   : {ok.angle_e2_after[d2].max():.2e} deg "
          f"({d2.sum()} shapes with distinct eigenvalues; {(~d2).sum()} ambiguous)")
    print(f"               max angle(e3, z) after   : {ok.angle_e3_after[d2].max():.2e} deg")
    # a moment that is zero up to rounding (relative to sum |f_k|) belongs to a symmetric axis
    fmin = ok[["f_x_after", "f_y_after", "f_z_after"]].min(axis=1)
    print(f"  flipping     shapes with all f_k >= 0 : {(fmin >= 0).sum()}/{len(ok)}, "
          f"with all f_k >= 0 up to rounding (relative {ZERO_MOMENT:g}): "
          f"{(ok.f_rel_min_after >= -ZERO_MOMENT).sum()}/{len(ok)}")
    print(f"  scaling      d_max after              : {ok.d_max_after.min():.6f} .. {ok.d_max_after.max():.6f}")
    # the signed volume is only meaningful for closed shapes (for open ones it depends on the origin)
    step2 = pd.read_csv(PROJECT_ROOT / "report" / "data" / "step2" / "resample_log_full.csv")
    closed = ok.merge(step2[["class", "file", "orient_open_components"]], on=["class", "file"])
    closed = closed[closed.orient_open_components == 0]
    same = (closed.signed_volume_before > 0) == (closed.signed_volume_after > 0)
    print(f"  orientation  closed shapes with the same sign of the signed volume: {same.sum()}/{len(closed)}")
    print(f"               faces turned inside-out: {ok.faces_turned.sum()} of {ok.faces_checked.sum():,} "
          f"({ok.faces_skipped.sum()} slivers skipped; shapes with a mirror, det < 0: {(ok.det_L < 0).sum()})")

    # the statistics tool of Step 2.2, run on the database before and after the normalization
    D = PROJECT_ROOT / "report" / "data" / "step3"
    before = scan_database(list_shapes(STEP2_DIR), cache_csv=D / "stats_step2_resampled.csv", force=args.force)
    after = scan_database(list_shapes(STEP3_DIR), cache_csv=D / "stats_step3_normalized.csv", force=args.force)
    for name, s in (("before", before), ("after", after)):
        ext = pd.concat([s[f"bbox_max_{a}"] - s[f"bbox_min_{a}"] for a in "xyz"], axis=1)
        print(f"  Step 2.2 statistics {name:6s}: {len(s)} shapes, vertices {s.n_vertices.min()}..{s.n_vertices.max()}, "
              f"largest bbox side {ext.max(axis=1).min():.4g}..{ext.max(axis=1).max():.4g}, "
              f"bbox side order x>=y>=z for {((ext[0] >= ext[1]) & (ext[1] >= ext[2])).sum()} shapes, "
              f"orientation conflicts {s.n_conflicting_edges.sum()}, inverted closed components "
              f"{s.n_inverted_components.sum()}")
    m = before.merge(after, on=["class", "file"], suffixes=("_b", "_a"))
    print(f"  Step 2.2 statistics: same vertex/face counts {((m.n_vertices_b == m.n_vertices_a) & (m.n_faces_b == m.n_faces_a)).sum()}"
          f"/{len(m)}, same orientation conflicts {(m.n_conflicting_edges_b == m.n_conflicting_edges_a).sum()}/{len(m)}")


if __name__ == "__main__":
    main()
