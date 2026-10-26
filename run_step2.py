"""Re-run the Step 2 computations from the command line.

    python src/run_step2.py                 # uses cached csv results where they exist (seconds)
    python src/run_step2.py --force         # recompute everything (about 6 minutes with 8 worker processes)
    python src/run_step2.py --force --skip-full   # only the database scan and the reduced database (about 2 minutes)
"""
import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from mr_io import PROJECT_ROOT
from mr_resample import run_pipeline
from mr_stats import list_shapes, scan_database, select_reduced


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true", help="recompute even if a cached csv exists")
    ap.add_argument("--skip-full", action="store_true", help="do not resample the full database")
    args = ap.parse_args()

    D = PROJECT_ROOT / "report" / "data" / "step2"
    P = PROJECT_ROOT / "processed"

    stats = scan_database(list_shapes(), cache_csv=D / "shape_stats_raw.csv", force=args.force)
    print(f"2.1/2.2 scan: {len(stats)} shapes, face types {stats['face_type'].value_counts().to_dict()}")

    reduced_csv = D / "reduced_database.csv"
    if args.force or not reduced_csv.exists():
        select_reduced(stats).to_csv(reduced_csv, index=False)
    reduced = pd.read_csv(reduced_csv)

    jobs = [("reduced (adaptive)", reduced, P / "step2_reduced", "adaptive", "resample_log_reduced.csv"),
            ("reduced (uniform, comparison only)", reduced, None, "uniform", "resample_log_reduced_uniform.csv")]
    if not args.skip_full:
        jobs.append(("full database", stats, P / "step2_full", "adaptive", "resample_log_full.csv"))
    for name, shapes, out_dir, refine, csv in jobs:
        log = run_pipeline(shapes, out_dir, refine=refine, cache_csv=D / csv, force=args.force)
        print(f"2.3 {name}: {len(log)} shapes, status {log.status.value_counts().to_dict()}, "
              f"vertices after: mean {log.v_out.mean():.0f}, std {log.v_out.std():.0f}, "
              f"in band {100 * log.v_out.between(5000, 10000).mean():.2f} %, "
              f"orientation_ok {log.orientation_ok.sum()}/{len(log)}")


if __name__ == "__main__":
    main()
