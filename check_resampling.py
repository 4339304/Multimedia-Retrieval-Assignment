resample_log_path = DATA_OUT / "resample_log_full.csv"
refined_dir = PROCESSED / "step2_full"

if not resample_log_path.exists():
    raise FileNotFoundError(
        "The resampling log is missing. "
        "Please run Step 2.3 first."
    )

if not refined_dir.exists():
    raise FileNotFoundError(
        "The processed mesh directory is missing. "
        "Please run Step 2.3 first."
    )

full = pd.read_csv(resample_log_path)

refined_files = list(refined_dir.rglob("*.obj"))

print("STEP 2.4: RESAMPLING CHECK")
print("-" * 50)
print(f"Shapes in processing log: {len(full)}")
print(f"Refined OBJ files found: {len(refined_files)}")

print("\nProcessing actions:")
display(full["action"].value_counts().to_frame("Number of shapes"))

print("\nProcessing status:")
display(full["status"].value_counts(dropna=False).to_frame("Number of shapes"))

vertex_summary = full[
    ["v_raw", "v_clean", "v_out"]
].describe().round(2)

print("\nVertex count statistics:")
display(vertex_summary)

# Check the target range.
LOWER_LIMIT = 5000
UPPER_LIMIT = 10000

inside_range = full["v_out"].between(
    LOWER_LIMIT, UPPER_LIMIT
)

print("\nTarget-range check:")
print(f"Shapes within range: {inside_range.sum()} / {len(full)}")
print(f"Shapes outside range: {(~inside_range).sum()}")

face_summary = full["f_out"].describe().round(2)

print("\nProcessed face count statistics:")
display(face_summary)

refined_stats = full.copy()

refined_stats.to_csv(
    DATA_OUT / "step2_4_refined_database_statistics.csv",
    index=False
)

vertex_summary.to_csv(
    DATA_OUT / "step2_4_vertex_summary.csv"
)

face_summary.to_csv(
    DATA_OUT / "step2_4_face_summary.csv"
)

print("\nStatistics saved successfully.")