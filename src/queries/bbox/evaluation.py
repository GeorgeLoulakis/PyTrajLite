import pandas as pd
from pathlib import Path
from time import time
from typing import Tuple

from .loaders import *
from .queries import *
from .utils import *

def evaluate_all_files(bbox: Tuple[float, float, float, float]):
    """
    Load and evaluate the bounding box query on all four data formats:
    - Measures query time per file
    - Computes number of matching records
    - Compares relative difference to a reference file (first successful one)
    - Optionally saves results in CSV or JSON format
    """
    base_parquet_path = Path("data/processed/trajectories.parquet")
    csv_path = Path("data/processed/trajectories.csv")
    seg_fixed_path = Path("data/processed/trajectory_segments_fixed.parquet")
    seg_grid_path = Path("data/processed/trajectory_segments_grid.parquet")

    files = {
        # Base / CSV
        "Base Parquet": (load_base_parquet, run_bbox_query_on_points, base_parquet_path),
        "Base Parquet (Pushdown)": (lambda p: load_base_parquet_with_pushdown(p, bbox), run_bbox_query_on_points, base_parquet_path),
        "CSV File": (load_csv, run_bbox_query_on_points, csv_path),

        # Fixed-size segments
        "Fixed Segments (Optimized)": (lambda p: p, run_bbox_query_on_segments_optimized, seg_fixed_path),
        "Fixed Segments (Pushdown v3 NP)": (lambda p: load_segmented_parquet_with_pushdown_v3_np(p, bbox), run_bbox_query_on_segments_numpy2_optimized_v2, seg_fixed_path),
        "Fixed Segments (Memory Optimized Batch)": (lambda p: load_segmented_parquet_with_pushdown_v3_np(p, bbox), lambda df, bbox: memory_optimized_batch_query(df, bbox, batch_size=50000), seg_fixed_path),
        "Fixed Segments (Pushdown Optimized)": (lambda p: load_segmented_parquet_with_pushdown_optimized(p, bbox), run_bbox_query_on_segments_numpy2_optimized, seg_fixed_path),

        # Grid-based segments
        "Grid Segments (Optimized)": (lambda p: p, run_bbox_query_on_segments_optimized, seg_grid_path),
        "Grid Segments (Pushdown v3 NP)": (lambda p: load_segmented_parquet_with_pushdown_v3_np(p, bbox), run_bbox_query_on_segments_numpy2_optimized_v2, seg_grid_path),
        "Grid Segments (Memory Optimized Batch)": (lambda p: load_segmented_parquet_with_pushdown_v3_np(p, bbox), lambda df, bbox: memory_optimized_batch_query(df, bbox, batch_size=50000), seg_grid_path),
        "Grid Segments (Pushdown Optimized)": (lambda p: load_segmented_parquet_with_pushdown_optimized(p, bbox), run_bbox_query_on_segments_numpy2_optimized, seg_grid_path),
    }


    # Ask user if they want to save results and in what format
    save_format = input("\nSave results? Choose format (csv / json / none): ").strip().lower()
    save_results = save_format in {"csv", "json"}

    reference_count = None
    summary = []

    # Find next bbox_try# folder
    base_results_dir = Path("results")
    base_results_dir.mkdir(exist_ok=True)
    existing = sorted([d for d in base_results_dir.glob("bbox_try*") if d.is_dir()])
    if existing:
        last_index = max(int(d.name.replace("bbox_try", "")) for d in existing if d.name.replace("bbox_try", "").isdigit())
        run_index = last_index + 1
    else:
        run_index = 1

    output_dir = base_results_dir / f"bbox_try{run_index}"
    output_dir.mkdir(parents=True)
    print(f"\nSaving results to folder: {output_dir}")
    total_methods = len(files)
    for i, (name, (load_fn, query_fn, path)) in enumerate(files.items()):
        if not path.exists():
            print(f"[{name}] File not found: {path}")
            continue

        try:
            load_start = time()
            df = load_fn(path)
            load_time = time() - load_start

            query_start = time()
            results = query_fn(df, bbox)
            query_time = time() - query_start

            print(f"→ Refinement applied to: {name}")
            if "Base Parquet" in name or "CSV" in name:
                # already has points, just drop duplicates
                results = results.drop_duplicates(subset=["traj_id", "lat", "lon"])
            else:
                results = extract_points_from_segments(results, bbox)



            elapsed = load_time + query_time


            match_count = len(results)

            if reference_count is None:
                reference_count = match_count
            
            print(f"[{name}] {match_count} matches in {elapsed:.3f} sec")

            summary.append((name, match_count, load_time, query_time, elapsed))

            if save_results and not results.empty:
                safe_name = name.lower().replace(" ", "_").replace("(", "").replace(")", "").replace("-", "_")
                output_path = output_dir / f"{safe_name}_bbox_results.{save_format}"
                progress_percent = (i + 1) / total_methods * 100
                print(f"Saving: {output_path.name} ({i + 1}/{total_methods} — {progress_percent:.1f}%)")

                # If results are empty, skip saving
                if not all(col in results.columns for col in ["traj_id", "lat", "lon", "timestamp"]):
                    if "vals_x" in results.columns and "vals_y" in results.columns:
                        from numpy import fromstring

                        new_rows = []
                        for _, row in results.iterrows():
                            try:
                                xs = fromstring(str(row["vals_x"]).replace("\n", " ").replace("  ", " ").replace("[", "").replace("]", ""), sep=' ')
                                ys = fromstring(str(row["vals_y"]).replace("\n", " ").replace("  ", " ").replace("[", "").replace("]", ""), sep=' ')
                                for lat, lon in zip(xs, ys):
                                    new_rows.append({
                                        "traj_id": -1,
                                        "lat": round(lat, 6),
                                        "lon": round(lon, 6),
                                        "timestamp": "unknown"
                                    })
                            except Exception:
                                continue
                        results = pd.DataFrame(new_rows)

                if save_format == "csv":
                    results.to_csv(output_path, index=False)
                elif save_format == "json":
                    results.to_json(output_path, orient="records", indent=2)


        except Exception as e:
            print(f"[{name}] Error during evaluation: {e}")

    # Print a summary table by category
    print("\n--- Summary ---")
    df_summary = pd.DataFrame(summary, columns=["Format", "Points", "Load (s)", "Query (s)", "Total (s)"])

    def classify_category(format_name):
        if "Base Parquet" in format_name:
            return "Base Parquet"
        elif "CSV" in format_name:
            return "CSV"
        elif "Fixed Segments" in format_name:
            return "Fixed Segments"
        elif "Grid Segments" in format_name:
            return "Grid Segments"
        else:
            return "Other"

    df_summary["Category"] = df_summary["Format"].apply(classify_category)
    df_summary = df_summary.sort_values(by=["Category", "Total (s)"])

    grouped = df_summary.groupby("Category")

    for category, group in grouped:
        print(f"\n[{category}]")
        print(f"{'Format':<40} {'Matches':>10} {'Load (s)':>10} {'Query (s)':>12} {'Total (s)':>12}")
        print("-" * 95)
        for _, row in group.iterrows():
            print(f"{row['Format']:<40} {int(row['Points']):10} {row['Load (s)']:10.3f} {row['Query (s)']:12.3f} {row['Total (s)']:12.3f}")

def run_bbox_evaluation():
    """
    Entry point for running the BBox query with user-defined coordinates.
    """
    print("Enter bounding box coordinates:")
    min_lat = float(input("  Min Latitude eg. 39.9840: "))
    max_lat = float(input("  Max Latitude eg. 39.9850: "))
    min_lon = float(input("  Min Longitude eg. 116.3160: "))
    max_lon = float(input("  Max Longitude eg. 116.3185: "))
    bbox = (min_lat, max_lat, min_lon, max_lon)
    evaluate_all_files(bbox)

# not used in the current implementation

# files = {
#     "Base Parquet": (load_base_parquet, run_bbox_query_on_points, base_parquet_path),
#     "Base Parquet (Optimized)": (load_base_parquet_bbox_only, run_bbox_query_on_points, base_parquet_path),
#     "Base Parquet (Pushdown)": (lambda p: load_base_parquet_with_pushdown(p, bbox), run_bbox_query_on_points, base_parquet_path),
#     "CSV File": (load_csv, run_bbox_query_on_points, csv_path),

#     "Fixed Segments (Default)": (load_segmented_parquet, run_bbox_query_on_segments, seg_fixed_path),
#     "Fixed Segments (NumPy)": (load_segmented_parquet, run_bbox_query_on_segments_numpy, seg_fixed_path),
#     "Fixed Segments (NumPy V2 Raw)": (load_segmented_parquet, run_bbox_query_on_segments_numpy2, seg_fixed_path),
#     "Fixed Segments (Numpy v2 Combined)": (load_segmented_parquet,run_bbox_query_on_segments_numpy_v2,seg_fixed_path),
#     "Fixed Segments (Optimized)": (lambda p: p, run_bbox_query_on_segments_optimized, seg_fixed_path),
#     "Fixed Segments (Pushdown)": (lambda p: load_segmented_parquet_with_pushdown(p, bbox), run_bbox_query_on_segments_numpy2, seg_fixed_path),
#     "Fixed Segments (Pushdown Optimized)": (lambda p: load_segmented_parquet_with_pushdown_optimized(p, bbox),run_bbox_query_on_segments_numpy2_optimized,seg_fixed_path),
#     "Fixed Segments (Pushdown v2)": (lambda p: load_segmented_parquet_with_pushdown_optimized_v2(p, bbox),run_bbox_query_on_segments_numpy2_optimized_v2,seg_fixed_path),
#     "Fixed Segments (Pushdown v3 NP)": (lambda p: load_segmented_parquet_with_pushdown_v3_np(p, bbox), run_bbox_query_on_segments_numpy2_optimized_v2, seg_fixed_path),
#     "Fixed Segments (Vectorized)": (lambda p: load_segmented_parquet_with_pushdown_v3_np(p, bbox),lambda df, bbox: vectorized_bbox_query(df, bbox),seg_fixed_path),
#     "Fixed Segments (Memory Optimized Batch)": (lambda p: load_segmented_parquet_with_pushdown_v3_np(p, bbox),lambda df, bbox: memory_optimized_batch_query(df, bbox, batch_size=50000),seg_fixed_path),


#     "Grid Segments (Default)": (load_segmented_parquet, run_bbox_query_on_segments, seg_grid_path),
#     "Grid Segments (NumPy)": (load_segmented_parquet, run_bbox_query_on_segments_numpy, seg_grid_path),
#     "Grid Segments (NumPy V2 Raw)": (load_segmented_parquet, run_bbox_query_on_segments_numpy2, seg_grid_path),
#     "Grid Segments (Numpy v2 Combined)": (load_segmented_parquet,run_bbox_query_on_segments_numpy_v2,seg_grid_path),
#     "Grid Segments (Optimized)": (lambda p: p, run_bbox_query_on_segments_optimized, seg_grid_path),
#     "Grid Segments (Pushdown)": (lambda p: load_segmented_parquet_with_pushdown(p, bbox), run_bbox_query_on_segments_numpy2, seg_grid_path),
#     "Grid Segments (Pushdown Optimized)": (lambda p: load_segmented_parquet_with_pushdown_optimized(p, bbox),run_bbox_query_on_segments_numpy2_optimized,seg_grid_path),
#     "Grid Segments (Pushdown v2)": (lambda p: load_segmented_parquet_with_pushdown_optimized_v2(p, bbox),run_bbox_query_on_segments_numpy2_optimized_v2,seg_grid_path),
#     "Grid Segments (Pushdown v3 NP)": (lambda p: load_segmented_parquet_with_pushdown_v3_np(p, bbox), run_bbox_query_on_segments_numpy2_optimized_v2, seg_grid_path),
#     "Grid Segments (Vectorized)": (lambda p: load_segmented_parquet_with_pushdown_v3_np(p, bbox),lambda df, bbox: vectorized_bbox_query(df, bbox),seg_grid_path),
#     "Grid Segments (Memory Optimized Batch)": (lambda p: load_segmented_parquet_with_pushdown_v3_np(p, bbox),lambda df, bbox: memory_optimized_batch_query(df, bbox, batch_size=50000),seg_grid_path)
# }
