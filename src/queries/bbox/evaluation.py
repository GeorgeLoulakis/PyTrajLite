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
        "Base Parquet": (load_base_parquet, run_bbox_query_on_points, base_parquet_path),
        "Base Parquet (Optimized)": (load_base_parquet_bbox_only, run_bbox_query_on_points, base_parquet_path),
        "Base Parquet (Pushdown)": (lambda p: load_base_parquet_with_pushdown(p, bbox), run_bbox_query_on_points, base_parquet_path),
        "CSV File": (load_csv, run_bbox_query_on_points, csv_path),

        # Fixed-size segments (11 versions)
        "Fixed Segments (Default)": (load_segmented_parquet, run_bbox_query_on_segments, seg_fixed_path),
        "Fixed Segments (NumPy)": (load_segmented_parquet, run_bbox_query_on_segments_numpy, seg_fixed_path),
        "Fixed Segments (NumPy V2 Raw)": (load_segmented_parquet, run_bbox_query_on_segments_numpy2, seg_fixed_path),
        "Fixed Segments (Numpy v2 Combined)": (load_segmented_parquet,run_bbox_query_on_segments_numpy_v2,seg_fixed_path),
        "Fixed Segments (Optimized)": (lambda p: p, run_bbox_query_on_segments_optimized, seg_fixed_path),
        "Fixed Segments (Pushdown)": (lambda p: load_segmented_parquet_with_pushdown(p, bbox), run_bbox_query_on_segments_numpy2, seg_fixed_path),
        "Fixed Segments (Pushdown Optimized)": (lambda p: load_segmented_parquet_with_pushdown_optimized(p, bbox),run_bbox_query_on_segments_numpy2_optimized,seg_fixed_path),
        "Fixed Segments (Pushdown v2)": (lambda p: load_segmented_parquet_with_pushdown_optimized_v2(p, bbox),run_bbox_query_on_segments_numpy2_optimized_v2,seg_fixed_path),
        "Fixed Segments (Pushdown v3 NP)": (lambda p: load_segmented_parquet_with_pushdown_v3_np(p, bbox), run_bbox_query_on_segments_numpy2_optimized_v2, seg_fixed_path),
        "Fixed Segments (Vectorized)": (lambda p: load_segmented_parquet_with_pushdown_v3_np(p, bbox),lambda df, bbox: vectorized_bbox_query(df, bbox),seg_fixed_path),
        "Fixed Segments (Memory Optimized Batch)": (lambda p: load_segmented_parquet_with_pushdown_v3_np(p, bbox),lambda df, bbox: memory_optimized_batch_query(df, bbox, batch_size=50000),seg_fixed_path),



        # Grid-based segments (11 versions)
        "Grid Segments (Default)": (load_segmented_parquet, run_bbox_query_on_segments, seg_grid_path),
        "Grid Segments (NumPy)": (load_segmented_parquet, run_bbox_query_on_segments_numpy, seg_grid_path),
        "Grid Segments (NumPy V2 Raw)": (load_segmented_parquet, run_bbox_query_on_segments_numpy2, seg_grid_path),
        "Grid Segments (Numpy v2 Combined)": (load_segmented_parquet,run_bbox_query_on_segments_numpy_v2,seg_grid_path),
        "Grid Segments (Optimized)": (lambda p: p, run_bbox_query_on_segments_optimized, seg_grid_path),
        "Grid Segments (Pushdown)": (lambda p: load_segmented_parquet_with_pushdown(p, bbox), run_bbox_query_on_segments_numpy2, seg_grid_path),
        "Grid Segments (Pushdown Optimized)": (lambda p: load_segmented_parquet_with_pushdown_optimized(p, bbox),run_bbox_query_on_segments_numpy2_optimized,seg_grid_path),
        "Grid Segments (Pushdown v2)": (lambda p: load_segmented_parquet_with_pushdown_optimized_v2(p, bbox),run_bbox_query_on_segments_numpy2_optimized_v2,seg_grid_path),
        "Grid Segments (Pushdown v3 NP)": (lambda p: load_segmented_parquet_with_pushdown_v3_np(p, bbox), run_bbox_query_on_segments_numpy2_optimized_v2, seg_grid_path),
        "Grid Segments (Vectorized)": (lambda p: load_segmented_parquet_with_pushdown_v3_np(p, bbox),lambda df, bbox: vectorized_bbox_query(df, bbox),seg_grid_path),
        "Grid Segments (Memory Optimized Batch)": (lambda p: load_segmented_parquet_with_pushdown_v3_np(p, bbox),lambda df, bbox: memory_optimized_batch_query(df, bbox, batch_size=50000),seg_grid_path)
    }


    # Ask user if they want to save results and in what format
    save_format = input("\nSave results? Choose format (csv / json / none): ").strip().lower()
    save_results = save_format in {"csv", "json"}

    reference_count = None
    summary = []
    output_dir = Path("data/results")
    output_dir.mkdir(parents=True, exist_ok=True)

    for name, (load_fn, query_fn, path) in files.items():
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

            # Apply refinement only if MBR-only query (no NumPy-level check)
            REFINABLE_FORMATS = {
                "Fixed Segments (Default)",
                "Fixed Segments (Optimized)",
                "Grid Segments (Default)",
                "Grid Segments (Optimized)"
            }

            if name in REFINABLE_FORMATS:
                print(f"→ Refinement applied to: {name}")
                results = refine_bbox_candidates(results, bbox)

            elapsed = load_time + query_time


            match_count = len(results)

            if reference_count is None:
                reference_count = match_count
                percent_diff = 0.0
            else:
                percent_diff = 100 * (match_count - reference_count) / reference_count

            print(f"[{name}] {match_count} matches in {elapsed:.3f} sec ({percent_diff:+.1f}% diff)")

            summary.append((name, match_count, load_time, query_time, elapsed, percent_diff))

            if save_results and not results.empty:
                safe_name = name.lower().replace(" ", "_").replace("(", "").replace(")", "").replace("-", "_")
                output_path = output_dir / f"{safe_name}_bbox_results.{save_format}"
                if save_format == "csv":
                    results.to_csv(output_path, index=False)
                elif save_format == "json":
                    results.to_json(output_path, orient="records", indent=2)

        except Exception as e:
            print(f"[{name}] Error during evaluation: {e}")

    # Print a summary table by category
    print("\n--- Summary ---")
    df_summary = pd.DataFrame(summary, columns=["Format", "Matches", "Load (s)", "Query (s)", "Total (s)", "% Diff"])

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
        print(f"{'Format':<40} {'Matches':>10} {'Load (s)':>10} {'Query (s)':>12} {'Total (s)':>12} {'% Diff':>10}")
        print("-" * 95)
        for _, row in group.iterrows():
            print(f"{row['Format']:<40} {int(row['Matches']):10} {row['Load (s)']:10.3f} {row['Query (s)']:12.3f} {row['Total (s)']:12.3f} {row['% Diff']:10.1f}")

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
