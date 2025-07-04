from shapely.geometry import box
from pathlib import Path
from time import time
from typing import Tuple
from datetime import datetime
from .geoparquet_utils import load_geoparquet, run_bbox_query_geoparquet

def evaluate_geoparquet(path: str, bbox: Tuple[float, float, float, float]) -> None:
    """
    Run and evaluate a bounding box query on a GeoParquet file,
    keeping all available attributes, removing only truly identical rows,
    and exporting the result to CSV if requested.
    """
    filename = Path(path).name
    print("\n--- GeoParquet Evaluation ---")
    print(f"Target file: {filename}")

    total_start = time()

    # Step 1: Load (only points within lon/lat bounds)
    print("Step 1: Loading GeoParquet file (filtered)...")
    start_load = time()
    gdf = load_geoparquet(path, bbox)
    load_time = time() - start_load
    print(f"File loaded in {load_time:.3f} seconds.")

    # Step 2: Exact spatial filter
    print("Step 2: Executing spatial filtering (including boundary)...")
    start_query = time()
    minx, miny, maxx, maxy = bbox
    bbox_poly = box(minx, miny, maxx, maxy)
    results = gdf[gdf.geometry.intersects(bbox_poly)]
    query_time = time() - start_query
    print(f"Query completed in {query_time:.3f} seconds.")
    print("Columns after spatial filter:", results.columns.tolist())

    # --- Step 3: Collapse per‐trajectory coordinate duplicates, as Base does ---
    print("Step 3: Removing duplicate (traj_id, lat, lon)...")
    before = len(results)
    results = results.drop_duplicates(subset=["traj_id","lat","lon"])
    after = len(results)
    print(f"Removed {before - after} duplicates. Remaining rows: {after}")

    total_time = time() - total_start

    # Step 4: Summary
    print("\n--- Summary ---\n")
    print(f"{'Metric':<30} {'Value'}")
    print("-" * 45)
    print(f"{'Total matched rows:':<30} {len(results)}")
    print(f"{'Load time (s):':<30} {load_time:.3f}")
    print(f"{'Query time (s):':<30} {query_time:.3f}")
    print(f"{'Total time (s):':<30} {total_time:.3f}")

    # Step 5: Optional CSV export
    choice = input("\nDo you want to save the results? (csv / none): ").strip().lower()
    if choice == "csv":
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = Path(f"data/results/geoparquet_bbox_results_{timestamp}.csv")

        # Save all columns except geometry
        columns_to_save = [col for col in results.columns if col != "geometry"]
        results.to_csv(output_path, index=False, columns=columns_to_save)

        print(f"CSV saved to: {output_path}")
    else:
        print("Results were not saved.")


def run_geoparquet_interactive():
    """
    Interactively ask for BBox coordinates and run the GeoParquet evaluation on both files.
    """
    print("Enter bounding box coordinates for GeoParquet:")
    min_lat = float(input("  Min Latitude eg. 39.9840: "))
    max_lat = float(input("  Max Latitude eg. 39.9850: "))
    min_lon = float(input("  Min Longitude eg. 116.3160: "))
    max_lon = float(input("  Max Longitude eg. 116.3185: "))
    bbox = (min_lon, min_lat, max_lon, max_lat)

    print("\n========================================")
    print("Evaluating: Compressed GeoParquet (Snappy)")
    evaluate_geoparquet("data/processed/trajectories_geoparquet_compressed_snappy.parquet", bbox)
    
    print("\n========================================")
    print("Evaluating: Uncompressed GeoParquet")
    evaluate_geoparquet("data/processed/trajectories_geoparquet_uncompressed.parquet", bbox)



