from pathlib import Path
from time import time
from typing import Tuple
from .geoparquet_utils import load_geoparquet, run_bbox_query_geoparquet

def evaluate_geoparquet(path: str, bbox: Tuple[float, float, float, float]) -> None:
    """
    Run and evaluate a bounding box query on a GeoParquet file, with summary and timing information.
    """
    print("\n--- GeoParquet Evaluation ---")
    print(f"Target file: {path}")
    
    total_start = time()

    # Step 1: Load file
    print("Step 1: Loading GeoParquet file...")
    start_load = time()
    gdf = load_geoparquet(path)
    load_time = time() - start_load
    print(f"File loaded in {load_time:.3f} seconds.")

    # Step 2: Apply BBox query
    print("Step 2: Executing bounding box query...")
    start_query = time()
    results = run_bbox_query_geoparquet(gdf, bbox)
    query_time = time() - start_query
    print(f"Query completed in {query_time:.3f} seconds.")

    total_time = time() - total_start

    # Step 3: Output summary
    print("\n--- Summary ---")
    print(f"{'Metric':<25} {'Value'}")
    print("-" * 40)
    print(f"{'Number of matches:':<25} {len(results)}")
    print(f"{'Load time (s):':<25} {load_time:.3f}")
    print(f"{'Query time (s):':<25} {query_time:.3f}")
    print(f"{'Total time (s):':<25} {total_time:.3f}")

    # Step 4: Optionally save results
    choice = input("\nDo you want to save the results? (geojson / none): ").strip().lower()
    if choice == "geojson":
        output_path = Path("data/results/geoparquet_bbox_results.geojson")
        results.to_file(output_path, driver="GeoJSON")
        print(f"Results saved to: {output_path}")
    else:
        print("Results were not saved.")

def run_geoparquet_interactive():
    """
    Interactively ask for BBox coordinates and run the GeoParquet evaluation.
    Intended to be called from the main interface.
    """
    print("Enter bounding box coordinates for GeoParquet:")
    min_lat = float(input("  Min Latitude eg. 39.9840: "))
    max_lat = float(input("  Max Latitude eg. 39.9850: "))
    min_lon = float(input("  Min Longitude eg. 116.3160: "))
    max_lon = float(input("  Max Longitude eg. 116.3185: "))
    bbox = (min_lat, max_lat, min_lon, max_lon)

    path = "data/processed/trajectories_geoparquet.parquet"
    evaluate_geoparquet(path, bbox)
