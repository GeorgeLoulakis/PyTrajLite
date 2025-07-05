import duckdb
import pyarrow.parquet as pq
import geopandas as gpd
from shapely.geometry import Point, box
from pathlib import Path
from time import time
from typing import Tuple, Dict
from datetime import datetime


def load_geoparquet(path: str, bbox: Tuple[float, float, float, float]) -> gpd.GeoDataFrame:
    """
    Load only the points within the bounding box directly from the GeoParquet file
    using DuckDB for efficient predicate pushdown. Returns a GeoDataFrame
    containing all original columns plus geometry.
    """
    minx, miny, maxx, maxy = bbox
    query = f"""
    SELECT *
    FROM read_parquet('{path}')
    WHERE lon BETWEEN {minx} AND {maxx}
      AND lat BETWEEN {miny} AND {maxy}
    """
    df = duckdb.query(query).to_df()
    return gpd.GeoDataFrame(
        df,
        geometry=gpd.points_from_xy(df.lon, df.lat),
        crs="EPSG:4326"
    )


def evaluate_geoparquet(path: str, bbox: Tuple[float, float, float, float], save_choice: str = None) -> Tuple[gpd.GeoDataFrame, Dict]:
    """
    Run and evaluate a bounding box query on a GeoParquet file,
    returns both the result GeoDataFrame and a metrics dict.
    If save_choice == 'csv', automatically saves the CSV for each run.
    """
    filename = Path(path).name
    total_start = time()

    # Step 1: Load
    start_load = time()
    gdf = load_geoparquet(path, bbox)
    load_time = time() - start_load

    # Step 2: Spatial filter (inclusive of boundary)
    start_query = time()
    minx, miny, maxx, maxy = bbox
    bbox_poly = box(minx, miny, maxx, maxy)
    results = gdf[gdf.geometry.intersects(bbox_poly)]
    query_time = time() - start_query

    # Step 3: Deduplicate per trajectory + location
    before = len(results)
    results = results.drop_duplicates(subset=["traj_id","lat","lon"])
    after = len(results)

    total_time = time() - total_start
    metrics = {
        'matches': after,
        'load_time': load_time,
        'query_time': query_time,
        'total_time': total_time
    }

    # Step 4: Optional save
    if save_choice == 'csv':
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_csv = Path(f"data/results/{filename.replace('.parquet','')}_{timestamp}.csv")
        results.drop(columns="geometry", errors="ignore").to_csv(out_csv, index=False)

    return results, metrics


def run_geoparquet_interactive():
    """
    Ask once whether to save results and display a single summary table
    for both Compressed and Uncompressed GeoParquet.
    """
    print("Enter bounding box coordinates for GeoParquet:")
    min_lat = float(input("  Min Latitude eg. 39.9840: "))
    max_lat = float(input("  Max Latitude eg. 39.9850: "))
    min_lon = float(input("  Min Longitude eg. 116.3160: "))
    max_lon = float(input("  Max Longitude eg. 116.3185: "))
    # Correct bbox ordering: (min_lon, min_lat, max_lon, max_lat)
    bbox = (min_lon, min_lat, max_lon, max_lat)

    save_choice = input("Do you want to save the results as CSV? (csv/none): ").strip().lower()

    runs = [
        ("Compressed", "data/processed/trajectories_geoparquet_compressed_snappy.parquet"),
        ("Uncompressed", "data/processed/trajectories_geoparquet_uncompressed.parquet")
    ]

    summary = []
    for label, path in runs:
        results, m = evaluate_geoparquet(path, bbox, save_choice)
        summary.append({
            'Format': label,
            'Matches': m['matches'],
            'Load (s)': f"{m['load_time']:.3f}",
            'Query (s)': f"{m['query_time']:.3f}",
            'Total (s)': f"{m['total_time']:.3f}"
        })

    # Print a single summary table
    print("\n=== Summary ===")
    header = f"{'Format':<15}{'Matches':<10}{'Load (s)':<12}{'Query (s)':<12}{'Total (s)':<12}"
    print(header)
    print('-'*len(header))
    for row in summary:
        print(f"{row['Format']:<15}{row['Matches']:<10}{row['Load (s)']:<12}{row['Query (s)']:<12}{row['Total (s)']:<12}")
