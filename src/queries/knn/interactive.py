from pathlib import Path
import pandas as pd
from geopandas import GeoDataFrame
from shapely.geometry import Point

# Import necessary functions from query.py (adjust relative path if needed)
from .query import (
    run_knn_query_on_parquet,
    run_knn_query_on_segments,
    run_knn_query_early_filter,
    compare_methods
)

def run_knn_general_interactive():
    base_path = Path("data/processed/trajectories.parquet")
    fixed_path = Path("data/processed/trajectory_segments_fixed_knn.parquet")
    grid_path = Path("data/processed/trajectory_segments_grid_knn.parquet")

    print("Enter coordinates for reference point:")
    lat = float(input("  Latitude eg. 39.9800: "))
    lon = float(input("  Longitude eg. 116.3200: "))
    k = int(input("  Number of nearest points (k): "))

    save_choice = input("Save to GeoJSON (y/n)? ").strip().lower()
    should_save = save_choice == "y"

    # Standard kNN search for Base, Fixed, and Grid datasets
    inputs = [
        ("Base Parquet", base_path, run_knn_query_on_parquet, "lat", "lon"),
        ("Fixed Segments", fixed_path, run_knn_query_on_segments, "centroid_lat", "centroid_lon"),
        ("Grid Segments", grid_path, run_knn_query_on_segments, "centroid_lat", "centroid_lon"),
    ]

    for label, path, method, lat_col, lon_col in inputs:
        print(f"\n{'='*15} kNN Results ({label}) {'='*15}")
        df = pd.read_parquet(path)
        results = method(df, (lat, lon), k, lat_col, lon_col) if method == run_knn_query_on_segments else method(path, (lat, lon), k)

        if not results.empty:
            print(f"\nTop {k} results for {label}:\n")
            print(results.to_string(index=False))

            # Optional GeoJSON export
            if should_save:
                geojson_path = Path(f"data/results/knn_{label.replace(' ', '_').lower()}.geojson")
                results_gdf = results.copy()
                results_gdf["geometry"] = [Point(lon_val, lat_val) for lat_val, lon_val in zip(results[lat_col], results[lon_col])]
                gdf = GeoDataFrame(results_gdf, geometry="geometry", crs="EPSG:4326")
                gdf.to_file(geojson_path, driver="GeoJSON")
                print(f"Results saved to: {geojson_path}")
        else:
            print(f"No results found for {label}.")

    # Early Filtering: BBox filter followed by distance-based refinement
    print(f"\n{'='*15} kNN Results (Base Parquet - Early Filtering) {'='*15}")
    df_base = pd.read_parquet(base_path)
    num_base, early_base, time_base = run_knn_query_early_filter(df_base, (lat, lon), k, "lat", "lon")
    if not early_base.empty:
        early_base["traj_id"] = df_base.loc[early_base.index, "traj_id"].values
        early_base["lat"] = df_base.loc[early_base.index, "lat"].values
        early_base["lon"] = df_base.loc[early_base.index, "lon"].values
        print(f"\nTop {k} results for Base Parquet - Early Filtering:\n")
        print(early_base[["traj_id", "lat", "lon", "distance"]].to_string(index=False))
    else:
        print("No results found.")

    print(f"\n{'='*15} kNN Results (Fixed Segments - Early Filtering) {'='*15}")
    df_fixed = pd.read_parquet(fixed_path)
    num_fixed, early_fixed, time_fixed = run_knn_query_early_filter(df_fixed, (lat, lon), k, "centroid_lat", "centroid_lon", delta=70.0)
    if not early_fixed.empty:
        early_fixed["entity_id"] = df_fixed.loc[early_fixed.index, "entity_id"].values
        early_fixed["centroid_lat"] = df_fixed.loc[early_fixed.index, "centroid_lat"].values
        early_fixed["centroid_lon"] = df_fixed.loc[early_fixed.index, "centroid_lon"].values
        print(f"\nTop {k} results for Fixed Segments - Early Filtering:\n")
        print(early_fixed[["entity_id", "centroid_lat", "centroid_lon", "distance"]].to_string(index=False))
    else:
        print("No results found.")

    print(f"\n{'='*15} kNN Results (Grid Segments - Early Filtering) {'='*15}")
    df_grid = pd.read_parquet(grid_path)
    num_grid, early_grid, time_grid = run_knn_query_early_filter(df_grid, (lat, lon), k, "centroid_lat", "centroid_lon", delta=70.0)
    if not early_grid.empty:
        early_grid["entity_id"] = df_grid.loc[early_grid.index, "entity_id"].values
        early_grid["centroid_lat"] = df_grid.loc[early_grid.index, "centroid_lat"].values
        early_grid["centroid_lon"] = df_grid.loc[early_grid.index, "centroid_lon"].values
        print(f"\nTop {k} results for Grid Segments - Early Filtering:\n")
        print(early_grid[["entity_id", "centroid_lat", "centroid_lon", "distance"]].to_string(index=False))
    else:
        print("No results found.")

    # Show filtering summary and timings
    print(f"\n{'='*15} Early Filtering Candidate Summary {'='*15}\n")
    print(f"[Base Parquet - Early Filtering]")
    print(f"Candidates after bounding box filter: {num_base}")
    print(f"Filtering Time: {time_base:.6f} sec\n")
    print(f"[Fixed Segments - Early Filtering]")
    print(f"Candidates after bounding box filter: {num_fixed}")
    print(f"Filtering Time: {time_fixed:.6f} sec\n")
    print(f"[Grid Segments - Early Filtering]")
    print(f"Candidates after bounding box filter: {num_grid}")
    print(f"Filtering Time: {time_grid:.6f} sec\n")

    # Compare full execution times across all formats
    print(f"\n{'='*15} Execution Time Comparison Table {'='*15}\n")
    df_summary = compare_methods(
        base_path,
        fixed_path,
        grid_path,
        (lat, lon),
        k,
        delta_base=1.0,
        delta_fixed=70.0,
        delta_grid=70.0
    )
    print(df_summary.to_string(index=False))
    print(f"\n{'='*62}\n")
