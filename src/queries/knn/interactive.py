#!/usr/bin/env python3
"""
interactive.py

Interactive CLI for running kNN queries on:
  1) Base Parquet (raw trajectory points)
  2) Fixed segments
  3) Grid segments

Implements two‐phase pipeline for segments (no centroids), mirroring
the original TrajParquet Java implementation.
"""

from pathlib import Path
import time
import pandas as pd
from geopandas import GeoDataFrame
from shapely.geometry import Point, LineString

from .query import (
    run_knn_query_on_parquet,
    run_knn_query_on_segments,
    run_knn_query_early_filter,
    compare_methods
)


def run_knn_general_interactive():
    base_path = Path("data/processed/trajectories.parquet")
    fixed_path = Path("data/processed/trajectory_segments_fixed.parquet")
    grid_path = Path("data/processed/trajectory_segments_grid_knn.parquet")

    print("Enter coordinates for reference point:")
    try:
        lat = float(input("  Latitude eg. 39.9800: ").strip())
        lon = float(input("  Longitude eg. 116.3200: ").strip())
        k = int(input("  Number of nearest points (k): ").strip())
    except ValueError:
        print("Invalid input; please enter numeric values.")
        return

    save_choice = input("Save standard results to GeoJSON (y/n)? ").strip().lower()
    should_save = (save_choice == "y")

    # --- Standard kNN queries ---
    # 1) Base Parquet
    print(f"\n{'='*15} kNN Results (Base Parquet) {'='*15}")
    base_results = run_knn_query_on_parquet(str(base_path), (lat, lon), k)
    if not base_results.empty:
        print(base_results.to_string(index=False))
        if should_save:
            geojson_path = Path(f"data/results/knn_base_parquet.geojson")
            gdf = GeoDataFrame(
                base_results.assign(
                    geometry=[Point(lon_, lat_) for lat_, lon_ in zip(base_results["lat"], base_results["lon"])]
                ),
                crs="EPSG:4326"
            )
            gdf.to_file(geojson_path, driver="GeoJSON")
            print(f"Saved GeoJSON: {geojson_path}")
    else:
        print("No results found for Base Parquet.")

    # 2) Fixed Segments
    print(f"\n{'='*15} kNN Results (Fixed Segments) {'='*15}")
    df_fixed = pd.read_parquet(fixed_path)
    fixed_results = run_knn_query_on_segments(
        df_fixed,
        (lat, lon),
        k,
        start_lat_col="min_x",   
        start_lon_col="min_y",
        end_lat_col  ="max_x",
        end_lon_col  ="max_y",
        grid_cell_col=None
    )
    if not fixed_results.empty:
        print(fixed_results.to_string(index=False))
        if should_save:
            geojson_path = Path(f"data/results/knn_fixed_segments.geojson")
            gdf = GeoDataFrame(
                fixed_results.assign(
                    geometry=[
                        LineString([
                            (row["min_x"], row["min_y"]),
                            (row["max_x"], row["max_y"])
                        ]) for _, row in fixed_results.iterrows()
                    ]
                ),
                crs="EPSG:4326"
            )
            gdf.to_file(geojson_path, driver="GeoJSON")
            print(f"Saved GeoJSON: {geojson_path}")
    else:
        print("No results found for Fixed Segments.")

    # 3) Grid Segments
    print(f"\n{'='*15} kNN Results (Grid Segments) {'='*15}")
    df_grid = pd.read_parquet(grid_path)
    grid_results = run_knn_query_on_segments(
        df_grid,
        (lat, lon),
        k,
        start_lat_col="min_x",
        start_lon_col="min_y",
        end_lat_col  ="max_x",
        end_lon_col  ="max_y",
        grid_cell_col="grid_cell",
        cell_size=0.001,
        grid_ring=1
    )
    if not grid_results.empty:
        print(grid_results.to_string(index=False))
        if should_save:
            geojson_path = Path(f"data/results/knn_grid_segments.geojson")
            gdf = GeoDataFrame(
                grid_results.assign(
                    geometry=[
                        LineString([
                            (row["min_x"], row["min_y"]),
                            (row["max_x"], row["max_y"])
                        ]) for _, row in grid_results.iterrows()
                    ]
                ),
                crs="EPSG:4326"
            )
            gdf.to_file(geojson_path, driver="GeoJSON")
            print(f"Saved GeoJSON: {geojson_path}")
    else:
        print("No results found for Grid Segments.")

    # --- Early Filtering kNN queries ---
    print(f"\n{'='*15} kNN Results (Base Parquet - Early Filtering) {'='*15}")
    df_base = pd.read_parquet(base_path)
    num_base, early_base, time_base = run_knn_query_early_filter(
        df_base, (lat, lon), k, lat_col="lat", lon_col="lon"
    )
    if not early_base.empty:
        early_base = early_base.assign(
            traj_id=df_base.loc[early_base.index, "traj_id"].values,
            lat=df_base.loc[early_base.index, "lat"].values,
            lon=df_base.loc[early_base.index, "lon"].values
        )
        print(early_base[["traj_id", "lat", "lon", "distance"]].to_string(index=False))
    else:
        print("No results found for Base Parquet - Early Filtering.")

    print(f"\n{'='*15} kNN Results (Fixed Segments - Early Filtering) {'='*15}")
    # manual early filtering for segments:
    delta = 0.01
    t0 = time.time()
    mask_fixed = (
        ((df_fixed["min_x"] >= lat - delta) & (df_fixed["min_x"] <= lat + delta) &
        (df_fixed["min_y"] >= lon - delta) & (df_fixed["min_y"] <= lon + delta))
        |
        ((df_fixed["max_x"] >= lat - delta) & (df_fixed["max_x"] <= lat + delta) &
        (df_fixed["max_y"] >= lon - delta) & (df_fixed["max_y"] <= lon + delta))
    )
    filtered_fixed = df_fixed[mask_fixed].copy()
    time_fixed = time.time() - t0
    num_fixed = len(filtered_fixed)
    print(f"[Early Filtering] Candidates after filter: {num_fixed}")
    if num_fixed:
        early_fixed = run_knn_query_on_segments(
            filtered_fixed, (lat, lon), k,
            start_lat_col="min_x", start_lon_col="min_y",
            end_lat_col  ="max_x", end_lon_col  ="max_y",
            # grid_resolution=0.01,
            grid_ring=1
        )
        print(early_fixed[["entity_id", "min_y", "min_x", "max_y", "max_x", "distance"]].to_string(index=False))
    else:
        print("No results found for Fixed Segments - Early Filtering.")

    print(f"\n{'='*15} kNN Results (Grid Segments - Early Filtering) {'='*15}")
    t0 = time.time()
    mask_grid = (
        ((df_grid["min_x"] >= lat - delta) & (df_grid["min_x"] <= lat + delta) &
        (df_grid["min_y"] >= lon - delta) & (df_grid["min_y"] <= lon + delta))
        |
        ((df_grid["max_x"] >= lat - delta) & (df_grid["max_x"] <= lat + delta) &
        (df_grid["max_y"] >= lon - delta) & (df_grid["max_y"] <= lon + delta))
    )
    filtered_grid = df_grid[mask_grid].copy()
    time_grid = time.time() - t0
    num_grid = len(filtered_grid)
    print(f"[Early Filtering] Candidates after filter: {num_grid}")
    if num_grid:
        early_grid = run_knn_query_on_segments(
            filtered_grid,               # <- τα segments που πέρασαν το bbox-mask
            (lat, lon), k,
            start_lat_col="min_x",       # lat  = min_x
            start_lon_col="min_y",       # lon  = min_y
            end_lat_col="max_x",
            end_lon_col="max_y",
            grid_ring=1                  # hash-index δεν χρησιμοποιείται εδώ
        )
        print(
            early_grid[[
                "entity_id", "min_y", "min_x", "max_y", "max_x", "distance"
            ]].to_string(index=False)
        )
    else:
        print("No results found for Grid Segments - Early Filtering.")

    # --- Early Filtering Summary ---
    print(f"\n{'='*15} Early Filtering Candidate Summary {'='*15}\n")
    print(f"[Base Parquet]     Candidates: {num_base}, Time: {time_base:.6f}s")
    print(f"[Fixed Segments]   Candidates: {num_fixed}, Time: {time_fixed:.6f}s")
    print(f"[Grid Segments]    Candidates: {num_grid}, Time: {time_grid:.6f}s")

    # --- Compare methods summary ---
    # print(f"\n{'='*15} Execution Time Comparison Table {'='*15}\n")
    # summary_df = compare_methods(
    #     str(base_path),
    #     str(fixed_path),
    #     str(grid_path),
    #     (lat, lon),
    #     k,
    #     delta_base=1.0,
    #     delta_fixed=0.01,
    #     delta_grid=0.01
    # )
    # print(summary_df.to_string(index=False))
    # print(f"\n{'='*62}\n")


if __name__ == "__main__":
    run_knn_general_interactive()
