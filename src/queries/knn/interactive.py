#!/usr/bin/env python3
"""
interactive.py

Interactive CLI for running kNN queries on:
  1) Base Parquet (raw trajectory points)
  2) Fixed segments
  3) Grid segments

Implements two‑phase pipeline for segments (no centroids), mirroring
the original TrajParquet Java implementation.
"""

from pathlib import Path
import pandas as pd
import time

from .query import (
    run_knn_query,
    run_knn_query_on_parquet,
    run_knn_query_on_segments,
    run_knn_query_on_fixed_segments_vectorized,
)

import numpy as np


def run_knn_general_interactive():
    base_path  = Path("data/processed/trajectories.parquet")
    fixed_path = Path("data/processed/trajectory_segments_fixed_knn.parquet")
    grid_path  = Path("data/processed/trajectory_segments_grid_knn.parquet")

    # --- 1) User input ---
    try:
        lat = float(input("Latitude (e.g. 39.9800): ").strip())
        lon = float(input("Longitude (e.g. 116.3200): ").strip())
        k   = int(input("Number of nearest points (k): ").strip())
    except ValueError:
        print("Invalid input – please enter numeric values.")
        return

    save_choice = input("Do you want to save the results as CSV? (y/n): ").strip().lower()
    should_save = (save_choice == "y")
    results_dir = Path("data/results")
    results_dir.mkdir(parents=True, exist_ok=True)

    # --- 2) Base Parquet: load + query ---
    t0 = time.time()
    df_base = pd.read_parquet(base_path)
    t_load_base = time.time() - t0

    t0 = time.time()
    _, tmp = run_knn_query(df_base, (lat, lon), k)
    base_results = tmp.copy()
    base_results["traj_id"] = df_base.loc[tmp.index, "traj_id"].values
    base_results["lat"]     = df_base.loc[tmp.index, "lat"].values
    base_results["lon"]     = df_base.loc[tmp.index, "lon"].values
    t_query_base = time.time() - t0

    print(f"\n{'='*10} kNN Base Parquet {'='*10}")
    print(f"Load time: {t_load_base:.4f}s, Query time: {t_query_base:.4f}s")
    if not base_results.empty:
        print(base_results.to_string(index=False))
    else:
        print("No results found in Base Parquet.")

    # --- 3) Fixed Segments: load + query ---
    t0 = time.time()
    df_fixed = pd.read_parquet(fixed_path)
    t_load_fixed = time.time() - t0

    t0 = time.time()
    fixed_results = run_knn_query_on_fixed_segments_vectorized(
        parquet_path=fixed_path,
        query_point=(lat, lon),
        k=k,
        top_n_centroids=10000
    )
    t_query_fixed = time.time() - t0

    print(f"\n{'='*10} kNN Fixed Segments {'='*10}")
    print(f"Load time: {t_load_fixed:.4f}s, Query time: {t_query_fixed:.4f}s")
    if not fixed_results.empty:
        print(fixed_results.to_string(index=False))
    else:
        print("No results found in Fixed Segments.")

    # --- 4) Grid Segments: coarse pre-filter + precise kNN ---
    from src.models.grid import Grid

    # (a) load metadata for grid segments
    t0 = time.time()
    meta_cols = ["entity_id", "grid_cell", "min_x", "max_x", "min_y", "max_y"]
    df_meta = pd.read_parquet(grid_path, columns=meta_cols)

    # create grid and find neighbors
    # (x=lat, y=lon) so min_x/max_x are lat bounds
    grid = Grid(
        min_lat   = df_meta["min_x"].min(),
        max_lat   = df_meta["max_x"].max(),
        min_lon   = df_meta["min_y"].min(),
        max_lon   = df_meta["max_y"].max(),
        cell_size = 0.001
    )
    ref_cell  = grid.get_cell_id(lat, lon)
    neighbors = [
        f"{ref_cell[0]+di}_{ref_cell[1]+dj}"
        for di in (-1, 0, 1)
        for dj in (-1, 0, 1)
    ]

    # Coarse filter στα metadata
    df_meta = df_meta[df_meta["grid_cell"].isin(neighbors)]

    if df_meta.empty:
        # if no segments match the coarse filter, skip the query
        t_load_grid  = time.time() - t0
        t_query_grid = 0.0
        grid_results = pd.DataFrame(columns=["traj_id", "lat", "lon", "distance"])
    else:
        # (b) load only the necessary columns for the filtered segments
        t_load_grid = None
        df_grid = pd.read_parquet(
            grid_path,
            columns=["entity_id", "vals_x", "vals_y"],
            filters=[("entity_id", "in", df_meta["entity_id"].tolist())]
        )
        t_load_grid = time.time() - t0

        # (c) execute kNN query on the filtered segments
        t1 = time.time()
        grid_results = run_knn_query_on_segments(df_grid, (lat, lon), k)
        t_query_grid = time.time() - t1

    print(f"\n{'='*10} kNN Grid Segments {'='*10}")
    print(f"Load time: {t_load_grid:.4f}s, Query time: {t_query_grid:.4f}s")
    if not grid_results.empty:
        print(grid_results.to_string(index=False))
    else:
        print("No results found in Grid Segments.")


    # --- 5) Summary ---
    summary = pd.DataFrame([
        {"method": "Base",  "load": t_load_base,  "query": t_query_base},
        {"method": "Fixed", "load": t_load_fixed, "query": t_query_fixed},
        {"method": "Grid",  "load": t_load_grid,  "query": t_query_grid},
    ])
    summary["total"] = summary["load"] + summary["query"]
    summary["rank"]  = summary["total"].rank(method="dense").astype(int)
    summary = summary.sort_values("rank")

    print("\n" + "="*20 + " Summary " + "="*20)
    print(
        summary.to_string(
            index=False,
            columns=["rank", "method", "load", "query", "total"],
            header=["#", "Method", "Load(s)", "Query(s)", "Total(s)"],
            float_format="%.4f"
        )
    )
    print("="*50)

    # --- 6) Save results if requested ---
    if should_save:
        base_results.to_csv( results_dir / "knn_base.csv",  index=False)
        fixed_results.to_csv(results_dir  / "knn_fixed.csv", index=False)
        grid_results.to_csv(results_dir  / "knn_grid.csv",  index=False)
        print(f"\nCSV files saved to: {results_dir}")
    else:
        print("\nSkipping result saving. No files created.")

    # --- 7) Compare Fixed vs Base Results ---
    def compare_results(df1, df2, label1="Base", label2="Fixed"):
        from math import isclose

        if df1.empty or df2.empty:
            print(f"\nCannot compare {label1} and {label2}: one is empty.")
            return

        diffs = []
        for i, (row1, row2) in enumerate(zip(df1.itertuples(), df2.itertuples()), 1):
            dist = haversine_distance(row1.lat, row1.lon, row2.lat, row2.lon)
            diffs.append({
                "rank": i,
                "traj_1": row1.traj_id,
                "traj_2": row2.traj_id,
                "lat_diff": abs(row1.lat - row2.lat),
                "lon_diff": abs(row1.lon - row2.lon),
                "distance_m": dist,
                "match": "same" if dist < 0.5 else ("small dif" if dist < 1.0 else "big dif")
            })

        df_diff = pd.DataFrame(diffs)
        print("\n" + "="*20 + f" {label2} vs {label1} Comparison " + "="*20)
        print(df_diff.to_string(index=False, float_format="%.3f"))

    compare_results(base_results, fixed_results, label1="Base", label2="Fixed")

def haversine_distance(lat1, lon1, lat2, lon2):
    """
    Compute the great-circle distance (in meters) between two points
    on the Earth using the Haversine formula.
    """
    R = 6371000  # Earth radius in meters
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    delta_phi = np.radians(lat2 - lat1)
    delta_lambda = np.radians(lon2 - lon1)

    a = np.sin(delta_phi / 2) ** 2 + \
        np.cos(phi1) * np.cos(phi2) * np.sin(delta_lambda / 2) ** 2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))

    return R * c



if __name__ == "__main__":
    run_knn_general_interactive()
