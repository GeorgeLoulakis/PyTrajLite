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
from time import time

from .query import (
    run_knn_query,
    run_knn_query_on_parquet,
    run_knn_query_on_segments,
)


def run_knn_general_interactive():
    base_path  = Path("data/processed/trajectories.parquet")
    fixed_path = Path("data/processed/trajectory_segments_fixed.parquet")
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
    t0 = time()
    df_base = pd.read_parquet(base_path)
    t_load_base = time() - t0

    t0 = time()
    _, tmp = run_knn_query(df_base, (lat, lon), k)
    base_results = tmp.copy()
    base_results["traj_id"] = df_base.loc[tmp.index, "traj_id"].values
    base_results["lat"]     = df_base.loc[tmp.index, "lat"].values
    base_results["lon"]     = df_base.loc[tmp.index, "lon"].values
    t_query_base = time() - t0

    print(f"\n{'='*10} kNN Base Parquet {'='*10}")
    print(f"Load time: {t_load_base:.4f}s, Query time: {t_query_base:.4f}s")
    if not base_results.empty:
        print(base_results.to_string(index=False))
    else:
        print("No results found in Base Parquet.")

    # --- 3) Fixed Segments: load + query ---
    t0 = time()
    df_fixed = pd.read_parquet(fixed_path)
    t_load_fixed = time() - t0

    t0 = time()
    fixed_results = run_knn_query_on_segments(
        df_fixed, (lat, lon), k,
        start_lat_col="min_x", start_lon_col="min_y",
        end_lat_col="max_x",   end_lon_col="max_y",
        grid_cell_col=None
    )
    t_query_fixed = time() - t0

    print(f"\n{'='*10} kNN Fixed Segments {'='*10}")
    print(f"Load time: {t_load_fixed:.4f}s, Query time: {t_query_fixed:.4f}s")
    if not fixed_results.empty:
        print(fixed_results.to_string(index=False))
    else:
        print("No results found in Fixed Segments.")

    # --- 4) Grid Segments: load only needed cols + query ---
    t0 = time()
    df_grid = pd.read_parquet(
        grid_path,
        columns=[
            "entity_id",
            "vals_x",
            "vals_y",
            "grid_cell",
            "min_x",
            "max_x",
            "min_y",
            "max_y"
        ]
    )
    t_load_grid = time() - t0

    t0 = time()
    grid_results = run_knn_query_on_segments(
        df_grid, (lat, lon), k,
        start_lat_col="min_x", start_lon_col="min_y",
        end_lat_col="max_x",   end_lon_col="max_y",
        grid_cell_col="grid_cell",
        cell_size=0.001, grid_ring=1
    )
    t_query_grid = time() - t0

    if not grid_results.empty:
        grid_results = grid_results.drop_duplicates(subset=["lat", "lon"] )

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


if __name__ == "__main__":
    run_knn_general_interactive()
