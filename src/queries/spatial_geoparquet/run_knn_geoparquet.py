#!/usr/bin/env python3
"""
run_knn_geoparquet.py

Interactive CLI for running kNN on a GeoParquet file (via DuckDB) with timing summary.
"""
import pandas as pd
from pathlib import Path
from .knn_geoparquet import run_knn_query_geoparquet_timed


def run_knn_interactive():
    # --- User input ---
    try:
        lat = float(input("Latitude (e.g. 39.9800): ").strip())
        lon = float(input("Longitude (e.g. 116.3200): ").strip())
        k   = int(input("Number of nearest points (k): ").strip())
    except ValueError:
        print("Invalid input; please enter numeric values.")
        return

    # paths
    geoparquet_path = "data/processed/trajectories_geoparquet_compressed_snappy.parquet"
    results_dir     = Path("data/results")
    results_dir.mkdir(parents=True, exist_ok=True)

    # run timed query
    df, load_time, query_time = run_knn_query_geoparquet_timed(
        geoparquet_path, (lat, lon), k
    )
    total_time = load_time + query_time

    # summary
    summary = pd.DataFrame([{
        "Load(s)":  load_time,
        "Query(s)": query_time,
        "Total(s)": total_time
    }])
    print("\n" + "="*25 + " GeoParquet kNN Summary " + "="*25)
    print(summary.to_string(index=False, float_format="%.4f"))
    print("="*80)

    # results
    if df.empty:
        print("No results found.")
    else:
        print("\n--- kNN Results ---")
        print(df.to_string(index=False))

    # save
    save = input("Save to CSV? (y/n): ").strip().lower()
    if save == 'y':
        out_path = results_dir / "geoparquet_knn_results.csv"
        df.to_csv(out_path, index=False)
        print(f"Results saved to {out_path}")


if __name__ == "__main__":
    run_knn_interactive()
