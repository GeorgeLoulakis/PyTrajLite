import os
import pandas as pd
import numpy as np
from pathlib import Path
from collections import defaultdict

# Path configuration
CURRENT_FILE = Path(__file__).resolve()
PROJECT_ROOT = CURRENT_FILE.parents[2]
PARQUET_PATH = PROJECT_ROOT / "data" / "processed" / "trajectory_segments.parquet"

def get_cell_id(lat: float, lon: float, min_lat: float, min_lon: float, cell_size: float):
    """
    Calculates the (i, j) grid cell index for given lat/lon coordinates.
    """
    i = int((lat - min_lat) / cell_size)
    j = int((lon - min_lon) / cell_size)
    return (i, j)

def debug_segment_cell_visits(traj_id: str, cell_size: float = 0.001):
    """
    Displays which grid cells are visited by which trajectory segments.
    """
    print(f"\n🔍 Loading segments for trajectory: {traj_id}")
    if not PARQUET_PATH.exists():
        print(f"[!] Parquet file not found at: {PARQUET_PATH}")
        return

    df = pd.read_parquet(PARQUET_PATH)
    df = df[df['entity_id'] == traj_id]

    if df.empty:
        print("No segments found for this trajectory.")
        return

    all_lats = np.concatenate(df['vals_x'].values)
    all_lons = np.concatenate(df['vals_y'].values)
    min_lat, max_lat = np.min(all_lats), np.max(all_lats)
    min_lon, max_lon = np.min(all_lons), np.max(all_lons)

    # cell_id → set(segment_ids), list(points)
    cell_to_segments = defaultdict(set)
    cell_to_points = defaultdict(list)

    for _, row in df.iterrows():
        segment_id = row['segment_id']
        for lat, lon in zip(row['vals_x'], row['vals_y']):
            cell_id = get_cell_id(lat, lon, min_lat, min_lon, cell_size)
            cell_to_segments[cell_id].add(segment_id)
            cell_to_points[cell_id].append((round(lat, 6), round(lon, 6)))  # rounded for readability

    print("\n📦 Cells visited by segments (with points):")
    sorted_cells = sorted(cell_to_segments.items(), key=lambda x: (x[0][0], x[0][1]))

    for cell, segments in sorted_cells:
        points = cell_to_points[cell]
        segment_list = sorted(list(segments))
        print(f"Cell {cell}: Segments {segment_list}")
        print(f"   Points in cell: {points}\n")

def main():
    traj_id = input("Enter trajectory ID: ").strip()
    try:
        cell_size_input = input("Enter grid cell size (e.g. 0.001): ").strip()
        cell_size = float(cell_size_input) if cell_size_input else 0.001
    except ValueError:
        print("Invalid input. Using default cell size: 0.001")
        cell_size = 0.001

    debug_segment_cell_visits(traj_id, cell_size)

if __name__ == "__main__":
    main()
