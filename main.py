'''
Main entry point for PyTrajLite
Handles user options, loads data, and manages Parquet and CSV generation.
'''

from pathlib import Path
from time import time
from datetime import datetime
import pandas as pd

# Core imports for trajectory parsing and saving
from src.raw_input_loader import parse_plt_file
from src.fileio import (
    save_segments_to_parquet,
    save_trajectories_to_parquet,
    load_trajectories_from_parquet,
    load_segments_from_parquet,
)

# Import spatial grid structure and segmentation methods
from src.models.grid import Grid
from src.segmentation import (
    segment_trajectory_by_fixed_size,
    segment_trajectory_by_grid,
)

# UI utilities for menu and flow control
from src.utils import display_menu, pause_and_clear

# Queries and format comparisons
from src.queries.bbox.evaluation import run_bbox_evaluation
from src.queries.compare_parquet_vs_csv import compare_all_formats
from src.queries.spatial_geoparquet import evaluate_geoparquet,run_geoparquet_interactive

# Trajectory Data Processing Functions

def generate_base_parquet(base_parquet_path: Path, user_dirs) -> list:
    """Load and convert raw .plt files into a base Parquet file."""
    print("\nBase Parquet file not found. Creating from raw PLT data...\n")
    start_time = time()
    trajectories = []
    total_dirs = len(user_dirs)

    for i, user_dir in enumerate(user_dirs, start=1):
        percent = (i / total_dirs) * 100
        print(f"\r[{percent:5.1f}%] Loading {user_dir.name}...", end="")

        for file in (user_dir / "Trajectory").glob("*.plt"):
            traj = parse_plt_file(file)
            if len(traj) > 0:
                trajectories.append(traj)

    if not trajectories:
        print("No trajectories found in raw data. Exiting.")
        pause_and_clear()
        return []

    save_trajectories_to_parquet(trajectories, base_parquet_path)
    duration = time() - start_time
    print(f"\nBase Parquet file created in {duration:.2f} seconds.")
    return trajectories

def generate_fixed_segments(trajectories, fixed_parquet_path: Path):
    """Perform fixed-size segmentation on trajectories and save as Parquet."""
    print("\nFixed-size segment Parquet file not found. Creating segments using fixed-size partitioning...")
    all_fixed_segments = []
    start_fixed_time = time()
    total_trajs = len(trajectories)

    for i, traj in enumerate(trajectories, start=1):
        percent = (i / total_trajs) * 100
        print(f"\r[{percent:5.1f}%] Segmenting (fixed) {traj.traj_id}...", end="")
        segments = segment_trajectory_by_fixed_size(traj, max_segment_size=100)
        all_fixed_segments.extend(segments)

    duration = time() - start_fixed_time
    save_segments_to_parquet(all_fixed_segments, fixed_parquet_path)
    print(f"\n{len(all_fixed_segments)} fixed-size segments saved to: {fixed_parquet_path} in {duration:.2f} seconds.")

def generate_grid_segments(trajectories, grid_parquet_path: Path):
    """Perform grid-based segmentation on trajectories and save as Parquet."""
    print("\nGrid-based segment Parquet file not found. Creating segments using grid-based partitioning...")
    grid = Grid.from_trajectories(trajectories, cell_size=0.001)
    all_grid_segments = []
    start_grid_time = time()
    total_trajs = len(trajectories)

    for i, traj in enumerate(trajectories, start=1):
        percent = (i / total_trajs) * 100
        print(f"\r[{percent:5.1f}%] Segmenting (grid) {traj.traj_id}...", end="")
        segments = segment_trajectory_by_grid(traj, grid)
        all_grid_segments.extend(segments)

    duration = time() - start_grid_time
    save_segments_to_parquet(all_grid_segments, grid_parquet_path)
    print(f"\n{len(all_grid_segments)} grid-based segments saved to: {grid_parquet_path} in {duration:.2f} seconds.")

def generate_geoparquet(base_parquet_path: Path, geoparquet_path: Path):
    """Create and export a GeoParquet file by adding geometry to trajectory records."""
    try:
        import geopandas as gpd
        from shapely.geometry import Point

        print("\n[GeoParquet] Creating GeoParquet from base trajectories.parquet...")
        df = pd.read_parquet(base_parquet_path)

        gdf = gpd.GeoDataFrame(
            df,
            geometry=[Point(lon, lat) for lon, lat in zip(df["lon"], df["lat"])],
            crs="EPSG:4326"
        )

        gdf.to_parquet(geoparquet_path, index=False)
        print(f"[GeoParquet] Saved to: {geoparquet_path}")

    except Exception as e:
        print(f"[GeoParquet] Error during creation: {e}")

# Menu Option 1: Generate all required Parquet files

def create_parquet_from_raw():
    """Main workflow for creating all necessary Parquet and GeoParquet files."""
    base_parquet_path = Path("data/processed/trajectories.parquet")
    data_path = Path("data/raw/Data")
    user_dirs = sorted([p for p in data_path.iterdir() if p.is_dir()])

    trajectories = []
    if not base_parquet_path.exists():
        trajectories = generate_base_parquet(base_parquet_path, user_dirs)
    else:
        print("\nBase Parquet file already exists.")
        print(f"File: {base_parquet_path}")
        trajectories = load_trajectories_from_parquet(base_parquet_path)

    if not trajectories:
        return

    fixed_parquet_path = Path("data/processed/trajectory_segments_fixed.parquet")
    grid_parquet_path = Path("data/processed/trajectory_segments_grid.parquet")

    if not fixed_parquet_path.exists():
        generate_fixed_segments(trajectories, fixed_parquet_path)
    else:
        print(f"\nFixed-size segments already exist at: {fixed_parquet_path}")

    if not grid_parquet_path.exists():
        generate_grid_segments(trajectories, grid_parquet_path)
    else:
        print(f"\nGrid-based segments already exist at: {grid_parquet_path}")

    geoparquet_path = Path("data/processed/trajectories_geoparquet.parquet")
    if not geoparquet_path.exists():
        generate_geoparquet(base_parquet_path, geoparquet_path)
    else:
        print(f"[GeoParquet] GeoParquet already exists: {geoparquet_path}")

    pause_and_clear()

# Menu Option 2: BBox Query on regular Parquet files

def run_bbox_eval():
    """Run BBox evaluation on all segment formats and show results."""
    run_bbox_evaluation()
    pause_and_clear()

# Menu Option 3: Benchmark CSV vs Parquet

def run_compare_all_formats():
    """Compare CSV and Parquet formats in terms of I/O and size."""
    compare_all_formats()
    pause_and_clear()

# Application Entry Point

if __name__ == "__main__":
    while True:
        display_menu()
        choice = input("Enter your choice (0-4): ")

        if choice == "0":
            print("Exiting PyTrajLite.")
            pause_and_clear()
            break
        elif choice == "1":
            create_parquet_from_raw()
        elif choice == "2":
            run_bbox_eval()
        elif choice == "3":
            run_compare_all_formats()
        elif choice == "4":
            run_geoparquet_interactive()
        else:
            print("Invalid option. Please enter 0, 1, 2, 3 or 4.")
            pause_and_clear()
