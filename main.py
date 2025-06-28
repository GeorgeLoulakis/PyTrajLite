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
from src.queries.spatial_geoparquet.run_bbox_geoparquet import evaluate_geoparquet, run_geoparquet_interactive
from src.queries.spatial_geoparquet.run_knn_geoparquet import run_knn_interactive

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
    """Perform fixed-size segmentation on trajectories and save as Parquet, with timing information."""
    print("\nFixed-size segment Parquet file not found. Creating segments using fixed-size partitioning...")
    all_fixed_segments = []
    total_trajs = len(trajectories)

    start_seg_time = time()
    for i, traj in enumerate(trajectories, start=1):
        percent = (i / total_trajs) * 100
        print(f"\r[{percent:5.1f}%] Segmenting (fixed) {traj.traj_id}...", end="")
        segments = segment_trajectory_by_fixed_size(traj, max_segment_size=100)
        all_fixed_segments.extend(segments)
    seg_duration = time() - start_seg_time

    print("\nSaving fixed-size segments to Parquet...")
    start_save_time = time()
    save_segments_to_parquet(all_fixed_segments, fixed_parquet_path)
    save_duration = time() - start_save_time

    total_duration = seg_duration + save_duration
    print(f"{len(all_fixed_segments)} fixed-size segments saved to: {fixed_parquet_path}")
    print(f"Segmentation time: {seg_duration:.2f} seconds.")
    print(f"Save time: {save_duration:.2f} seconds.")
    print(f"Total duration: {total_duration:.2f} seconds.")

def generate_grid_segments(trajectories, grid_parquet_path: Path):
    """Perform grid-based segmentation on trajectories and save as Parquet, with timing information."""
    print("\nGrid-based segment Parquet file not found. Creating segments using grid-based partitioning...")
    grid = Grid.from_trajectories(trajectories, cell_size=0.001)
    all_grid_segments = []
    total_trajs = len(trajectories)

    start_seg_time = time()
    for i, traj in enumerate(trajectories, start=1):
        percent = (i / total_trajs) * 100
        print(f"\r[{percent:5.1f}%] Segmenting (grid) {traj.traj_id}...", end="")
        segments = segment_trajectory_by_grid(traj, grid)
        all_grid_segments.extend(segments)
    seg_duration = time() - start_seg_time

    print("\nSaving grid-based segments to Parquet...")
    start_save_time = time()
    save_segments_to_parquet(all_grid_segments, grid_parquet_path)
    save_duration = time() - start_save_time

    total_duration = seg_duration + save_duration
    print(f"{len(all_grid_segments)} grid-based segments saved to: {grid_parquet_path}")
    print(f"Segmentation time: {seg_duration:.2f} seconds.")
    print(f"Save time: {save_duration:.2f} seconds.")
    print(f"Total duration: {total_duration:.2f} seconds.")

def generate_geoparquet_versions(base_parquet_path: Path):
    """
    Generate two GeoParquet files:
    - Uncompressed: trajectories_geoparquet_uncompressed.parquet
    - Compressed (snappy): trajectories_geoparquet_compressed_snappy.parquet
    """
    try:
        import geopandas as gpd
        from shapely.geometry import Point

        print("\n[GeoParquet] Loading base Parquet file...")
        df = pd.read_parquet(base_parquet_path)
        print(f"Loaded DataFrame with {len(df):,} rows.")

        if "traj_id" in df.columns:
            num_trajs = df['traj_id'].nunique()
            print(f"Found {num_trajs:,} unique trajectories.")

        # 1. Δημιουργία γεωμετρίας με μέτρηση χρόνου
        print("Creating geometry column...")
        start_geom = time()
        total_rows = len(df)
        geometry = []
        for i, (lon, lat) in enumerate(zip(df["lon"], df["lat"]), start=1):
            geometry.append(Point(lon, lat))
            if i % (total_rows // 100) == 0 or i == total_rows:
                percent = (i / total_rows) * 100
                print(f"\r[ {percent:4.1f}%] Processing trajectory points...", end="")
        geom_duration = time() - start_geom

        print("\nCreating GeoDataFrame...")
        gdf = gpd.GeoDataFrame(df, geometry=geometry, crs="EPSG:4326")

        # 2. Αποθήκευση uncompressed
        uncompressed_path = Path("data/processed/trajectories_geoparquet_uncompressed.parquet")
        start_uncompressed = time()
        gdf.to_parquet(uncompressed_path, index=False)
        uncompressed_duration = time() - start_uncompressed
        print(f"[GeoParquet] Saved uncompressed to: {uncompressed_path}")
        print(f"Save time (uncompressed): {uncompressed_duration:.2f} seconds.")

        # 3. Αποθήκευση compressed
        compressed_path = Path("data/processed/trajectories_geoparquet_compressed_snappy.parquet")
        start_compressed = time()
        gdf.to_parquet(
            compressed_path,
            index=False,
            compression="snappy",
            row_group_size=10000
        )
        compressed_duration = time() - start_compressed
        print(f"[GeoParquet] Saved compressed (snappy) to: {compressed_path}")
        print(f"Save time (compressed): {compressed_duration:.2f} seconds.")

        # 4. Συνολικός χρόνος
        total_time = geom_duration + uncompressed_duration + compressed_duration
        print(f"Total time for GeoParquet generation: {total_time:.2f} seconds.")

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
        generate_geoparquet_versions(base_parquet_path)
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
        elif choice == "5":
            run_knn_interactive()
        else:
            print("Invalid option. Please enter 0, 1, 2, 3 or 4.")

        pause_and_clear()
