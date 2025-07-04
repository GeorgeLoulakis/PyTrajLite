"""
Main entry point for PyTrajLite
Handles user options, loads data, and manages Parquet and CSV generation.
"""

from pathlib import Path
from time import time
from datetime import datetime
import pandas as pd

# Core imports for trajectory parsing and saving
from src.raw_input_loader import parse_plt_file
from src.fileio import (
    save_knn_friendly_segments,
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
from src.queries import (
    compare_all_formats,
    run_bbox_evaluation,
    run_knn_general_interactive,
)
from src.queries.spatial_geoparquet import (
    run_geoparquet_interactive,
    run_knn_interactive,
)

# Queries for kNN on Parquet files (Base/Fixed/Grid)
from src.queries.knn import run_knn_general_interactive


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

    print("\nSaving base trajectories to Parquet...")
    save_start = time()
    save_trajectories_to_parquet(trajectories, base_parquet_path)
    save_duration = time() - save_start
    total_duration = time() - start_time

    save_trajectories_to_csv(trajectories, Path("data/processed/trajectories.csv"))

    print(f"Base trajectories saved to: {base_parquet_path}")
    print(f"Save time: {save_duration:.2f} seconds.")
    print(f"Total duration: {total_duration:.2f} seconds.")
    return trajectories

def save_trajectories_to_csv(trajectories: list, output_path: Path) -> None:
    """
    Save a list of Trajectory objects to a CSV file.
    Shows progress and timing.
    """
    print("\nPreparing DataFrame for CSV saving...")
    start_prep = time()
    rows = []
    total = len(trajectories)

    for i, traj in enumerate(trajectories, start=1):
        percent = (i / total) * 100
        print(f"\r[{percent:5.1f}%] Processing trajectory {traj.traj_id}...", end="")
        for p in traj.points:
            rows.append({
                "traj_id": traj.traj_id,
                "lat": p.lat,
                "lon": p.lon,
                "altitude": p.altitude,
                "timestamp": p.timestamp
            })
    prep_duration = time() - start_prep

    print("\nSaving CSV file...")
    start_save = time()
    df = pd.DataFrame(rows)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    save_duration = time() - start_save

    print(f"CSV file saved to: {output_path}")
    print(f"CSV preparation time: {prep_duration:.2f} seconds.")
    print(f"CSV save time: {save_duration:.2f} seconds.")
    print(f"Total time: {prep_duration + save_duration:.2f} seconds.")

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
            num_trajs = df["traj_id"].nunique()
            print(f"Found {num_trajs:,} unique trajectories.")

        # 1. Geometry creation with timing
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

        # 2. Save uncompressed version
        uncompressed_path = Path(
            "data/processed/trajectories_geoparquet_uncompressed.parquet"
        )
        start_uncompressed = time()
        gdf.to_parquet(uncompressed_path, index=False)
        uncompressed_duration = time() - start_uncompressed
        print(f"[GeoParquet] Saved uncompressed to: {uncompressed_path}")
        print(f"Save time (uncompressed): {uncompressed_duration:.2f} seconds.")

        # 3. Save compressed version
        compressed_path = Path(
            "data/processed/trajectories_geoparquet_compressed_snappy.parquet"
        )
        start_compressed = time()
        gdf.to_parquet(
            compressed_path, index=False, compression="snappy", row_group_size=10000
        )
        compressed_duration = time() - start_compressed
        print(f"[GeoParquet] Saved compressed (snappy) to: {compressed_path}")
        print(f"Save time (compressed): {compressed_duration:.2f} seconds.")

        # 4. Total time
        total_time = geom_duration + uncompressed_duration + compressed_duration
        print(f"Total time for GeoParquet generation: {total_time:.2f} seconds.")

    except Exception as e:
        print(f"[GeoParquet] Error during creation: {e}")

def generate_fixed_segments_with_centroids(trajectories, fixed_parquet_path: Path, fixed_knn_path: Path):
    print("\nGenerating fixed-size segments with centroid support...")
    all_fixed_segments = []
    total = len(trajectories)

    start_time = time()
    for i, traj in enumerate(trajectories, start=1):
        percent = (i / total) * 100
        print(f"\r[{percent:5.1f}%] Segmenting (fixed+centroid) {traj.traj_id}...", end="")
        segments = segment_trajectory_by_fixed_size(traj, max_segment_size=100)
        all_fixed_segments.extend(segments)
    seg_duration = time() - start_time

    print("\nSaving fixed-size segments to original Parquet...")
    save_segments_to_parquet(all_fixed_segments, fixed_parquet_path)

    print("Saving centroid-enhanced version for kNN queries...")
    save_start = time()
    save_knn_friendly_segments(fixed_parquet_path, fixed_knn_path)
    save_duration = time() - save_start

    print(f"kNN-ready fixed segments saved to: {fixed_knn_path}")
    print(f"Segmentation time: {seg_duration:.2f} seconds.")
    print(f"Save time: {save_duration:.2f} seconds.")
    print(f"Total time: {seg_duration + save_duration:.2f} seconds.")

def generate_grid_segments_with_centroids(trajectories, grid_parquet_path: Path, grid_knn_path: Path):
    print("\nGenerating grid-based segments with centroid support...")
    grid = Grid.from_trajectories(trajectories, cell_size=0.001)
    all_grid_segments = []
    total = len(trajectories)

    start_time = time()
    for i, traj in enumerate(trajectories, start=1):
        percent = (i / total) * 100
        print(f"\r[{percent:5.1f}%] Segmenting (grid+centroid) {traj.traj_id}...", end="")
        segments = segment_trajectory_by_grid(traj, grid)
        all_grid_segments.extend(segments)
    seg_duration = time() - start_time

    print("\nSaving grid-based segments to original Parquet...")
    save_segments_to_parquet(all_grid_segments, grid_parquet_path)

    print("Saving centroid-enhanced version for kNN queries...")
    save_start = time()
    save_knn_friendly_segments(grid_parquet_path, grid_knn_path)
    save_duration = time() - save_start

    print(f"kNN-ready grid segments saved to: {grid_knn_path}")
    print(f"Segmentation time: {seg_duration:.2f} seconds.")
    print(f"Save time: {save_duration:.2f} seconds.")
    print(f"Total time: {seg_duration + save_duration:.2f} seconds.")

# Menu Option 1: Generate all required Parquet files
def create_parquet_from_raw():
    """Main workflow for creating all necessary Parquet and GeoParquet files."""
    base_parquet_path = Path("data/processed/trajectories.parquet")
    csv_path = Path("data/processed/trajectories.csv")
    data_path = Path("data/raw/Data")
    user_dirs = sorted([p for p in data_path.iterdir() if p.is_dir()])

    trajectories = []
    if not base_parquet_path.exists():
        trajectories = generate_base_parquet(base_parquet_path, user_dirs)
    else:
        print("\nBase Parquet file already exists.")
        print(f"File: {base_parquet_path}")
        trajectories = load_trajectories_from_parquet(base_parquet_path)

        # if CSV does not exist, generate it from loaded trajectories
        if not csv_path.exists():
            print("[CSV] File not found — generating from loaded trajectories.")
            save_trajectories_to_csv(trajectories, csv_path)
        else:
            print("[CSV] File already exists.")

    if not trajectories:
        return

    fixed_parquet_path = Path("data/processed/trajectory_segments_fixed.parquet")
    fixed_knn_path = Path("data/processed/trajectory_segments_fixed_knn.parquet")
    grid_parquet_path = Path("data/processed/trajectory_segments_grid.parquet")
    grid_knn_path = Path("data/processed/trajectory_segments_grid_knn.parquet")
    geoparquet_path = Path("data/processed/trajectories_geoparquet.parquet")


    if not fixed_parquet_path.exists():
        generate_fixed_segments(trajectories, fixed_parquet_path)
    else:
        print(f"\nFixed-size segments already exist at: {fixed_parquet_path}")

    if not fixed_knn_path.exists():
        generate_fixed_segments_with_centroids(trajectories, fixed_parquet_path, fixed_knn_path)
    else:
        print(f"\nFixed-knn-size segments already exist at: {fixed_knn_path}")

    if not grid_parquet_path.exists():
        generate_grid_segments(trajectories, grid_parquet_path)
    else:
        print(f"\nGrid-based segments already exist at: {grid_parquet_path}")

    if not grid_knn_path.exists():
        generate_grid_segments_with_centroids(trajectories, grid_parquet_path, grid_knn_path)
    else:
        print(f"\nGrid-knn-size segments already exist at: {grid_knn_path}")

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
        choice = input("Enter your choice (0-6): ")

        if choice == "0":
            print("Exiting PyTrajLite.")
            pause_and_clear()
            break
        elif choice == "1":
            create_parquet_from_raw()
        elif choice == "2":
            run_compare_all_formats()
        elif choice == "3":
            run_bbox_eval()
        elif choice == "4":
            run_geoparquet_interactive()
        elif choice == "5":
            run_knn_general_interactive()
        elif choice == "6":
            run_knn_interactive()
        else:
            print("Invalid option. Please enter 0, 1, 2, 3 or 4.")

        pause_and_clear()
