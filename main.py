'''
Main entry point for PyTrajLite
Handles user options, loads data, and manages Parquet and CSV generation.
'''

from pathlib import Path
from time import time
from datetime import datetime
import pandas as pd

from src.raw_input_loader import parse_plt_file
from src.fileio import save_segments_to_parquet, save_trajectories_to_parquet, load_trajectories_from_parquet, load_segments_from_parquet
from src.models.grid import Grid
from src.segmentation import segment_trajectory_by_fixed_size, segment_trajectory_by_grid
from src.utils import display_menu, pause_and_clear
from src.queries import run_bbox_evaluation , compare_all_formats


def create_parquet_from_raw():
    """
    Handle menu option 1: Create Parquet file from raw data,
    and generate segmented Parquet if not already created.
    """
    base_parquet_path = Path("data/processed/trajectories.parquet")
    #segment_parquet_path = Path("data/processed/trajectory_segments.parquet")

    data_path = Path("data/raw/Data")
    user_dirs = sorted([p for p in data_path.iterdir() if p.is_dir()])
    total_dirs = len(user_dirs)
    trajectories = []

    # STEP 1: Create base trajectories.parquet
    if not base_parquet_path.exists():
        print("\nBase Parquet file not found. Creating from raw PLT data...\n")
        start_time = time()

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
            return
        else:
            save_trajectories_to_parquet(trajectories, base_parquet_path)
            duration = time() - start_time
            print(f"\nBase Parquet file created in {duration:.2f} seconds.")
    else:
        print("\nBase Parquet file already exists.")
        print(f"File: {base_parquet_path}")
        trajectories = load_trajectories_from_parquet(base_parquet_path)

    # STEP 2: Create both segment files
    fixed_parquet_path = Path("data/processed/trajectory_segments_fixed.parquet")
    grid_parquet_path = Path("data/processed/trajectory_segments_grid.parquet")

    total_trajs = len(trajectories)

    # (a) Fixed-size segmentation
    if not fixed_parquet_path.exists():
        print("\nFixed-size segment Parquet file not found. Creating segments using fixed-size partitioning...")
        all_fixed_segments = []
        start_fixed_time = time()

        for i, traj in enumerate(trajectories, start=1):
            percent = (i / total_trajs) * 100
            print(f"\r[{percent:5.1f}%] Segmenting (fixed) {traj.traj_id}...", end="")
            segments = segment_trajectory_by_fixed_size(traj, max_segment_size=100)
            all_fixed_segments.extend(segments)

        duration = time() - start_fixed_time
        save_segments_to_parquet(all_fixed_segments, fixed_parquet_path)
        print(f"\n{len(all_fixed_segments)} fixed-size segments saved to: {fixed_parquet_path} in {duration:.2f} seconds.")
    else:
        print(f"\nFixed-size segments already exist at: {fixed_parquet_path}")

    # (b) Grid-based segmentation
    if not grid_parquet_path.exists():
        print("\nGrid-based segment Parquet file not found. Creating segments using grid-based partitioning...")
        grid = Grid.from_trajectories(trajectories, cell_size=0.001)
        all_grid_segments = []
        start_grid_time = time()

        for i, traj in enumerate(trajectories, start=1):
            percent = (i / total_trajs) * 100
            print(f"\r[{percent:5.1f}%] Segmenting (grid) {traj.traj_id}...", end="")
            segments = segment_trajectory_by_grid(traj, grid)
            all_grid_segments.extend(segments)

        duration = time() - start_grid_time
        save_segments_to_parquet(all_grid_segments, grid_parquet_path)
        print(f"\n{len(all_grid_segments)} grid-based segments saved to: {grid_parquet_path} in {duration:.2f} seconds.")
    else:
        print(f"\nGrid-based segments already exist at: {grid_parquet_path}")

    pause_and_clear()


def run_bbox_eval():
    """Handle menu option 2: Run a Bounding Box spatial query on the Parquet data."""
    run_bbox_evaluation()
    pause_and_clear()


def run_compare_all_formats():
    """Handle menu option 3: Compare size and read performance of Parquet vs CSV files."""
    compare_all_formats()
    pause_and_clear()


if __name__ == "__main__":
    while True:
        display_menu()
        choice = input("Enter your choice (0-3): ")

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

        else:
            print("Invalid option. Please enter 0, 1, 2, or 3.")
            pause_and_clear()
