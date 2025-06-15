'''
Main entry point for PyTrajLite
Handles user options, loads data, and manages Parquet and CSV generation.
'''

from pathlib import Path
from time import time
import pandas as pd

from src.loader import load_all_trajectories
from src.fileio import save_trajectories_to_parquet
from src.utils import display_menu, pause_and_clear
from src.queries.bbox_query import bbox_query
from src.queries.compare_parquet_vs_csv import compare_parquet_vs_csv


if __name__ == "__main__":
    while True:
        display_menu()
        choice = input("Enter your choice (0-3): ")

        if choice == "0":
            print("Exiting PyTrajLite.")
            pause_and_clear()
            break

        # Option 1 – Create and load Parquet from raw data
        elif choice == "1":
            parquet_path = Path("data/processed/trajectories.parquet")

            if not parquet_path.exists():
                print("Parquet file not found. Creating it from raw data...")
                from datetime import datetime
                from src.loader import parse_plt_file

                data_path = Path("data/raw/Data")
                user_dirs = sorted([p for p in data_path.iterdir() if p.is_dir()])
                total_dirs = len(user_dirs)
                trajectories = []

                start_time = time()

                for i, user_dir in enumerate(user_dirs, start=1):
                    percent = (i / total_dirs) * 100
                    print(f"[{percent:5.1f}%] Loading {user_dir.name}...")

                    for file in (user_dir / "Trajectory").glob("*.plt"):
                        traj = parse_plt_file(file)
                        if len(traj) > 0:
                            trajectories.append(traj)

                if not trajectories:
                    print("No trajectories found in raw data. Exiting.")
                    pause_and_clear()
                    continue

                save_trajectories_to_parquet(trajectories, parquet_path)
                duration = time() - start_time
                print(f"Parquet file created in {duration:.2f} seconds.")
                pause_and_clear()

            else:
                print("Parquet file already exists. Skipping creation.")
                print(f"File: {parquet_path}")
                pause_and_clear()

        # Option 2 – Run a Bounding Box query
        elif choice == "2":
            bbox_query()
            pause_and_clear()

        # Option 3 – Compare Parquet and CSV formats
        elif choice == "3":
            compare_parquet_vs_csv()
            pause_and_clear()

        else:
            print("Invalid option. Please enter 0, 1, 2, or 3.")
            pause_and_clear()
