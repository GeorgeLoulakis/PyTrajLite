"""
Main entry point for PyTrajLite
Loads and displays basic statistics about the loaded trajectories.
"""

from pathlib import Path
from src.loader import load_all_trajectories
from src.parquet_io import save_trajectories_to_parquet, load_trajectories_from_parquet

if __name__ == "__main__":
    data_path = Path("data/raw/Data")
    parquet_path = Path("data/processed/trajectories.parquet")

    if parquet_path.exists():
        print("Loading trajectories from Parquet...")
        trajectories = load_trajectories_from_parquet(parquet_path)
    else:
        print("Loading trajectories from PLT files...")
        trajectories = load_all_trajectories(data_path)
        print("Saving trajectories to Parquet...")
        save_trajectories_to_parquet(trajectories, parquet_path)

    print(f"Total trajectories loaded: {len(trajectories)}")

    total_points = sum(len(traj) for traj in trajectories)
    print(f"Total points: {total_points}")

    if trajectories:
        print("Sample trajectory (first 3 points):")
        for p in trajectories[0].points[:3]:
            print(f"  Lat: {p.lat}, Lon: {p.lon}, Timestamp: {p.timestamp}")