"""
Main entry point for PyTrajLite
Loads and displays basic statistics about the loaded trajectories.
"""

from pathlib import Path
from src.loader import load_all_trajectories

if __name__ == "__main__":
    print("Loading trajectories...")
    data_path = Path("data/raw/Data")
    trajectories = load_all_trajectories(data_path)
    print(f"Total trajectories loaded: {len(trajectories)}")

    total_points = sum(len(traj) for traj in trajectories)
    print(f"Total points: {total_points}")

    if trajectories:
        print("Sample trajectory (first 3 points):")
        for p in trajectories[0].points[:3]:
            print(f"  Lat: {p.lat}, Lon: {p.lon}, Timestamp: {p.timestamp}")
