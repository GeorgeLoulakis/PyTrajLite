import pandas as pd
from pathlib import Path
from src.models.base import Trajectory

def save_trajectories_to_parquet(trajectories, output_path: Path):
    """
    Save a list of Trajectory objects to a single Parquet file.
    Each row corresponds to a point, including trajectory ID.
    """
    rows = []
    for traj in trajectories:
        for p in traj.points:
            rows.append({
                "traj_id": traj.traj_id,
                "lat": p.lat,
                "lon": p.lon,
                "altitude": p.altitude,
                "timestamp": p.timestamp
            })

    df = pd.DataFrame(rows)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path, index=False)

def load_trajectories_from_parquet(parquet_path: Path):
    """
    Load a Parquet file into a list of Trajectory objects.
    """
    df = pd.read_parquet(parquet_path)
    grouped = df.groupby("traj_id")

    from models.base import Point, Trajectory
    trajectories = []
    for traj_id, group in grouped:
        points = [
            Point(row.lat, row.lon, row.altitude, row.timestamp)
            for row in group.itertuples()
        ]
        trajectories.append(Trajectory(traj_id, points))
    return trajectories
