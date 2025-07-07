import pandas as pd
from pathlib import Path
from ..models import Point, Trajectory

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
    print(f"Loading trajectories from Parquet file: {parquet_path}")
    
    df = pd.read_parquet(parquet_path)
    print(f"Loaded DataFrame with {len(df)} rows.")

    grouped = df.groupby("traj_id")
    total_trajs = len(grouped)
    print(f"Found {total_trajs} unique trajectories.")

    trajectories = []
    for i, (traj_id, group) in enumerate(grouped, start=1):
        percent = (i / total_trajs) * 100
        print(f"\r[{percent:5.1f}%] Processing trajectory {traj_id}...", end="")

        points = [
            Point(row.lat, row.lon, row.altitude, row.timestamp)
            for row in group.itertuples()
        ]
        trajectories.append(Trajectory(traj_id, points))

    print(f"\nFinished loading {len(trajectories)} trajectories.")
    return trajectories