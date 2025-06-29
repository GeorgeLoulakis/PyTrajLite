import pandas as pd
from pathlib import Path
from src.models import Point, Trajectory

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
    
def save_knn_friendly_segments(input_path: Path, output_path: Path):
    """
    Load a segment Parquet file (fixed/grid), compute centroid fields,
    and save to new file for use in kNN queries.
    """
    import pandas as pd

    df = pd.read_parquet(input_path)

    if "min_y" not in df.columns or "max_y" not in df.columns:
        print(f"Cannot compute centroid: min_y/max_y not found in {input_path}")
        return
    if "min_x" not in df.columns or "max_x" not in df.columns:
        print(f"Cannot compute centroid: min_x/max_x not found in {input_path}")
        return

    df["centroid_lat"] = (df["min_y"] + df["max_y"]) / 2
    df["centroid_lon"] = (df["min_x"] + df["max_x"]) / 2

    df.to_parquet(output_path, index=False)
    print(f"Saved kNN-ready segments to: {output_path}")
