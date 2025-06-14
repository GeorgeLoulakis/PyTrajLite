import os
from pathlib import Path
import pandas as pd
from datetime import datetime
from typing import List
from src.models.base import Point, Trajectory

def parse_plt_file(file_path: Path) -> Trajectory:
    """
    Parse a .plt trajectory file and return a Trajectory object.
    """
    with open(file_path, 'r') as f:
        lines = f.readlines()[6:]  # Skip metadata header

    points = []
    for line in lines:
        parts = line.strip().split(',')
        lat = float(parts[0])
        lon = float(parts[1])
        altitude = float(parts[3])
        timestamp = datetime.strptime(parts[5] + ' ' + parts[6], "%Y-%m-%d %H:%M:%S")
        points.append(Point(lat, lon, altitude, timestamp))

    traj_id = file_path.stem
    return Trajectory(traj_id=traj_id, points=points)

def load_all_trajectories(data_dir: Path) -> List[Trajectory]:
    """
    Traverse data directory and load all .plt files.
    """
    trajectories = []
    for subdir in data_dir.iterdir():
        if subdir.is_dir():
            for file in subdir.glob("Trajectory/*.plt"):
                traj = parse_plt_file(file)
                if len(traj) > 0:
                    trajectories.append(traj)
    return trajectories