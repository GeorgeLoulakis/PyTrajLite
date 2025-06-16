from pathlib import Path
from datetime import datetime
from src.models import Point, Trajectory

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
