from dataclasses import dataclass
from datetime import datetime
from typing import List

@dataclass
class Point:
    lat: float
    lon: float
    altitude: float
    timestamp: datetime

@dataclass
class Trajectory:
    traj_id: str
    points: List[Point]

    def __len__(self):
        return len(self.points)

    def get_bbox(self):
        lats = [p.lat for p in self.points]
        lons = [p.lon for p in self.points]
        return min(lats), min(lons), max(lats), max(lons)