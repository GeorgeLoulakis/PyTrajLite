from dataclasses import dataclass, field
from typing import List
from datetime import datetime

@dataclass
class TrajectorySegment:
    entity_id: str
    segment_id: int
    vals_x: List[float]
    vals_y: List[float]
    vals_t: List[str]
    min_x: float = field(init=False)
    max_x: float = field(init=False)
    min_y: float = field(init=False)
    max_y: float = field(init=False)
    min_t: str = field(init=False)
    max_t: str = field(init=False)

    def __post_init__(self):
        self.compute_bounds()

    def compute_bounds(self):
        self.min_x = min(self.vals_x)
        self.max_x = max(self.vals_x)
        self.min_y = min(self.vals_y)
        self.max_y = max(self.vals_y)
        times = [datetime.fromisoformat(t) for t in self.vals_t]
        self.min_t = min(times).isoformat()
        self.max_t = max(times).isoformat()

    def to_dict(self):
        return {
            "entity_id": self.entity_id,
            "segment_id": self.segment_id,
            "vals_x": self.vals_x,
            "vals_y": self.vals_y,
            "vals_t": self.vals_t,
            "min_x": self.min_x,
            "max_x": self.max_x,
            "min_y": self.min_y,
            "max_y": self.max_y,
            "min_t": self.min_t,
            "max_t": self.max_t
        }