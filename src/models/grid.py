from math import floor

class Grid:
    def __init__(self, min_lat, max_lat, min_lon, max_lon, cell_size=0.001):
        self.min_lat = min_lat
        self.max_lat = max_lat
        self.min_lon = min_lon
        self.max_lon = max_lon
        self.cell_size = cell_size

    def get_cell_id(self, lat, lon):
        """
        Calculates the cell_id as a tuple (i, j) for a given lat, lon
        """
        i = int((lat - self.min_lat) / self.cell_size)
        j = int((lon - self.min_lon) / self.cell_size)
        return (i, j)

    @classmethod
    def from_trajectories(cls, trajectories, cell_size=0.001):
        """
        Creates a Grid that covers all trajectories
        """
        all_lats = [p.lat for traj in trajectories for p in traj.points]
        all_lons = [p.lon for traj in trajectories for p in traj.points]
        return cls(min(all_lats), max(all_lats), min(all_lons), max(all_lons), cell_size)
