from src.models import TrajectorySegment
from datetime import datetime

def segment_trajectory(traj, grid):
    """
    Splits a trajectory into segments based on cell changes in the grid.
    The last point of each segment is also included in the next one.
    """
    segments = []
    current_vals_x = []
    current_vals_y = []
    current_vals_t = []

    last_cell = None
    segment_id = 0

    for idx, point in enumerate(traj.points):
        cell = grid.get_cell_id(point.lat, point.lon)

        if last_cell is None:
            last_cell = cell

        if cell != last_cell and len(current_vals_x) > 0:
            # Close the previous segment
            seg = TrajectorySegment(
                entity_id=traj.traj_id,
                segment_id=segment_id,
                vals_x=current_vals_x,
                vals_y=current_vals_y,
                vals_t=current_vals_t
            )
            segments.append(seg)
            segment_id += 1

            # Start a new segment, keeping the last point
            current_vals_x = [current_vals_x[-1]]
            current_vals_y = [current_vals_y[-1]]
            current_vals_t = [current_vals_t[-1]]

        current_vals_x.append(point.lat)
        current_vals_y.append(point.lon)
        current_vals_t.append(point.timestamp.isoformat())
        last_cell = cell

    if len(current_vals_x) > 0:
        seg = TrajectorySegment(
            entity_id=traj.traj_id,
            segment_id=segment_id,
            vals_x=current_vals_x,
            vals_y=current_vals_y,
            vals_t=current_vals_t
        )
        segments.append(seg)

    return segments
