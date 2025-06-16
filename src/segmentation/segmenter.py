from src.models import TrajectorySegment

def segment_trajectory(traj, grid):
    """
    Splits a trajectory into segments based on grid cell changes.
    Implements TrajParquet-style segmentation:
    The last point of each segment is also added as the first point of the next segment.
    """
    segments = []
    current_vals_x = []
    current_vals_y = []
    current_vals_t = []

    last_cell = None
    segment_id = 0

    for point in traj.points:
        cell = grid.get_cell_id(point.lat, point.lon)

        if last_cell is None:
            last_cell = cell

        # If the point belongs to a new cell, close the current segment
        if cell != last_cell and current_vals_x:
            seg = TrajectorySegment(
                entity_id=traj.traj_id,
                segment_id=segment_id,
                vals_x=current_vals_x,
                vals_y=current_vals_y,
                vals_t=current_vals_t
            )
            segments.append(seg)
            segment_id += 1

            # Start a new segment using the last point of the previous segment
            current_vals_x = [current_vals_x[-1]]
            current_vals_y = [current_vals_y[-1]]
            current_vals_t = [current_vals_t[-1]]

        # Add the current point to the segment
        current_vals_x.append(point.lat)
        current_vals_y.append(point.lon)
        current_vals_t.append(point.timestamp.isoformat())
        last_cell = cell

    # Save the final segment
    if current_vals_x:
        seg = TrajectorySegment(
            entity_id=traj.traj_id,
            segment_id=segment_id,
            vals_x=current_vals_x,
            vals_y=current_vals_y,
            vals_t=current_vals_t
        )
        segments.append(seg)

    return segments
