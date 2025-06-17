from src.models import TrajectorySegment

def segment_trajectory_by_fixed_size(traj, max_segment_size=100):
    """
    Splits a trajectory into fixed-size segments.
    Each segment has up to `max_segment_size` points.
    Implements the TrajParquet-style segmentation.

    The last point of each segment is repeated as the first point of the next.
    """
    segments = []
    points = traj.points
    segment_id = 0

    if not points:
        return segments

    i = 0
    while i < len(points):
        # Get the subset of points
        sub_points = points[i:i + max_segment_size]

        # If there is a previous segment, also add its last point as the first point here        
        if segments and i > 0:
            sub_points = [points[i - 1]] + sub_points

        # Extract lat/lon/time
        vals_x = [p.lat for p in sub_points]
        vals_y = [p.lon for p in sub_points]
        vals_t = [p.timestamp.isoformat() for p in sub_points]

        segment = TrajectorySegment(
            entity_id=traj.traj_id,
            segment_id=segment_id,
            vals_x=vals_x,
            vals_y=vals_y,
            vals_t=vals_t
        )
        segments.append(segment)
        segment_id += 1

        # Next segment (with offset +N)
        i += max_segment_size

    return segments