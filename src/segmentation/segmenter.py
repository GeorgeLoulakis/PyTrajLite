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

from src.models import TrajectorySegment, Grid
from typing import List

def segment_trajectory_by_grid(traj, grid: Grid):
    """
    Splits a trajectory into segments based on grid cell transitions.
    Each time the trajectory moves to a new cell, a new segment is created.
    
    Args:
        traj: The Trajectory object to segment
        grid: The Grid object used for cell assignment
    
    Returns:
        List of TrajectorySegment objects
    """
    segments = []
    if not traj.points:
        return segments
    
    current_segment_points = []
    prev_cell_id = None
    
    for i, point in enumerate(traj.points):
        current_cell_id = grid.get_cell_id(point.lat, point.lon)
        
        # First point - always add to current segment
        if prev_cell_id is None:
            current_segment_points.append(point)
            prev_cell_id = current_cell_id
            continue
        
        # Still in same cell - continue current segment
        if current_cell_id == prev_cell_id:
            current_segment_points.append(point)
            continue
        
        # Cell changed - finalize current segment and start new one
        if current_segment_points:
            # Create segment with overlap point (last point of previous segment)
            segment = create_segment_from_points(
                traj.traj_id, 
                len(segments), 
                current_segment_points
            )
            segments.append(segment)
            
            # Start new segment with last point of previous segment
            current_segment_points = [current_segment_points[-1], point]
            prev_cell_id = current_cell_id
    
    # Add the last segment if there are remaining points
    if current_segment_points:
        segment = create_segment_from_points(
            traj.traj_id, 
            len(segments), 
            current_segment_points
        )
        segments.append(segment)
    
    return segments

def create_segment_from_points(traj_id: str, segment_id: int, points: list):
    """
    Helper function to create a TrajectorySegment from a list of points.
    """
    vals_x = [p.lat for p in points]
    vals_y = [p.lon for p in points]
    vals_t = [p.timestamp.isoformat() for p in points]
    
    return TrajectorySegment(
        entity_id=traj_id,
        segment_id=segment_id,
        vals_x=vals_x,
        vals_y=vals_y,
        vals_t=vals_t
    )