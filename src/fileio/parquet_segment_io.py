import pandas as pd
from pathlib import Path
from src.models import TrajectorySegment

def save_segments_to_parquet(segments: list[TrajectorySegment], output_path: Path):
    """
    Save a list of TrajectorySegment objects to a Parquet file.
    """
    rows = [seg.to_dict() for seg in segments]
    df = pd.DataFrame(rows)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path, index=False)

def load_segments_from_parquet(parquet_path: Path) -> list[TrajectorySegment]:
    """
    Load a Parquet file of TrajectorySegment records.
    """
    df = pd.read_parquet(parquet_path)
    segments = []
    for _, row in df.iterrows():
        seg = TrajectorySegment(
            entity_id=row["entity_id"],
            segment_id=row["segment_id"],
            vals_x=row["vals_x"],
            vals_y=row["vals_y"],
            vals_t=row["vals_t"],
        )
        segments.append(seg)
    return segments
