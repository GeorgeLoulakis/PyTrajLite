import numpy as np
import pandas as pd
from typing import Tuple
from pathlib import Path



def run_bbox_query_on_points(df: pd.DataFrame, bbox: Tuple[float, float, float, float]) -> pd.DataFrame:
    """
    Apply a bounding box filter on individual points.
    """
    min_lat, max_lat, min_lon, max_lon = bbox
    return df[
        (df['lat'] >= min_lat) & (df['lat'] <= max_lat) &
        (df['lon'] >= min_lon) & (df['lon'] <= max_lon)
    ]

def run_bbox_query_on_segments(df: pd.DataFrame, bbox: Tuple[float, float, float, float]) -> pd.DataFrame:
    """
    Apply a bounding box filter on segment-level metadata (min/max coordinates).
    """
    min_lat, max_lat, min_lon, max_lon = bbox
    return df[
        (df['max_x'] >= min_lat) & (df['min_x'] <= max_lat) &
        (df['max_y'] >= min_lon) & (df['min_y'] <= max_lon)
    ]

def run_bbox_query_on_segments_optimized(path: Path, bbox: Tuple[float, float, float, float]) -> pd.DataFrame:
    """
    Optimized BBox filtering on segment metadata:
    - Loads min/max and vals_x/y columns (refinement support)
    - Converts stringified vals_x/y to NumPy if needed
    - Returns filtered rows by min/max
    """
    min_lat, max_lat, min_lon, max_lon = bbox

    # Load only the required columns for bounding box filtering
    df = pd.read_parquet(path, columns=['entity_id', 'min_x', 'max_x', 'min_y', 'max_y', 'vals_x', 'vals_y'])

    # Convert stringified arrays to NumPy arrays if necessary
    for col in ['vals_x', 'vals_y']:
        if pd.api.types.is_object_dtype(df[col]) and isinstance(df[col].dropna().iloc[0], str):
            df[col] = df[col].str.strip('{}').apply(
                lambda s: np.fromstring(s, sep=',', dtype=np.float32)
            )

    # Coarse filtering based on segment bounding boxes (min/max)
    return df[
        (df['max_x'] >= min_lat) & (df['min_x'] <= max_lat) &
        (df['max_y'] >= min_lon) & (df['min_y'] <= max_lon)
    ]

def run_bbox_query_on_segments_numpy2_optimized(
    df: pd.DataFrame,
    bbox: Tuple[float, float, float, float]
) -> pd.DataFrame:
    """
    Fast bounding box filter using min/max bounds only.
    Assumes `vals_x` and `vals_y` are already NumPy arrays.
    Actual point-level refinement will be handled separately.
    """
    if df.empty:
        return df

    min_lat, max_lat, min_lon, max_lon = bbox

    return df[
        (df['max_x'] >= min_lat) & (df['min_x'] <= max_lat) &
        (df['max_y'] >= min_lon) & (df['min_y'] <= max_lon)
    ]

def run_bbox_query_on_segments_numpy2_optimized_v2(
    df: pd.DataFrame,
    bbox: Tuple[float, float, float, float]
) -> pd.DataFrame:
    """
    Optimized version 2 of numpy segment query with:
    - itertuples() instead of iterrows()
    - Applies only coarse filtering using min/max bounds.
    - Assumes `vals_x`, `vals_y` are already NumPy arrays.
    """
    if df.empty:
        return df

    min_lat, max_lat, min_lon, max_lon = bbox

    # Coarse filtering using min/max bounds
    df_filtered = df[
        (df['max_x'] >= min_lat) & (df['min_x'] <= max_lat) &
        (df['max_y'] >= min_lon) & (df['min_y'] <= max_lon)
    ]

    return df_filtered

def extract_points_from_segments(df: pd.DataFrame, bbox: Tuple[float, float, float, float]) -> pd.DataFrame:
    """
    Extracts only the points (lat, lon) from segment data that fall inside the given BBox.
    Removes duplicates based on (entity_id, lat, lon).
    """
    if df.empty:
        return pd.DataFrame(columns=["entity_id", "lat", "lon", "timestamp"])

    min_lat, max_lat, min_lon, max_lon = bbox
    seen = set()
    rows = []

    for row in df.itertuples():
        try:
            entity = getattr(row, "entity_id", None)
            x_vals = row.vals_x
            y_vals = row.vals_y

            for x, y in zip(x_vals, y_vals):
                if min_lat <= x <= max_lat and min_lon <= y <= max_lon:
                    key = (entity, round(x, 6), round(y, 6))  # use rounding to reduce float precision noise
                    if key not in seen:
                        seen.add(key)
                        rows.append({
                            "entity_id": entity,
                            "lat": x,
                            "lon": y,
                            "timestamp": "unknown"  # placeholder
                        })
        except Exception:
            continue

    return pd.DataFrame(rows)

# not used in the current implementation

# def run_bbox_query_on_segments_numpy(df: pd.DataFrame, bbox: Tuple[float, float, float, float]) -> pd.DataFrame:
    """
    Only apply fast BBox filtering on segment min/max bounds.
    Actual point-level refinement will be handled later.
    """
    min_lat, max_lat, min_lon, max_lon = bbox

    # Stage 1: Fast bounding box filter
    return df[
        (df['max_x'] >= min_lat) & (df['min_x'] <= max_lat) &
        (df['max_y'] >= min_lon) & (df['min_y'] <= max_lon)
    ]
