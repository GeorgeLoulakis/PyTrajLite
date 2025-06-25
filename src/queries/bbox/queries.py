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

def run_bbox_query_on_segments_numpy(df: pd.DataFrame, bbox: Tuple[float, float, float, float]) -> pd.DataFrame:
    """
    Apply a two-stage BBox filter on segment-level metadata:
    1. Fast prefilter using min/max bounds of segments.
    2. Fine-grained filter using NumPy on vals_x and vals_y.
    """
    min_lat, max_lat, min_lon, max_lon = bbox

    # Stage 1: Bounding box filter on segment metadata
    filtered = df[
        (df['max_x'] >= min_lat) & (df['min_x'] <= max_lat) &
        (df['max_y'] >= min_lon) & (df['min_y'] <= max_lon)
    ]

    if filtered.empty:
        return filtered

    # Stage 2: Internal point-level check using NumPy
    final_rows = []
    for _, row in filtered.iterrows():
        x_vals = np.array(row['vals_x'])
        y_vals = np.array(row['vals_y'])

        mask = (
            (x_vals >= min_lat) & (x_vals <= max_lat) &
            (y_vals >= min_lon) & (y_vals <= max_lon)
        )

        if np.any(mask):
            final_rows.append(row)

    return pd.DataFrame(final_rows)

def run_bbox_query_on_segments_numpy2(df: pd.DataFrame, bbox: Tuple[float, float, float, float]) -> pd.DataFrame:
    """
    Efficient BBox filter for segments:
    - Stage 1: Early rejection based on min/max bounds.
    - Stage 2: NumPy-based check for internal points (vals_x/y) within BBox.
    """
    min_lat, max_lat, min_lon, max_lon = bbox

    # Stage 1: Fast min/max bounding filter (split into stages for efficiency)
    df = df[df['max_x'] >= min_lat]
    df = df[df['min_x'] <= max_lat]
    df = df[df['max_y'] >= min_lon]
    df = df[df['min_y'] <= max_lon]

    if df.empty:
        return df

    # Stage 2: Internal NumPy-based point check
    final_rows = []
    for _, row in df.iterrows():
        try:
            x_vals = np.asarray(row['vals_x'], dtype=np.float32)
            y_vals = np.asarray(row['vals_y'], dtype=np.float32)
            if x_vals.size == 0 or y_vals.size == 0:
                continue

            in_bbox_mask = (
                (x_vals >= min_lat) & (x_vals <= max_lat) &
                (y_vals >= min_lon) & (y_vals <= max_lon)
            )

            if np.any(in_bbox_mask):
                final_rows.append(row)

        except Exception:
            continue  # skip problematic row

    return pd.DataFrame(final_rows) if final_rows else pd.DataFrame(columns=df.columns)

def run_bbox_query_on_segments_optimized(path: Path, bbox: Tuple[float, float, float, float]) -> pd.DataFrame:
    """
    Optimized BBox filtering on segment metadata:
    - Loads min/max and vals_x/y columns (refinement support)
    - Converts stringified vals_x/y to NumPy if needed
    - Returns filtered rows by min/max
    """
    min_lat, max_lat, min_lon, max_lon = bbox

    # Load only the required columns for bounding box filtering
    df = pd.read_parquet(path, columns=['min_x', 'max_x', 'min_y', 'max_y', 'vals_x', 'vals_y'])

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
    Efficient NumPy-based bounding box query assuming:
    - Data has been pre-filtered using predicate pushdown
    - `vals_x`, `vals_y` are already NumPy arrays
    """
    if df.empty:
        return df

    min_lat, max_lat, min_lon, max_lon = bbox
    results = []

    for _, row in df.iterrows():
        try:
            x_vals = row['vals_x']
            y_vals = row['vals_y']

            if not isinstance(x_vals, np.ndarray) or not isinstance(y_vals, np.ndarray):
                continue

            mask = (
                (x_vals >= min_lat) & (x_vals <= max_lat) &
                (y_vals >= min_lon) & (y_vals <= max_lon)
            )

            if np.any(mask):
                results.append(row)

        except Exception:
            continue

    return pd.DataFrame(results) if results else pd.DataFrame(columns=df.columns)

def run_bbox_query_on_segments_numpy2_optimized_v2(
    df: pd.DataFrame,
    bbox: Tuple[float, float, float, float]
) -> pd.DataFrame:
    """
    Optimized version 2 of numpy segment query with:
    - itertuples() instead of iterrows()
    - Simplified logic (no redundant checks)
    """
    if df.empty:
        return df

    min_lat, max_lat, min_lon, max_lon = bbox
    results = []

    for row in df.itertuples():
        try:
            x_vals = row.vals_x
            y_vals = row.vals_y
            
            # Vectorized numpy operations
            mask = (
                (x_vals >= min_lat) & (x_vals <= max_lat) &
                (y_vals >= min_lon) & (y_vals <= max_lon)
            )
            
            if np.any(mask):
                results.append(row)
        except Exception as e:
            print(f"[Query v2] Error processing row: {e}")
            continue

    # Convert list of namedtuples back to DataFrame
    if results:
        return pd.DataFrame.from_records(
            [r._asdict() for r in results],
            columns=df.columns
        )
    return pd.DataFrame(columns=df.columns)

def run_bbox_query_on_segments_numpy_v2(
        # run for run_bbox_query_on_segments_numpy2_optimized_v2 and load_segmented_parquet_with_pushdown_optimized_v2
    df: pd.DataFrame,
    bbox: Tuple[float, float, float, float]
) -> pd.DataFrame:
    """
    Optimized version of the two-stage filter with:
    - Single combined filter
    - itertuples()
    """
    min_lat, max_lat, min_lon, max_lon = bbox

    # Stage 1: Combined bounding box filter
    df = df[
        (df['max_x'] >= min_lat) & (df['min_x'] <= max_lat) &
        (df['max_y'] >= min_lon) & (df['min_y'] <= max_lon)
    ]

    if df.empty:
        return df

    # Stage 2: NumPy point check
    results = []
    for row in df.itertuples():
        try:
            x_vals = row.vals_x
            y_vals = row.vals_y
            mask = (
                (x_vals >= min_lat) & (x_vals <= max_lat) &
                (y_vals >= min_lon) & (y_vals <= max_lon)
            )
            if np.any(mask):
                results.append(row)
        except Exception:
            continue

    if results:
        return pd.DataFrame.from_records(
            [r._asdict() for r in results],
            columns=df.columns
        )
    return pd.DataFrame(columns=df.columns)

def refine_bbox_candidates(df: pd.DataFrame, bbox: Tuple[float, float, float, float]) -> pd.DataFrame:
    """
    Refines segment-level BBox matches by checking internal points (vals_x, vals_y).
    Returns only segments that contain at least one point inside the BBox.
    """
    if df.empty:
        return df

    min_lat, max_lat, min_lon, max_lon = bbox
    refined = []

    # Iterate through each candidate segment
    for row in df.itertuples():
        x_vals = row.vals_x
        y_vals = row.vals_y
        try:
            # Check if any point falls inside the bounding box
            mask = (
                (x_vals >= min_lat) & (x_vals <= max_lat) &
                (y_vals >= min_lon) & (y_vals <= max_lon)
            )
            if np.any(mask):
                refined.append(row)
        except Exception:
            continue  # Skip rows with invalid or malformed data

    # Return filtered DataFrame with matching segments
    if refined:
        return pd.DataFrame.from_records([r._asdict() for r in refined], columns=df.columns)
    return pd.DataFrame(columns=df.columns)
