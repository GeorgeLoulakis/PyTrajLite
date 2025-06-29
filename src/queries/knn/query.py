from pathlib import Path
from typing import Tuple
import pandas as pd
import numpy as np
from time import time


# --- Vectorized Haversine formula to compute distances in meters ---
def haversine_vectorized(lat1, lon1, lat2_array, lon2_array):
    R = 6371000  # Earth's radius in meters
    lat1_rad = np.radians(lat1)
    lon1_rad = np.radians(lon1)
    lat2_rad = np.radians(lat2_array)
    lon2_rad = np.radians(lon2_array)

    dlat = lat2_rad - lat1_rad
    dlon = lon2_rad - lon1_rad

    a = (
        np.sin(dlat / 2.0) ** 2
        + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon / 2.0) ** 2
    )
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    return R * c


# --- Standard kNN query ---
def run_knn_query(
    df: pd.DataFrame, point: Tuple[float, float], k: int, lat_col="lat", lon_col="lon"
):
    """
    Computes distances from a reference point to all entries in the DataFrame using vectorized Haversine.
    Returns top-k nearest entries and the execution time.
    """
    ref_lat, ref_lon = point
    start = time()
    df = df.copy()
    df["distance"] = haversine_vectorized(
        ref_lat, ref_lon, df[lat_col].values, df[lon_col].values
    )
    result = df.nsmallest(k, "distance")[["distance"]]
    duration = time() - start
    return duration, result


# --- Early Filtering before distance computation ---
def run_knn_query_early_filter(
    df: pd.DataFrame,
    point: Tuple[float, float],
    k: int,
    lat_col: str = "lat",
    lon_col: str = "lon",
    delta: float = 1.0
) -> Tuple[int, pd.DataFrame, float]:
    """
    Perform kNN query using bounding box filtering for early candidate reduction.

    Returns:
    - Number of candidates after filtering
    - Top-k results DataFrame (sorted by distance)
    - Time taken for the filtering step (in seconds)
    """
    from time import time

    ref_lat, ref_lon = point

    t0 = time()
    # Apply bounding box filter to reduce candidate space
    filtered_df = df[
        (df[lat_col] >= ref_lat - delta) &
        (df[lat_col] <= ref_lat + delta) &
        (df[lon_col] >= ref_lon - delta) &
        (df[lon_col] <= ref_lon + delta)
    ]
    t1 = time()
    filter_time = t1 - t0
    candidate_count = len(filtered_df)

    if candidate_count == 0:
        print("[Early Filtering] No data points found in the bounding box filter.")
        return 0, pd.DataFrame(columns=["distance"]), filter_time

    print(f"[Early Filtering - {lat_col} / {lon_col}] Candidates after filter: {candidate_count}")
    results_df = run_knn_query(filtered_df, point, k, lat_col, lon_col)[1]

    return candidate_count, results_df, filter_time


# --- Compare Standard and Early Filtering Methods ---
def compare_methods(
    base_path: Path,
    fixed_path: Path,
    grid_path: Path,
    point: Tuple[float, float],
    k: int,
    delta_base: float = 1.0,
    delta_fixed: float = 70.0,
    delta_grid: float = 70.0,
):
    """
    Compares execution time of standard vs early-filtered kNN queries across three formats.
    Returns a formatted summary table as DataFrame.
    """
    base_df = pd.read_parquet(base_path)
    fixed_df = pd.read_parquet(fixed_path)
    grid_df = pd.read_parquet(grid_path)

    rows = []
    for label, df, lat_col, lon_col, delta in [
        ("Base", base_df, "lat", "lon", delta_base),
        ("Fixed", fixed_df, "centroid_lat", "centroid_lon", delta_fixed),
        ("Grid", grid_df, "centroid_lat", "centroid_lon", delta_grid),
    ]:
        std_time, _ = run_knn_query(df, point, k, lat_col, lon_col)
        _, _, filt_time = run_knn_query_early_filter(df, point, k, lat_col, lon_col, delta=delta)

        rows.append([label, "Standard", f"{std_time:.3f} sec"])
        rows.append(
            [
                label,
                "Early Filtering",
                f"{filt_time:.3f} sec" if filt_time >= 0 else "No Results",
            ]
        )

    return pd.DataFrame(rows, columns=["Dataset", "Method", "Execution Time"])


# --- Query wrapper for base parquet ---
def run_knn_query_on_parquet(path: Path, point: Tuple[float, float], k: int):
    df = pd.read_parquet(path)
    _, result = run_knn_query(df, point, k)
    result["traj_id"] = df.loc[result.index, "traj_id"].values
    result["lat"] = df.loc[result.index, "lat"].values
    result["lon"] = df.loc[result.index, "lon"].values
    return result[["traj_id", "lat", "lon", "distance"]]


# --- Query wrapper for fixed/grid segments ---
def run_knn_query_on_segments(
    df: pd.DataFrame, point: Tuple[float, float], k: int, lat_col: str, lon_col: str
):
    """
    Computes kNN from reference point to segment centroids using Haversine distance.
    Returns top-k closest segments with entity_id and coordinates.
    """
    _, result = run_knn_query(df, point, k, lat_col, lon_col)
    result["entity_id"] = df.loc[result.index, "entity_id"].values
    result[lat_col] = df.loc[result.index, lat_col].values
    result[lon_col] = df.loc[result.index, lon_col].values
    return result[["entity_id", lat_col, lon_col, "distance"]]
