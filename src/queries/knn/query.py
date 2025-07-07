from pathlib import Path
from typing import Tuple, Optional
import pandas as pd
import numpy as np
from time import time

from src.models.grid import Grid

_EARTH_RADIUS = 6371000.0  # meters

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
    If lat_col/lon_col refer to segment endpoints (min_y/min_x or max_y/max_x),
    delegates to run_knn_query_on_segments; otherwise uses point-based kNN.
    Returns (candidate_count, top-k DataFrame, filter_time).
    """
    ref_lat, ref_lon = point
    t0 = time()
    filtered = df[
        (df[lat_col] >= ref_lat - delta) & (df[lat_col] <= ref_lat + delta) &
        (df[lon_col] >= ref_lon - delta) & (df[lon_col] <= ref_lon + delta)
    ]
    filter_time = time() - t0
    count = len(filtered)
    if count == 0:
        return 0, pd.DataFrame(columns=["distance"]), filter_time

    print(f"[Early Filtering - {lat_col}/{lon_col}] Candidates after filter: {count}")
    # segment case: lat_col/min_x etc.
    if {lat_col, lon_col} & {"min_y","max_y","min_x","max_x"}:
        topk = run_knn_query_on_segments(
            filtered,
            point,
            k,
            start_lat_col="min_x",   
            start_lon_col="min_y",
            end_lat_col="max_x",
            end_lon_col="max_y",
            # grid_resolution=delta,
            grid_ring=1
        )
    else:
        # fallback to point-based kNN
        _, topk = run_knn_query(filtered, point, k, lat_col, lon_col)

    return count, topk, filter_time

# def run_knn_query_early_filter(
#     df: pd.DataFrame,
#     point: Tuple[float, float],
#     k: int,
#     lat_col: str = "lat",
#     lon_col: str = "lon",
#     delta: float = 1.0
# ) -> Tuple[int, pd.DataFrame, float]:
#     """
#     Perform kNN query using bounding box filtering for early candidate reduction.

#     Returns:
#     - Number of candidates after filtering
#     - Top-k results DataFrame (sorted by distance)
#     - Time taken for the filtering step (in seconds)
#     """
#     from time import time

#     ref_lat, ref_lon = point

#     t0 = time()
#     # Apply bounding box filter to reduce candidate space
#     filtered_df = df[
#         (df[lat_col] >= ref_lat - delta) &
#         (df[lat_col] <= ref_lat + delta) &
#         (df[lon_col] >= ref_lon - delta) &
#         (df[lon_col] <= ref_lon + delta)
#     ]
#     t1 = time()
#     filter_time = t1 - t0
#     candidate_count = len(filtered_df)

#     if candidate_count == 0:
#         print("[Early Filtering] No data points found in the bounding box filter.")
#         return 0, pd.DataFrame(columns=["distance"]), filter_time

#     print(f"[Early Filtering - {lat_col} / {lon_col}] Candidates after filter: {candidate_count}")
#     results_df = run_knn_query(filtered_df, point, k, lat_col, lon_col)[1]

#     return candidate_count, results_df, filter_time


# --- Compare Standard and Early Filtering Methods ---
def compare_methods(base_path: Path,
                    fixed_path: Path,
                    grid_path: Path,
                    point: Tuple[float, float],
                    k: int) -> pd.DataFrame:

    # unpack query-point
    ref_lat, ref_lon = point

    # 1) Base
    base_df = pd.read_parquet(base_path)
    t0 = time()
    _, base_topk = run_knn_query(base_df, point, k, "lat", "lon")
    base_time = time() - t0

    # 2) Fixed segments
    fixed_df = pd.read_parquet(fixed_path)   # <-- ΟΧΙ df_fixed
    t0 = time()
    fixed_topk = run_knn_query_on_segments(
        fixed_df, point, k,
        start_lat_col="min_x", start_lon_col="min_y",
        end_lat_col="max_x",   end_lon_col="max_y"
    )
    fixed_time = time() - t0

    # 3) Grid segments
    grid_df = pd.read_parquet(grid_path)
    t0 = time()
    grid_topk = run_knn_query_on_segments(
        grid_df, point, k,
        start_lat_col="min_x", start_lon_col="min_y",
        end_lat_col="max_x",   end_lon_col="max_y",
        grid_cell_col="grid_cell",
        cell_size=0.001,
        grid_ring=1
    )
    grid_time = time() - t0

    return pd.DataFrame([
        {"Method": "Base",  "Candidates": len(base_topk),  "Time_s": base_time},
        {"Method": "Fixed", "Candidates": len(fixed_topk), "Time_s": fixed_time},
        {"Method": "Grid",  "Candidates": len(grid_topk),  "Time_s": grid_time},
    ])


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
    df: pd.DataFrame,
    point: Tuple[float, float],
    k: int,
    start_lat_col: str = "min_x",
    start_lon_col: str = "min_y",
    end_lat_col:   str = "max_x",
    end_lon_col:   str = "max_y",
    grid_cell_col: Optional[str] = None,
    cell_size: float = 0.001,
    grid_ring: int = 1
) -> pd.DataFrame:
    """
    Two-phase kNN for segments:
      • Fixed:   envelope filter on endpoints → ACOS-distance → top-k  
      • Grid:    precomputed grid_cell filter → envelope filter → ACOS-distance → top-k
    """
    ref_lat, ref_lon = point
    ref_lat_rad = np.radians(ref_lat)
    ref_lon_rad = np.radians(ref_lon)
    df = df.copy()

    # --- Phase 1: coarse prefilter ---
    if grid_cell_col:
        grid = Grid(
            min_lat=df[start_lat_col].min(),
            max_lat=df[end_lat_col].max(),
            min_lon=df[start_lon_col].min(),
            max_lon=df[end_lon_col].max(),
            cell_size=cell_size
        )
        ref_cell = grid.get_cell_id(ref_lat, ref_lon)

        neighbor_cells = [
            f"{ref_cell[0] + di}_{ref_cell[1] + dj}"
            for di in range(-grid_ring, grid_ring + 1)
            for dj in range(-grid_ring, grid_ring + 1)
        ]
        mask = df[grid_cell_col].isin(neighbor_cells)
    else:
        # Fixed segments: envelope filter on either start or end endpoint
        mask = (
            ((df[start_lat_col] <= ref_lat) & (df[end_lat_col]   >= ref_lat) &
             (df[start_lon_col] <= ref_lon) & (df[end_lon_col]   >= ref_lon))
        ) | (
            ((df[end_lat_col]   <= ref_lat) & (df[start_lat_col] >= ref_lat) &
             (df[end_lon_col]   <= ref_lon) & (df[start_lon_col] >= ref_lon))
        )

    candidates = df[mask].copy()
    if candidates.empty:
        return pd.DataFrame(columns=[
            "entity_id",
            start_lat_col, start_lon_col,
            end_lat_col,   end_lon_col,
            "distance"
        ])

    # --- Phase 2: exact ACOS-based distance on endpoints ---
    def acos_dist(lat_vals, lon_vals):
        lat_r = np.radians(lat_vals)
        lon_r = np.radians(lon_vals)
        cos_d = (
            np.cos(ref_lat_rad) * np.cos(lat_r) *
            np.cos(lon_r - ref_lon_rad) +
            np.sin(ref_lat_rad) * np.sin(lat_r)
        )
        cos_d = np.clip(cos_d, -1.0, 1.0)
        return _EARTH_RADIUS * np.arccos(cos_d)

    d_start = acos_dist(candidates[start_lat_col].values,
                        candidates[start_lon_col].values)
    d_end   = acos_dist(candidates[end_lat_col].values,
                        candidates[end_lon_col].values)
    candidates["distance"] = np.minimum(d_start, d_end)

    # --- Phase 3: pick top-k ---
    result = candidates.nsmallest(k, "distance")
    return result[
        ["entity_id", start_lat_col, start_lon_col, end_lat_col, end_lon_col, "distance"]
    ]

# def run_knn_query_on_segments(
#     df: pd.DataFrame, point: Tuple[float, float], k: int, lat_col: str, lon_col: str
# ):
#     """
#     Computes kNN from reference point to segment centroids using Haversine distance.
#     Returns top-k closest segments with entity_id and coordinates.
#     """
#     _, result = run_knn_query(df, point, k, lat_col, lon_col)
#     result["entity_id"] = df.loc[result.index, "entity_id"].values
#     result[lat_col] = df.loc[result.index, lat_col].values
#     result[lon_col] = df.loc[result.index, lon_col].values
#     return result[["entity_id", lat_col, lon_col, "distance"]]
