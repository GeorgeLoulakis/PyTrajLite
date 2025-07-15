from pathlib import Path
from typing import Tuple, Optional
import pandas as pd
import numpy as np
import duckdb
import time

from src.models.grid import Grid

_EARTH_RADIUS = 6371000.0  # meters

# --- Vectorized Haversine formula to compute distances in meters ---
def haversine_vectorized(lat1, lon1, lat2_array, lon2_array):
    R = _EARTH_RADIUS
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

# --- Standard kNN query on raw points ---
def run_knn_query(
    df: pd.DataFrame, point: Tuple[float, float], k: int, lat_col="lat", lon_col="lon"
):
    ref_lat, ref_lon = point
    start = time.time()
    df = df.copy()
    df["distance"] = haversine_vectorized(
        ref_lat, ref_lon, df[lat_col].values, df[lon_col].values
    )
    # result = df.nsmallest(k, "distance")[["distance"]]
    # sort → drop exact lat/lon duplicates → take top-k
    df_sorted = df.sort_values("distance")
    df_unique = df_sorted.drop_duplicates(subset=[lat_col, lon_col], keep="first")
    result = df_unique.head(k)[["distance"]]
    duration = time.time() - start
    return duration, result

# --- kNN on base parquet of raw points ---
def run_knn_query_on_parquet(path: Path, point: Tuple[float, float], k: int):
    df = pd.read_parquet(path)
    _, result = run_knn_query(df, point, k)
    result["traj_id"] = df.loc[result.index, "traj_id"].values
    result["lat"]     = df.loc[result.index, "lat"].values
    result["lon"]     = df.loc[result.index, "lon"].values
    return result[["traj_id", "lat", "lon", "distance"]]

# --- kNN on Fixed Segments: all raw points, no dedup ---
def run_knn_query_on_fixed_segments(
    df: pd.DataFrame,
    point: Tuple[float, float],
    k: int
) -> pd.DataFrame:
    ref_lat, ref_lon = point
    rows = []
    # iterate all raw points in every segment
    for seg in df.itertuples():
        xs = np.asarray(seg.vals_x, dtype=float)
        ys = np.asarray(seg.vals_y, dtype=float)
        d  = haversine_vectorized(ref_lat, ref_lon, xs, ys)
        for x, y, dist in zip(xs, ys, d):
            rows.append({
                "traj_id": seg.entity_id,
                "lat":      float(x),
                "lon":      float(y),
                "distance": float(dist),
            })
    rows_df = pd.DataFrame(rows)
    # pick top-k nearest raw points
    return rows_df.nsmallest(k, "distance")[["traj_id", "lat", "lon", "distance"]]

# --- kNN on Grid Segments: all raw points, skip exact lat/lon duplicates ---
def run_knn_query_on_grid_segments(
    df: pd.DataFrame,
    point: Tuple[float, float],
    k: int
) -> pd.DataFrame:
    ref_lat, ref_lon = point
    rows = []
    # gather all raw points
    for seg in df.itertuples():
        xs = np.asarray(seg.vals_x, dtype=float)
        ys = np.asarray(seg.vals_y, dtype=float)
        d  = haversine_vectorized(ref_lat, ref_lon, xs, ys)
        for x, y, dist in zip(xs, ys, d):
            rows.append({
                "traj_id": seg.entity_id,
                "lat":      float(x),
                "lon":      float(y),
                "distance": float(dist),
            })

    # on-the-fly skip exact duplicates and collect until k
    out  = []
    seen = set()
    for row in sorted(rows, key=lambda r: r["distance"]):
        pos = (row["lat"], row["lon"])
        if pos in seen:
            continue
        seen.add(pos)
        out.append(row)
        if len(out) >= k:
            break

    return pd.DataFrame(out)[["traj_id", "lat", "lon", "distance"]]

# --- Wrapper dispatching to Fixed or Grid implementation ---

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
    Vectorized kNN over grid segments, με:
      1) optional coarse prefiltering σε γειτονικά cells
      2) “flatten” όλων των vals_x / vals_y
      3) vectorized Haversine
      4) sort + drop_duplicates(subset=[lat,lon])
      5) head(k)
    """
    ref_lat, ref_lon = point

    # 1) Prefiltering enabled only if grid_cell_col is provided
    if grid_cell_col is not None:
        grid = Grid(
            min_lat   = df[start_lat_col].min(),
            max_lat   = df[end_lat_col].max(),
            min_lon   = df[start_lon_col].min(),
            max_lon   = df[end_lon_col].max(),
            cell_size = cell_size
        )
        ref_cell = grid.get_cell_id(ref_lat, ref_lon)
        neighbors = [
            f"{ref_cell[0] + di}_{ref_cell[1] + dj}"
            for di in range(-grid_ring, grid_ring + 1)
            for dj in range(-grid_ring, grid_ring + 1)
        ]
        df = df[df[grid_cell_col].isin(neighbors)]

    if df.empty:
        return pd.DataFrame(columns=["traj_id", "lat", "lon", "distance"])

    # 2) Flatten segment points to global arrays
    xs = np.concatenate(df["vals_x"].values).astype(float)
    ys = np.concatenate(df["vals_y"].values).astype(float)
    counts = [len(v) for v in df["vals_x"].values]
    traj_ids = np.repeat(df["entity_id"].values, counts)

    # 3) Vectorized Haversine distance calculation
    distances = haversine_vectorized(ref_lat, ref_lon, xs, ys)

    # 4) Create DataFrame and sort by distance
    df_pts = pd.DataFrame({
        "traj_id":  traj_ids,
        "lat":      xs,
        "lon":      ys,
        "distance": distances
    })
    result = (
        df_pts
        .sort_values("distance")
        .drop_duplicates(subset=["lat", "lon"], keep="first")
        .head(k)
        .reset_index(drop=True)
    )

    return result[["traj_id", "lat", "lon", "distance"]]

def haversine_distance(lat1, lon1, lat2, lon2):
    R = 6371000  # Earth radius in meters
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    delta_phi = np.radians(lat2 - lat1)
    delta_lambda = np.radians(lon2 - lon1)

    a = np.sin(delta_phi / 2) ** 2 + \
        np.cos(phi1) * np.cos(phi2) * np.sin(delta_lambda / 2) ** 2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))

    return R * c


def run_knn_query_on_fixed_segments_vectorized(
    parquet_path: Path,
    query_point: Tuple[float, float],
    k: int,
    top_n_centroids: int = 300
) -> pd.DataFrame:
    """
    Perform optimized kNN search over fixed segments stored in Parquet using:
    - Bounding box I/O filtering (interpreting x=lat, y=lon)
    - Centroid-based pre-filtering
    - Fine-grained kNN with Haversine
    """
    ref_lat, ref_lon = query_point
    start = time.time()

    # --- Step 1: BBox Filtering (corrected for x=lat, y=lon) ---
    query = f"""
    SELECT entity_id, vals_x, vals_y,
           min_x AS min_lat, max_x AS max_lat,
           min_y AS min_lon, max_y AS max_lon,
           centroid_x AS centroid_lat, centroid_y AS centroid_lon
    FROM read_parquet('{parquet_path}')
    WHERE min_x <= {ref_lat} AND max_x >= {ref_lat}
      AND min_y <= {ref_lon} AND max_y >= {ref_lon}
    """
    df = duckdb.sql(query).to_df()
    print(f"\nSegments after BBox I/O filter: {len(df)}")

    # --- Step 2: Centroid Prefiltering ---
    if {"centroid_lat", "centroid_lon"}.issubset(df.columns):
        dx = df["centroid_lon"] - ref_lon
        dy = df["centroid_lat"] - ref_lat
        df["centroid_dist"] = np.sqrt(dx**2 + dy**2)
        df = df.nsmallest(top_n_centroids, "centroid_dist")
        print(f"Segments after centroid prefilter: {len(df)}")

    if df.empty:
        return pd.DataFrame(columns=["traj_id", "lat", "lon", "distance"])

    # --- Step 3: Fine-grained kNN over raw points (x=lat, y=lon) ---
    results = []
    for row in df.itertuples():
        for lat, lon in zip(row.vals_x, row.vals_y):  # x=lat, y=lon
            dist = haversine_distance(ref_lat, ref_lon, lat, lon)
            results.append((row.entity_id, lat, lon, dist))

    results.sort(key=lambda r: r[3])
    top_k = results[:k]

    df_result = pd.DataFrame(results, columns=["traj_id", "lat", "lon", "distance"])
    df_result = df_result.drop_duplicates(subset=["lat", "lon"])
    df_result = df_result.sort_values("distance").head(k).reset_index(drop=True)
    elapsed = time.time() - start
    print(f"kNN query (fixed+vectorized) executed in {elapsed:.3f} seconds.")
    return df_result
