import time
import duckdb
import pandas as pd
from typing import Tuple

# Earth radius in meters
_EARTH_RADIUS = 6371000
# Coarse grid resolution in decimal degrees (approx 1.11km per 0.01°)
_GRID_RESOLUTION = 0.01  # adjust as needed for prefilter granularity
# Number of grid rings around the reference cell
_GRID_RING = 1

def run_knn_query_geoparquet(
    geoparquet_path: str,
    ref_point: Tuple[float, float],
    k: int = 5,
    grid_resolution: float = _GRID_RESOLUTION,
    grid_ring: int = _GRID_RING
) -> pd.DataFrame:
    """
    Execute a k-Nearest Neighbors (kNN) query on a GeoParquet file using DuckDB,
    with coarse-to-fine spatial bucketing via uniform grid prefiltering.

    Parameters
    ----------
    geoparquet_path : str
        Path to the GeoParquet file containing traj_id, lat, lon.
    ref_point : Tuple[float, float]
        A tuple (latitude, longitude) of the reference point.
    k : int, optional
        Number of nearest neighbors to return (default is 5).
    grid_resolution : float, optional
        Size of each grid cell in degrees (default 0.01°).
    grid_ring : int, optional
        Number of rings around reference cell to include (default 1).

    Returns
    -------
    DataFrame
        A DataFrame with columns [traj_id, lat, lon, distance] sorted by ascending distance.
    """
    start_time = time.time()
    ref_lat, ref_lon = ref_point

    # Determine grid cell indices around reference point
    lat_cell = int(ref_lat // grid_resolution)
    lon_cell = int(ref_lon // grid_resolution)
    lat_min_bin = lat_cell - grid_ring
    lat_max_bin = lat_cell + grid_ring
    lon_min_bin = lon_cell - grid_ring
    lon_max_bin = lon_cell + grid_ring

    # SQL query: prefilter using floor-based binning and compute Haversine distance
    query = f"""
    WITH candidates AS (
      SELECT
        traj_id,
        lat,
        lon,
        {_EARTH_RADIUS} * ACOS(
          COS(RADIANS({ref_lat})) * COS(RADIANS(lat)) *
          COS(RADIANS(lon) - RADIANS({ref_lon})) +
          SIN(RADIANS({ref_lat})) * SIN(RADIANS(lat))
        ) AS distance
      FROM read_parquet('{geoparquet_path}')
      WHERE floor(lat / {grid_resolution}) BETWEEN {lat_min_bin} AND {lat_max_bin}
        AND floor(lon / {grid_resolution}) BETWEEN {lon_min_bin} AND {lon_max_bin}
    )
    SELECT traj_id, lat, lon, distance
    FROM candidates
    ORDER BY distance
    LIMIT {k};
    """

    df = duckdb.sql(query).to_df()
    elapsed = time.time() - start_time
    print(f"kNN query with grid prefilter executed in {elapsed:.3f} seconds.")
    return df
