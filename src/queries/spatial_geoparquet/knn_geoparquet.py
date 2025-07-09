import time
import duckdb
import pandas as pd
from typing import Tuple

_EARTH_RADIUS = 6371000  # meters
_GRID_RESOLUTION = 0.01  # grid cell size in degrees
_GRID_RING = 1          # how many cells around


def run_knn_query_geoparquet_timed(
    geoparquet_path: str,
    ref_point: Tuple[float, float],
    k: int = 5,
    grid_resolution: float = _GRID_RESOLUTION,
    grid_ring: int = _GRID_RING
) -> Tuple[pd.DataFrame, float, float]:
    """
    Executes kNN query on a GeoParquet file using DuckDB with timing.

    Returns:
      df         : DataFrame with columns ['traj_id','lat','lon','distance'] for top-k
      load_time  : Time spent reading & computing distances (seconds)
      query_time : Time for post-processing (seconds) -- zero here
    """
    ref_lat, ref_lon = ref_point

    # compute integer bins for prefiltering
    lat_cell = int(ref_lat // grid_resolution)
    lon_cell = int(ref_lon // grid_resolution)
    lat_min = lat_cell - grid_ring
    lat_max = lat_cell + grid_ring
    lon_min = lon_cell - grid_ring
    lon_max = lon_cell + grid_ring

    # build and execute SQL with DuckDB
    start_load = time.time()
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
      WHERE floor(lat / {grid_resolution}) BETWEEN {lat_min} AND {lat_max}
        AND floor(lon / {grid_resolution}) BETWEEN {lon_min} AND {lon_max}
    )
    SELECT traj_id, lat, lon, distance
    FROM candidates
    ORDER BY distance
    LIMIT {k};
    """
    df = duckdb.sql(query).to_df()
    load_time = time.time() - start_load

    # query_time is minimal, as ordering & limit done in SQL
    query_time = 0.0
    return df, load_time, query_time
