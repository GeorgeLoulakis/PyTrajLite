import time
import duckdb
import pandas as pd
from typing import Tuple

# Earth radius in meters for Haversine formula
_EARTH_RADIUS = 6371000

def run_knn_query_geoparquet(
    geoparquet_path: str,
    ref_point: Tuple[float, float],
    k: int = 5
) -> pd.DataFrame:
    """
    Execute a k-Nearest Neighbors (kNN) query on a GeoParquet file using DuckDB.
    All distance computations are performed within DuckDB using the Haversine formula,
    leveraging efficient C execution and minimizing Python-side processing.

    Parameters
    ----------
    geoparquet_path : str
        Path to the GeoParquet file containing traj_id, lat, lon.
    ref_point : Tuple[float, float]
        A tuple (latitude, longitude) of the reference point.
    k : int, optional
        The number of nearest neighbors to return (default is 5).

    Returns
    -------
    DataFrame
        A DataFrame with columns [traj_id, lat, lon, distance] sorted by ascending distance.
    """
    start_time = time.time()
    ref_lat, ref_lon = ref_point

    # Rough bounding box filter to reduce computation
    margin = 0.1  # degrees latitude/longitude
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
        WHERE lat BETWEEN {ref_lat - margin} AND {ref_lat + margin}
          AND lon BETWEEN {ref_lon - margin} AND {ref_lon + margin}
    )
    SELECT traj_id, lat, lon, distance
    FROM candidates
    ORDER BY distance
    LIMIT {k};
    """

    # Execute query entirely in DuckDB
    df = duckdb.sql(query).to_df()

    elapsed = time.time() - start_time
    print(f"kNN query with DuckDB executed in {elapsed:.3f} seconds.")
    return df
