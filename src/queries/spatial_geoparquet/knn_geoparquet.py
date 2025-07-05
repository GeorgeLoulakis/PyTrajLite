import time
import duckdb
import pandas as pd
import numpy as np
from typing import Tuple
from pyproj import Transformer

def run_knn_query_geoparquet(
    geoparquet_path: str,
    ref_point: Tuple[float, float],
    k: int = 5
) -> pd.DataFrame:
    """
    Execute a k-Nearest Neighbors (kNN) query on a GeoParquet file
    using optimized vectorized operations. Reads minimal columns,
    applies coordinate projection in bulk, and identifies the k closest
    points with minimal Python overhead.

    Parameters
    ----------
    geoparquet_path : str
        Path to the GeoParquet file containing point geometries.
    ref_point : Tuple[float, float]
        A tuple of (latitude, longitude) for the reference point.
    k : int, optional
        The number of nearest neighbors to return (default is 5).

    Returns
    -------
    DataFrame
        A DataFrame containing the k closest points and their distances in meters.
    """
    start_time = time.time()
    ref_lat, ref_lon = ref_point

    # Step 1: Filter candidates with bounding box predicate via DuckDB
    margin = 0.1  # degrees latitude/longitude
    query = f"""
    SELECT traj_id, lat, lon
    FROM read_parquet('{geoparquet_path}')
    WHERE lat BETWEEN {ref_lat - margin} AND {ref_lat + margin}
      AND lon BETWEEN {ref_lon - margin} AND {ref_lon + margin}
    """
    df = duckdb.sql(query).to_df()

    if df.empty:
        print("No candidates found within bounding box.")
        return pd.DataFrame(columns=["traj_id", "lat", "lon", "distance"])

    # Step 2: Bulk project coordinates to a metric CRS (UTM Zone 50N)
    transformer = Transformer.from_crs("EPSG:4326", "EPSG:32650", always_xy=True)
    xs, ys = transformer.transform(df.lon.values, df.lat.values)
    ref_x, ref_y = transformer.transform(ref_lon, ref_lat)

    # Step 3: Compute Euclidean distances vectorized
    dx = xs - ref_x
    dy = ys - ref_y
    distances = np.hypot(dx, dy)

    # Step 4: Identify k nearest indices using argpartition
    k = min(k, len(distances))
    idx = np.argpartition(distances, k)[:k]

    # Step 5: Assemble results and sort by distance
    result = df.iloc[idx].copy()
    result["distance"] = distances[idx]
    result = result.sort_values("distance").reset_index(drop=True)

    elapsed = time.time() - start_time
    print(f"kNN query executed in {elapsed:.3f} seconds.")

    return result
