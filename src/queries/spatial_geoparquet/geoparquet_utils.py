import duckdb
import pyarrow.parquet as pq
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point, box
from typing import Tuple

def load_geoparquet(path: str, bbox: Tuple[float, float, float, float]) -> gpd.GeoDataFrame:
    """
    Load only the points within the bounding box directly from the GeoParquet file
    using DuckDB for efficient predicate pushdown. Returns a GeoDataFrame
    containing all original columns plus geometry.
    """
    # bbox is (min_lon, min_lat, max_lon, max_lat)
    minx, miny, maxx, maxy = bbox

    # DuckDB will scan only those row groups whose lat/lon ranges intersect
    query = f"""
    SELECT *
    FROM read_parquet('{path}')
    WHERE lon BETWEEN {minx} AND {maxx}
        AND lat BETWEEN {miny} AND {maxy}
    """

    # Execute and collect just the matching rows
    df = duckdb.query(query).to_df()

    # Build geometry column and return a GeoDataFrame
    return gpd.GeoDataFrame(
        df,
        geometry=gpd.points_from_xy(df.lon, df.lat),
        crs="EPSG:4326"
    )



def run_bbox_query_geoparquet(gdf: gpd.GeoDataFrame, bbox: Tuple[float, float, float, float]) -> gpd.GeoDataFrame:
    """
    (Optional) Further refine with an exact within() check.
    Keeps every attribute and returns only those rows whose geometry
    truly lies inside the bounding box.
    """
    from shapely.geometry import box

    # bbox is still (min_lon, min_lat, max_lon, max_lat)
    minx, miny, maxx, maxy = bbox
    bbox_poly = box(minx, miny, maxx, maxy)

    return gdf[gdf.geometry.within(bbox_poly)]