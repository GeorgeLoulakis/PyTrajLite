import duckdb
import geopandas as gpd
from shapely.geometry import Point
from typing import Tuple

def load_geoparquet(path: str, bbox: Tuple[float, float, float, float]) -> gpd.GeoDataFrame:
    """
    Load filtered GeoParquet file using DuckDB and return a GeoDataFrame.
    Only rows within the bounding box are loaded to speed up performance.
    """
    min_lat, max_lat, min_lon, max_lon = bbox

    query = f"""
    SELECT lat, lon
    FROM '{path}'
    WHERE lat BETWEEN {min_lat} AND {max_lat}
    AND lon BETWEEN {min_lon} AND {max_lon}
    """

    df = duckdb.query(query).to_df()

    gdf = gpd.GeoDataFrame(
        df,
        geometry=[Point(lon, lat) for lon, lat in zip(df["lon"], df["lat"])],
        crs="EPSG:4326"
    )

    return gdf

def run_bbox_query_geoparquet(gdf: gpd.GeoDataFrame, bbox: Tuple[float, float, float, float]) -> gpd.GeoDataFrame:
    """
    Apply an additional bounding box filter on the GeoDataFrame.
    (Optional if DuckDB already does it; acts as safety post-filter)
    """
    min_lat, max_lat, min_lon, max_lon = bbox
    return gdf.cx[min_lon:max_lon, min_lat:max_lat]
