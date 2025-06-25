import geopandas as gpd
from typing import Tuple

def load_geoparquet(path: str) -> gpd.GeoDataFrame:
    """
    Load a GeoParquet file using GeoPandas.
    """
    return gpd.read_parquet(path)

def run_bbox_query_geoparquet(gdf: gpd.GeoDataFrame, bbox: Tuple[float, float, float, float]) -> gpd.GeoDataFrame:
    """
    Perform a spatial BBox query using GeoPandas.
    bbox: (min_lat, max_lat, min_lon, max_lon)
    """
    min_lat, max_lat, min_lon, max_lon = bbox
    return gdf.cx[min_lon:max_lon, min_lat:max_lat]