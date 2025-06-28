import duckdb
import geopandas as gpd
from typing import Tuple
from shapely.geometry import Point

def run_knn_query_geoparquet(
    geoparquet_path: str,
    ref_point: Tuple[float, float],
    k: int = 5
) -> gpd.GeoDataFrame:
    """
    Return the k nearest points to ref_point from a GeoParquet file.
    """
    import duckdb
    import geopandas as gpd
    from shapely.geometry import Point

    ref_lat, ref_lon = ref_point

    # Step 1: Rough filtering using bounding box (+-0.1° to reduce dataset)
    margin = 0.1
    query = f"""
    SELECT *
    FROM read_parquet('{geoparquet_path}')
    WHERE lat BETWEEN {ref_lat - margin} AND {ref_lat + margin}
        AND lon BETWEEN {ref_lon - margin} AND {ref_lon + margin}
    """

    df = duckdb.sql(query).to_df()

    if df.empty:
        print("No candidates found.")
        return gpd.GeoDataFrame()

    # Step 2: Create GeoDataFrame with geometry
    gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.lon, df.lat), crs="EPSG:4326")

    # Convert to UTM Zone 50N (EPSG:32650) for distance calculations
    gdf = gdf.to_crs(epsg=32650)

    # Step 3: Compute distances
    ref_point_geom = Point(ref_lon, ref_lat)
    gdf["distance"] = gdf.geometry.distance(ref_point_geom)

    # Step 4: Get k nearest
    knn = gdf.nsmallest(k, "distance")
    return knn
