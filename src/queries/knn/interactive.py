from pathlib import Path
from .query import run_knn_query_on_parquet, run_knn_query_on_segments
import pandas as pd


def run_knn_general_interactive():
    base_path = Path("data/processed/trajectories.parquet")
    fixed_path = Path("data/processed/trajectory_segments_fixed_knn.parquet")
    grid_path = Path("data/processed/trajectory_segments_grid_knn.parquet")


    print("Enter coordinates for reference point:")
    lat = float(input("  Latitude: "))
    lon = float(input("  Longitude: "))
    k = int(input("  Number of nearest points (k): "))

    save_choice = input("Save to GeoJSON (y/n)? ").strip().lower()
    should_save = save_choice == "y"

    inputs = [
        ("Base Parquet", base_path, run_knn_query_on_parquet, "lat", "lon"),
        (
            "Fixed Segments",
            fixed_path,
            run_knn_query_on_segments,
            "centroid_lat",
            "centroid_lon",
        ),
        (
            "Grid Segments",
            grid_path,
            run_knn_query_on_segments,
            "centroid_lat",
            "centroid_lon",
        ),
    ]

    for label, path, method, lat_col, lon_col in inputs:
        print(f"\n--- kNN Results ({label}) ---")
        label_note = ""

        if method == run_knn_query_on_segments:
            df = pd.read_parquet(path)

            # Εάν δεν υπάρχουν οι centroid στήλες, υπολόγισε προσωρινά από start/end
            if lat_col not in df.columns or lon_col not in df.columns:
                if (
                    "start_lat" in df.columns
                    and "end_lat" in df.columns
                    and "start_lon" in df.columns
                    and "end_lon" in df.columns
                ):
                    print(
                        "Centroid columns not found — approximating with average of start/end."
                    )
                    df["centroid_lat"] = (df["start_lat"] + df["end_lat"]) / 2
                    df["centroid_lon"] = (df["start_lon"] + df["end_lon"]) / 2
                else:
                    print(
                        f"Cannot approximate centroid for {label} — missing start/end columns."
                    )
                    continue
                results = run_knn_query_on_segments(df, (lat, lon), k, lat_col, lon_col)
                label_note = " (using approximated centroids)"
            else:
                results = method(path, (lat, lon), k, lat_col, lon_col)
        else:
            results = method(path, (lat, lon), k)

        print(f"Top {k} results for {label}{label_note}:\n")
        print(results)

        if should_save:
            geojson_path = Path(
                f"data/results/knn_{label.replace(' ', '_').lower()}.geojson"
            )
            from geopandas import GeoDataFrame
            from shapely.geometry import Point

            results_gdf = results.copy()
            results_gdf["geometry"] = [
                Point(lon_val, lat_val)
                for lat_val, lon_val in zip(results[lat_col], results[lon_col])
            ]
            gdf = GeoDataFrame(results_gdf, geometry="geometry", crs="EPSG:4326")
            gdf.to_file(geojson_path, driver="GeoJSON")
            print(f"Results saved to: {geojson_path}")
