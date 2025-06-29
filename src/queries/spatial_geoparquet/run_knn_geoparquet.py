from .knn_geoparquet import run_knn_query_geoparquet

def run_knn_interactive():
    print("Enter coordinates for reference point:")
    lat = float(input("  Latitude eg. 39.9800: "))
    lon = float(input("  Longitude eg. 116.3200: "))
    k = int(input("  Number of nearest points (k): "))

    results = run_knn_query_geoparquet("data/processed/trajectories_geoparquet_compressed_snappy.parquet", (lat, lon), k)

    if results.empty:
        print("No results found.")
    else:
        print("\n--- kNN Results ---")
        print(results[["traj_id", "lat", "lon", "distance"]])
        save = input("Save to GeoJSON? (y/n): ").strip().lower()
        if save == "y":
            results.to_file("data/results/knn_results.geojson", driver="GeoJSON")
