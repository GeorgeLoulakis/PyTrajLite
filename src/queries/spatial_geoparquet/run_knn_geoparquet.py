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
        save = input("Save to CSV? (yes/no): ").strip().lower()
        if save == "yes":
            from time import strftime
            filename = f"data/results/GeoParKnn_results_{strftime('%Y%m%d_%H%M%S')}.csv"
            results.to_csv(filename, index=False)
            print(f"Results saved to: {filename}")
