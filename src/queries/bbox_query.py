from typing import Tuple
import pandas as pd
from pathlib import Path


def run_bbox_query(
    parquet_path: str,
    bbox: Tuple[float, float, float, float]
) -> pd.DataFrame:
    """
    Execute a bounding box query on the Parquet dataset.
    """
    min_lat, max_lat, min_lon, max_lon = bbox
    df = pd.read_parquet(parquet_path)
    filtered_df = df[
        (df['lat'] >= min_lat) & (df['lat'] <= max_lat) &
        (df['lon'] >= min_lon) & (df['lon'] <= max_lon)
    ]
    return filtered_df


from time import time  # for timestamp in filename

def bbox_query():
    """
    Run BBox query by asking user input for coordinates and print top 5 results.
    Optionally save all results to a uniquely named CSV file.
    """
    try:
        print("Enter bounding box coordinates:")
        min_lat = float(input("  Min Latitude: "))
        max_lat = float(input("  Max Latitude: "))
        min_lon = float(input("  Min Longitude: "))
        max_lon = float(input("  Max Longitude: "))

        parquet_path = "data/processed/trajectories.parquet"
        results = run_bbox_query(parquet_path, (min_lat, max_lat, min_lon, max_lon))

        if results.empty:
            print("No points found in the given bounding box.")
            return

        # Sort for consistent display
        results = results.sort_values(by=["traj_id", "timestamp"])

        print(f"\nFound {len(results)} points in the given BBox.")
        print(f"Distinct trajectories: {results['traj_id'].nunique()}")

        print("\n--- Top 5 Points ---")
        for _, row in results.head(5).iterrows():
            print(f"Trajectory: {row.traj_id} → Lat: {row.lat:.5f}, Lon: {row.lon:.5f}, Time: {row.timestamp}")

        # Ask user whether to save results
        choice = input("\nDo you want to save all results to CSV? (1 = Yes, 0 = No): ").strip()
        if choice == "1":
            from pathlib import Path
            output_dir = Path("data/results")
            output_dir.mkdir(parents=True, exist_ok=True)

            timestamp = int(time())  # Unique timestamp for filename
            filename = f"bbox_lat{min_lat}_{max_lat}_lon{min_lon}_{max_lon}_{timestamp}.csv"
            output_path = output_dir / filename

            # Save selected columns to CSV
            results[["traj_id", "lat", "lon", "altitude", "timestamp"]].to_csv(output_path, index=False)
            print(f"\nAll results saved to: {output_path}")
        else:
            print("Skipped saving to CSV.")

    except Exception as e:
        print(f"Error during BBox query: {e}")