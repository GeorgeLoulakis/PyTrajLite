from pathlib import Path
from time import time
import pandas as pd


def compare_parquet_vs_csv():
    """
    Compare size and load time between Parquet and CSV.
    If CSV doesn't exist, generate it from trajectories.
    """
    parquet_path = Path("data/processed/trajectories.parquet")
    csv_path = Path("data/processed/trajectories.csv")

    if not parquet_path.exists():
        print("Parquet file not found. Please create it first using option 1.")
        return

    if csv_path.exists():
        choice = input("CSV already exists. Use existing file? (y/n): ").strip().lower()
        if choice == 'n':
            csv_path.unlink()
            print("Old CSV deleted. A new one will be generated.")
        elif choice != 'y':
            print("Invalid input. Aborting.")
            return

    # Load from Parquet
    print("\nReading Parquet file...")
    start = time()
    df_parquet = pd.read_parquet(parquet_path)
    parquet_time = time() - start
    parquet_size = parquet_path.stat().st_size / 1024  # KB

    # Generate CSV if not exists
    if not csv_path.exists():
        print("Generating CSV from Parquet...")
        df_parquet.to_csv(csv_path, index=False)

    # Load from CSV
    print("Reading CSV file...")
    start = time()
    df_csv = pd.read_csv(csv_path)
    csv_time = time() - start
    csv_size = csv_path.stat().st_size / 1024  # KB

    # Print comparison
    print("\n--- Comparison Result ---")
    print(f"Parquet: {len(df_parquet)} rows, {parquet_size:.2f} KB, {parquet_time:.2f} sec")
    print(f"CSV:     {len(df_csv)} rows, {csv_size:.2f} KB, {csv_time:.2f} sec")
