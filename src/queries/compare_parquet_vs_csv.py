from pathlib import Path
from time import time
import pandas as pd

def load_file_and_measure(path: Path, file_type: str):
    """
    Load a file (Parquet or CSV), measure load time and size.

    Args:
        path (Path): Path to the file.
        file_type (str): 'parquet' or 'csv'.

    Returns:
        tuple: (number of rows, size in KB, load time in seconds)
    """
    start = time()
    if file_type == "parquet":
        df = pd.read_parquet(path)
    else:
        df = pd.read_csv(path)
    elapsed = time() - start
    size_kb = path.stat().st_size / 1024
    return len(df), size_kb, elapsed

def compare_all_formats():
    """
    Compare multiple file formats (CSV and Parquet variants) in terms of:
    - row count
    - file size (KB)
    - load time (s)
    - percentage size difference from the smallest file
    """
    base = Path("data/processed")

    files = {
        "CSV": base / "trajectories.csv",
        "Parquet (Base)": base / "trajectories.parquet",
        "Parquet (Fixed Segments)": base / "trajectory_segments_fixed.parquet",
        "Parquet (Grid Segments)": base / "trajectory_segments_grid.parquet",
    }

    results = []

    print("\n--- Comparing All Formats ---\n")

    # Load all available files and measure their size and load time
    for label, path in files.items():
        if not path.exists():
            print(f"[!] File not found: {path.name}")
            continue
        ext = path.suffix.replace('.', '')  # Determine format type by file extension
        rows, size_kb, load_time = load_file_and_measure(path, ext)
        results.append({
            "name": label,
            "rows": rows,
            "size_kb": size_kb,
            "load_time_sec": load_time
        })

    # Identify the file with the smallest size for relative comparison
    min_size = min(r["size_kb"] for r in results)

    # Print formatted comparison table
    print(f"{'Format':35} {'Rows':>8} {'Size (KB)':>12} {'Time (s)':>10} {'Difference from smallest (%)':>30}")
    print("-" * 100)
    for r in results:
        diff = 100 * (r["size_kb"] - min_size) / min_size if min_size > 0 else 0
        print(f"{r['name']:35} {r['rows']:8} {r['size_kb']:12.2f} {r['load_time_sec']:10.3f} {diff:30.1f}")

if __name__ == "__main__":
    compare_all_formats()
