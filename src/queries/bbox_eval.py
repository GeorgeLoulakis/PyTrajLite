import pandas as pd
from pathlib import Path
from typing import Tuple
from time import time


def load_base_parquet(parquet_path: Path) -> pd.DataFrame:
    """
    Load the base Parquet file that contains raw trajectory points.
    """
    return pd.read_parquet(parquet_path)


def load_csv(csv_path: Path) -> pd.DataFrame:
    """
    Load the CSV file containing trajectory points and convert timestamp to datetime.
    """
    df = pd.read_csv(csv_path)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    return df


def load_segmented_parquet(path: Path) -> pd.DataFrame:
    """
    Load a segmented Parquet file (fixed or grid) and convert stringified array columns to Python lists,
    only if necessary.
    """
    df = pd.read_parquet(path)
    for col in ['vals_x', 'vals_y', 'vals_t']:
        if pd.api.types.is_object_dtype(df[col]) and isinstance(df[col].dropna().iloc[0], str):
            df[col] = df[col].str.strip('{}').str.split(',').apply(
                lambda lst: [float(i) if 'T' not in i else i.strip("'") for i in lst]
            )
    return df


def run_bbox_query_on_points(df: pd.DataFrame, bbox: Tuple[float, float, float, float]) -> pd.DataFrame:
    """
    Apply a bounding box filter on individual points.
    """
    min_lat, max_lat, min_lon, max_lon = bbox
    return df[
        (df['lat'] >= min_lat) & (df['lat'] <= max_lat) &
        (df['lon'] >= min_lon) & (df['lon'] <= max_lon)
    ]


def run_bbox_query_on_segments(df: pd.DataFrame, bbox: Tuple[float, float, float, float]) -> pd.DataFrame:
    """
    Apply a bounding box filter on segment-level metadata (min/max coordinates).
    """
    min_lat, max_lat, min_lon, max_lon = bbox
    return df[
        (df['max_x'] >= min_lat) & (df['min_x'] <= max_lat) &
        (df['max_y'] >= min_lon) & (df['min_y'] <= max_lon)
    ]


from pathlib import Path
from typing import Tuple
from time import time
import pandas as pd


def evaluate_all_files(bbox: Tuple[float, float, float, float]):
    """
    Load and evaluate the bounding box query on all four data formats:
    - Measures query time per file
    - Computes number of matching records
    - Compares relative difference to a reference file (first successful one)
    - Optionally saves results in CSV or JSON format
    """
    base_parquet_path = Path("data/processed/trajectories.parquet")
    csv_path = Path("data/processed/trajectories.csv")
    seg_fixed_path = Path("data/processed/trajectory_segments_fixed.parquet")
    seg_grid_path = Path("data/processed/trajectory_segments_grid.parquet")

    files = {
        "Base Parquet": (load_base_parquet, run_bbox_query_on_points, base_parquet_path),
        "CSV File": (load_csv, run_bbox_query_on_points, csv_path),
        "Fixed-Size Segments": (load_segmented_parquet, run_bbox_query_on_segments, seg_fixed_path),
        "Grid-Based Segments": (load_segmented_parquet, run_bbox_query_on_segments, seg_grid_path),
    }

    # Ask user if they want to save results and in what format
    save_format = input("\nSave results? Choose format (csv / json / none): ").strip().lower()
    save_results = save_format in {"csv", "json"}

    reference_count = None
    summary = []
    output_dir = Path("data/results")
    output_dir.mkdir(parents=True, exist_ok=True)

    for name, (load_fn, query_fn, path) in files.items():
        if not path.exists():
            print(f"[{name}] File not found: {path}")
            continue

        try:
            df = load_fn(path)
            start = time()
            results = query_fn(df, bbox)
            elapsed = time() - start
            match_count = len(results)

            if reference_count is None:
                reference_count = match_count
                percent_diff = 0.0
            else:
                percent_diff = 100 * (match_count - reference_count) / reference_count

            print(f"[{name}] {match_count} matches in {elapsed:.3f} sec ({percent_diff:+.1f}% diff)")

            summary.append((name, match_count, elapsed, percent_diff))

            if save_results and not results.empty:
                safe_name = name.lower().replace(" ", "_").replace("(", "").replace(")", "").replace("-", "_")
                output_path = output_dir / f"{safe_name}_bbox_results.{save_format}"
                if save_format == "csv":
                    results.to_csv(output_path, index=False)
                elif save_format == "json":
                    results.to_json(output_path, orient="records", indent=2)

        except Exception as e:
            print(f"[{name}] Error during evaluation: {e}")

    # Print a summary table
    print("\n--- Summary ---")
    print(f"{'Format':25} {'Matches':>10} {'Time (s)':>10} {'% Diff':>10}")
    print("-" * 60)
    for name, count, t, diff in summary:
        print(f"{name:25} {count:10} {t:10.3f} {diff:10.1f}")

def run_bbox_evaluation():
    """
    Entry point for running the BBox query with user-defined coordinates.
    """
    print("Enter bounding box coordinates:")
    min_lat = float(input("  Min Latitude eg. 39.9840: "))
    max_lat = float(input("  Max Latitude eg. 39.9850: "))
    min_lon = float(input("  Min Longitude eg. 116.3160: "))
    max_lon = float(input("  Max Longitude eg. 116.3185: "))
    bbox = (min_lat, max_lat, min_lon, max_lon)
    evaluate_all_files(bbox)
