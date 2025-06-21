import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple, Optional
from time import time


def load_base_parquet(parquet_path: Path) -> pd.DataFrame:
    """
    Load the base Parquet file that contains raw trajectory points.
    """
    return pd.read_parquet(parquet_path)

def load_base_parquet_bbox_only(parquet_path: Path) -> pd.DataFrame:
    return pd.read_parquet(parquet_path, columns=["lat", "lon"])

def load_base_parquet_with_pushdown(parquet_path: Path, bbox: Tuple[float, float, float, float]) -> pd.DataFrame:
    """
    Load base Parquet file using predicate pushdown on lat/lon and select only necessary columns.
    """
    min_lat, max_lat, min_lon, max_lon = bbox

    filters = [
        ("lat", ">=", min_lat),
        ("lat", "<=", max_lat),
        ("lon", ">=", min_lon),
        ("lon", "<=", max_lon)
    ]

    try:
        df = pd.read_parquet(
            parquet_path,
            columns=["traj_id", "lat", "lon", "altitude", "timestamp"],
            filters=filters
        )
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        return df
    except Exception as e:
        print(f"[Base Pushdown] Error: {e}")
        return pd.DataFrame()

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

def load_segmented_parquet_with_pushdown(
    path: Path,
    bbox: Tuple[float, float, float, float],
    use_numpy_lists: bool = True
) -> pd.DataFrame:
    """
    Load Parquet file using predicate pushdown on bounding box columns.

    Parameters:
    - path: Path to the Parquet file.
    - bbox: (min_lat, max_lat, min_lon, max_lon)
    - use_numpy_lists: If True, will convert vals_x/y to NumPy arrays.

    Returns:
    - Filtered DataFrame containing only relevant row groups.
    """
    min_lat, max_lat, min_lon, max_lon = bbox

    filters = [
        ("max_x", ">=", min_lat),
        ("min_x", "<=", max_lat),
        ("max_y", ">=", min_lon),
        ("min_y", "<=", max_lon),
    ]

    columns = ["vals_x", "vals_y", "vals_t", "min_x", "max_x", "min_y", "max_y"]

    try:
        df = pd.read_parquet(
            path,
            engine="pyarrow",
            filters=filters,
            columns=columns
        )

        # Optional: ensure vals_x/vals_y are lists (convert from string if saved as stringified arrays)
        if use_numpy_lists and not df.empty:
            for col in ['vals_x', 'vals_y', 'vals_t']:
                if pd.api.types.is_object_dtype(df[col]) and isinstance(df[col].dropna().iloc[0], str):
                    df[col] = df[col].str.strip('{}').str.split(',').apply(
                        lambda lst: [float(i) if 'T' not in i else i.strip("'") for i in lst]
                    )

        return df

    except Exception as e:
        print(f"Error loading Parquet with pushdown: {e}")
        return pd.DataFrame()  # return empty if error

def load_segmented_parquet_with_pushdown_optimized(
    path: Path,
    bbox: Tuple[float, float, float, float],
    required_columns: Optional[list] = None
) -> pd.DataFrame:
    """
    Optimized Parquet loader with:
    - Predicate pushdown on min/max bbox
    - Column pruning (only required columns)
    - Automatic conversion of stringified array columns
    """
    min_lat, max_lat, min_lon, max_lon = bbox

    if required_columns is None:
        required_columns = ['segment_id', 'vals_x', 'vals_y', 'min_x', 'max_x', 'min_y', 'max_y']

    filters = [
        ("max_x", ">=", min_lat),
        ("min_x", "<=", max_lat),
        ("max_y", ">=", min_lon),
        ("min_y", "<=", max_lon)
    ]

    try:
        df = pd.read_parquet(path, columns=required_columns, filters=filters)

        # Convert stringified lists to NumPy arrays (if needed)
        for col in ['vals_x', 'vals_y']:
            if col in df.columns and isinstance(df[col].iloc[0], str):
                df[col] = df[col].str.strip('{}').str.split(',').apply(
                    lambda lst: np.array([float(x) for x in lst], dtype=np.float32)
                )

        return df

    except Exception as e:
        print(f"[Pushdown Loader] Error loading {path.name}: {str(e)}")
        return pd.DataFrame()

def load_segmented_parquet_with_pushdown_optimized_v2(
    path: Path,
    bbox: Tuple[float, float, float, float],
    required_columns: Optional[list] = None
) -> pd.DataFrame:
    """
    Optimized version 2 of pushdown loader with:
    - Vectorized string to array conversion
    - Better error handling
    """
    min_lat, max_lat, min_lon, max_lon = bbox

    if required_columns is None:
        required_columns = ['segment_id', 'vals_x', 'vals_y', 'min_x', 'max_x', 'min_y', 'max_y']

    filters = [
        ("max_x", ">=", min_lat),
        ("min_x", "<=", max_lat),
        ("max_y", ">=", min_lon),
        ("min_y", "<=", max_lon)
    ]

    try:
        # Load with pushdown and column pruning
        df = pd.read_parquet(
            path,
            columns=required_columns,
            filters=filters
        )

        # Vectorized conversion from string to NumPy array
        for col in ['vals_x', 'vals_y']:
            if col in df.columns and not df.empty and isinstance(df[col].iloc[0], str):
                df[col] = df[col].str.strip('{}').apply(
                    lambda s: np.fromstring(s, sep=',', dtype=np.float32)
                )

        return df

    except Exception as e:
        print(f"[Pushdown Loader v2] Error loading {path.name}: {str(e)}")
        return pd.DataFrame()

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

def run_bbox_query_on_segments_numpy(df: pd.DataFrame, bbox: Tuple[float, float, float, float]) -> pd.DataFrame:
    """
    Apply a two-stage BBox filter on segment-level metadata:
    1. Fast prefilter using min/max bounds of segments.
    2. Fine-grained filter using NumPy on vals_x and vals_y.
    """
    min_lat, max_lat, min_lon, max_lon = bbox

    # Stage 1: Bounding box filter on segment metadata
    filtered = df[
        (df['max_x'] >= min_lat) & (df['min_x'] <= max_lat) &
        (df['max_y'] >= min_lon) & (df['min_y'] <= max_lon)
    ]

    if filtered.empty:
        return filtered

    # Stage 2: Internal point-level check using NumPy
    final_rows = []
    for _, row in filtered.iterrows():
        x_vals = np.array(row['vals_x'])
        y_vals = np.array(row['vals_y'])

        mask = (
            (x_vals >= min_lat) & (x_vals <= max_lat) &
            (y_vals >= min_lon) & (y_vals <= max_lon)
        )

        if np.any(mask):
            final_rows.append(row)

    return pd.DataFrame(final_rows)

def run_bbox_query_on_segments_numpy2(df: pd.DataFrame, bbox: Tuple[float, float, float, float]) -> pd.DataFrame:
    """
    Efficient BBox filter for segments:
    - Stage 1: Early rejection based on min/max bounds.
    - Stage 2: NumPy-based check for internal points (vals_x/y) within BBox.
    """
    min_lat, max_lat, min_lon, max_lon = bbox

    # Stage 1: Fast min/max bounding filter (split into stages for efficiency)
    df = df[df['max_x'] >= min_lat]
    df = df[df['min_x'] <= max_lat]
    df = df[df['max_y'] >= min_lon]
    df = df[df['min_y'] <= max_lon]

    if df.empty:
        return df

    # Stage 2: Internal NumPy-based point check
    final_rows = []
    for _, row in df.iterrows():
        try:
            x_vals = np.asarray(row['vals_x'], dtype=np.float32)
            y_vals = np.asarray(row['vals_y'], dtype=np.float32)
            if x_vals.size == 0 or y_vals.size == 0:
                continue

            in_bbox_mask = (
                (x_vals >= min_lat) & (x_vals <= max_lat) &
                (y_vals >= min_lon) & (y_vals <= max_lon)
            )

            if np.any(in_bbox_mask):
                final_rows.append(row)

        except Exception:
            continue  # skip problematic row

    return pd.DataFrame(final_rows) if final_rows else pd.DataFrame(columns=df.columns)

def run_bbox_query_on_segments_optimized(path: Path, bbox: Tuple[float, float, float, float]) -> pd.DataFrame:
    """
    Optimized BBox filtering on segment metadata:
    - Loads only necessary columns (column pruning)
    - Applies bounding box filtering on segment-level metadata
    """
    min_lat, max_lat, min_lon, max_lon = bbox

    # ⚡ Only load metadata columns
    df = pd.read_parquet(path, columns=['min_x', 'max_x', 'min_y', 'max_y'])

    # 🧠 Bounding box filtering on metadata only
    return df[
        (df['max_x'] >= min_lat) & (df['min_x'] <= max_lat) &
        (df['max_y'] >= min_lon) & (df['min_y'] <= max_lon)
    ]

def run_bbox_query_on_segments_numpy2_optimized(
    df: pd.DataFrame,
    bbox: Tuple[float, float, float, float]
) -> pd.DataFrame:
    """
    Efficient NumPy-based bounding box query assuming:
    - Data has been pre-filtered using predicate pushdown
    - `vals_x`, `vals_y` are already NumPy arrays
    """
    if df.empty:
        return df

    min_lat, max_lat, min_lon, max_lon = bbox
    results = []

    for _, row in df.iterrows():
        try:
            x_vals = row['vals_x']
            y_vals = row['vals_y']

            if not isinstance(x_vals, np.ndarray) or not isinstance(y_vals, np.ndarray):
                continue

            mask = (
                (x_vals >= min_lat) & (x_vals <= max_lat) &
                (y_vals >= min_lon) & (y_vals <= max_lon)
            )

            if np.any(mask):
                results.append(row)

        except Exception:
            continue

    return pd.DataFrame(results) if results else pd.DataFrame(columns=df.columns)

def run_bbox_query_on_segments_numpy2_optimized_v2(
    df: pd.DataFrame,
    bbox: Tuple[float, float, float, float]
) -> pd.DataFrame:
    """
    Optimized version 2 of numpy segment query with:
    - itertuples() instead of iterrows()
    - Simplified logic (no redundant checks)
    """
    if df.empty:
        return df

    min_lat, max_lat, min_lon, max_lon = bbox
    results = []

    for row in df.itertuples():
        try:
            x_vals = row.vals_x
            y_vals = row.vals_y
            
            # Vectorized numpy operations
            mask = (
                (x_vals >= min_lat) & (x_vals <= max_lat) &
                (y_vals >= min_lon) & (y_vals <= max_lon)
            )
            
            if np.any(mask):
                results.append(row)
        except Exception as e:
            print(f"[Query v2] Error processing row: {e}")
            continue

    # Convert list of namedtuples back to DataFrame
    if results:
        return pd.DataFrame.from_records(
            [r._asdict() for r in results],
            columns=df.columns
        )
    return pd.DataFrame(columns=df.columns)

def run_bbox_query_on_segments_numpy_v2(
        # run for run_bbox_query_on_segments_numpy2_optimized_v2 and load_segmented_parquet_with_pushdown_optimized_v2
    df: pd.DataFrame,
    bbox: Tuple[float, float, float, float]
) -> pd.DataFrame:
    """
    Optimized version of the two-stage filter with:
    - Single combined filter
    - itertuples()
    """
    min_lat, max_lat, min_lon, max_lon = bbox

    # Stage 1: Combined bounding box filter
    df = df[
        (df['max_x'] >= min_lat) & (df['min_x'] <= max_lat) &
        (df['max_y'] >= min_lon) & (df['min_y'] <= max_lon)
    ]

    if df.empty:
        return df

    # Stage 2: NumPy point check
    results = []
    for row in df.itertuples():
        try:
            x_vals = row.vals_x
            y_vals = row.vals_y
            mask = (
                (x_vals >= min_lat) & (x_vals <= max_lat) &
                (y_vals >= min_lon) & (y_vals <= max_lon)
            )
            if np.any(mask):
                results.append(row)
        except Exception:
            continue

    if results:
        return pd.DataFrame.from_records(
            [r._asdict() for r in results],
            columns=df.columns
        )
    return pd.DataFrame(columns=df.columns)

#########################################################################################################################


from numba import njit, prange
import pyarrow.parquet as pq

#@njit(parallel=True)
def numba_bbox_filter(x_arrays, y_arrays, min_lat, max_lat, min_lon, max_lon):
    """Παράλληλο φιλτράρισμα με Numba"""
    results = np.zeros(len(x_arrays), dtype=np.bool_)
    for i in prange(len(x_arrays)):
        results[i] = np.any(
            (x_arrays[i] >= min_lat) & (x_arrays[i] <= max_lat) &
            (y_arrays[i] >= min_lon) & (y_arrays[i] <= max_lon)
        )
    return results

def run_bbox_query_numba_optimized(df: pd.DataFrame, bbox: Tuple[float, float, float, float]) -> pd.DataFrame:
    """Βελτιστοποιημένο query με Numba (χωρίς απαιτήσεις για ίδιου μεγέθους arrays)"""
    if df.empty:
        return df

    min_lat, max_lat, min_lon, max_lon = bbox

    df = df.copy()  # Avoid modifying original
    # Ensure arrays are of NumPy type
    df['vals_x'] = df['vals_x'].apply(lambda x: np.array(x, dtype=np.float32) if not isinstance(x, np.ndarray) else x)
    df['vals_y'] = df['vals_y'].apply(lambda y: np.array(y, dtype=np.float32) if not isinstance(y, np.ndarray) else y)

    x_arrays = np.array(df['vals_x'].values, dtype=object)
    y_arrays = np.array(df['vals_y'].values, dtype=object)

    mask = numba_bbox_filter(x_arrays, y_arrays, min_lat, max_lat, min_lon, max_lon)
    return df[mask]


def load_segmented_parquet_mmap(path: Path, bbox: Tuple[float, float, float, float]) -> pd.DataFrame:
    """Φόρτωση με memory mapping και column pruning"""
    min_lat, max_lat, min_lon, max_lon = bbox
    
    table = pq.read_table(
        path,
        columns=['vals_x', 'vals_y', 'min_x', 'max_x', 'min_y', 'max_y'],
        filters=[
            ("max_x", ">=", min_lat),
            ("min_x", "<=", max_lat),
            ("max_y", ">=", min_lon),
            ("min_y", "<=", max_lon)
        ],
        memory_map=True
    )
    
    df = table.to_pandas()
    
    # Convert string arrays to NumPy (αν χρειάζεται)
    for col in ['vals_x', 'vals_y']:
        if isinstance(df[col].iloc[0], str):
            df[col] = df[col].str.strip('{}').apply(
                lambda s: np.fromstring(s, sep=',', dtype=np.float32)
            )
    
    return df









#########################################################################################################################




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
        "Base Parquet (Optimized)": (load_base_parquet_bbox_only, run_bbox_query_on_points, base_parquet_path),
        "Base Parquet (Pushdown)": (lambda p: load_base_parquet_with_pushdown(p, bbox), run_bbox_query_on_points, base_parquet_path),
        "CSV File": (load_csv, run_bbox_query_on_points, csv_path),

        # Fixed-size segments (3 versions)
        "Fixed Segments (Default)": (load_segmented_parquet, run_bbox_query_on_segments, seg_fixed_path),
        "Fixed Segments (NumPy)": (load_segmented_parquet, run_bbox_query_on_segments_numpy, seg_fixed_path),
        "Fixed Segments (NumPy V2)": (load_segmented_parquet, run_bbox_query_on_segments_numpy2, seg_fixed_path),
        "Fixed Segments (Optimized)": (lambda p: p, run_bbox_query_on_segments_optimized, seg_fixed_path),
        "Fixed Segments (Pushdown)": (lambda p: load_segmented_parquet_with_pushdown(p, bbox), run_bbox_query_on_segments_numpy2, seg_fixed_path),
        "Fixed Segments (Pushdown Optimized)": (lambda p: load_segmented_parquet_with_pushdown_optimized(p, bbox),run_bbox_query_on_segments_numpy2_optimized,seg_fixed_path),

        # Grid-based segments (3 versions)
        "Grid Segments (Default)": (load_segmented_parquet, run_bbox_query_on_segments, seg_grid_path),
        "Grid Segments (NumPy)": (load_segmented_parquet, run_bbox_query_on_segments_numpy, seg_grid_path),
        "Grid Segments (NumPy V2)": (load_segmented_parquet, run_bbox_query_on_segments_numpy2, seg_grid_path),
        "Grid Segments (Optimized)": (lambda p: p, run_bbox_query_on_segments_optimized, seg_grid_path),
        "Grid Segments (Pushdown)": (lambda p: load_segmented_parquet_with_pushdown(p, bbox), run_bbox_query_on_segments_numpy2, seg_grid_path),
        "Grid Segments (Pushdown Optimized)": (lambda p: load_segmented_parquet_with_pushdown_optimized(p, bbox),run_bbox_query_on_segments_numpy2_optimized,seg_grid_path),

        # Add new optimized versions
        "Fixed Segments (Pushdown v2)": (
            lambda p: load_segmented_parquet_with_pushdown_optimized_v2(p, bbox),
            run_bbox_query_on_segments_numpy2_optimized_v2,
            seg_fixed_path
        ),
        "Grid Segments (Pushdown v2)": (
            lambda p: load_segmented_parquet_with_pushdown_optimized_v2(p, bbox),
            run_bbox_query_on_segments_numpy2_optimized_v2,
            seg_grid_path
        ),
        "Fixed Segments (Numpy v2)": (
            load_segmented_parquet,
            run_bbox_query_on_segments_numpy_v2,
            seg_fixed_path
        ),
        "Grid Segments (Numpy v2)": (
            load_segmented_parquet,
            run_bbox_query_on_segments_numpy_v2,
            seg_grid_path
        ),

        ############################## ... υπάρχον entries ...

        "Fixed Segments (Numba Ultra)": (
            lambda p: load_segmented_parquet_mmap(p, bbox),
            run_bbox_query_numba_optimized,
            seg_fixed_path
        ),
        "Grid Segments (Numba Ultra)": (
            lambda p: load_segmented_parquet_mmap(p, bbox),
            run_bbox_query_numba_optimized,
            seg_grid_path
        ),
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
            load_start = time()
            df = load_fn(path)
            load_time = time() - load_start

            query_start = time()
            results = query_fn(df, bbox)
            query_time = time() - query_start

            elapsed = load_time + query_time


            match_count = len(results)

            if reference_count is None:
                reference_count = match_count
                percent_diff = 0.0
            else:
                percent_diff = 100 * (match_count - reference_count) / reference_count

            print(f"[{name}] {match_count} matches in {elapsed:.3f} sec ({percent_diff:+.1f}% diff)")

            summary.append((name, match_count, load_time, query_time, elapsed, percent_diff))

            if save_results and not results.empty:
                safe_name = name.lower().replace(" ", "_").replace("(", "").replace(")", "").replace("-", "_")
                output_path = output_dir / f"{safe_name}_bbox_results.{save_format}"
                if save_format == "csv":
                    results.to_csv(output_path, index=False)
                elif save_format == "json":
                    results.to_json(output_path, orient="records", indent=2)

        except Exception as e:
            print(f"[{name}] Error during evaluation: {e}")

    # Print a summary table by category
    print("\n--- Summary ---")
    df_summary = pd.DataFrame(summary, columns=["Format", "Matches", "Load (s)", "Query (s)", "Total (s)", "% Diff"])

    def classify_category(format_name):
        if "Base Parquet" in format_name:
            return "Base Parquet"
        elif "CSV" in format_name:
            return "CSV"
        elif "Fixed Segments" in format_name:
            return "Fixed Segments"
        elif "Grid Segments" in format_name:
            return "Grid Segments"
        else:
            return "Other"

    df_summary["Category"] = df_summary["Format"].apply(classify_category)
    df_summary = df_summary.sort_values(by=["Category", "Total (s)"])

    grouped = df_summary.groupby("Category")

    for category, group in grouped:
        print(f"\n[{category}]")
        print(f"{'Format':<40} {'Matches':>10} {'Load (s)':>10} {'Query (s)':>12} {'Total (s)':>12} {'% Diff':>10}")
        print("-" * 95)
        for _, row in group.iterrows():
            print(f"{row['Format']:<40} {int(row['Matches']):10} {row['Load (s)']:10.3f} {row['Query (s)']:12.3f} {row['Total (s)']:12.3f} {row['% Diff']:10.1f}")


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
