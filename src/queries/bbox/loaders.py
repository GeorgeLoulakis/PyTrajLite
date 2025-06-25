import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from pathlib import Path
from typing import Tuple, Optional

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

def load_segmented_parquet_with_pushdown_v3_np(
    path: Path,
    bbox: Tuple[float, float, float, float],
    required_columns: Optional[list] = None
) -> pd.DataFrame:
    """
    Optimized Pushdown V3 with fastest numpy string-to-array conversion
    
    Args:
        path: Path to parquet file
        bbox: Tuple of (min_lat, max_lat, min_lon, max_lon)
        required_columns: List of columns to load (None for defaults)
    
    Returns:
        Filtered DataFrame with converted arrays
    """
    min_lat, max_lat, min_lon, max_lon = bbox

    # Default columns if not specified
    if required_columns is None:
        required_columns = ['segment_id', 'vals_x', 'vals_y', 
                            'min_x', 'max_x', 'min_y', 'max_y']

    # Bounding box filters for predicate pushdown
    filters = [
        ("max_x", ">=", min_lat),
        ("min_x", "<=", max_lat),
        ("max_y", ">=", min_lon),
        ("min_y", "<=", max_lon)
    ]

    try:
        # 1. Load with pushdown filtering and column pruning
        df = pd.read_parquet(
            path,
            columns=required_columns,
            filters=filters,
            engine='pyarrow'
        )

        if df.empty:
            return df

        # 2. Ultra-fast numpy conversion
        def str_to_np_array(s: str) -> np.ndarray:
            """Inner function for fastest string-to-array conversion"""
            return np.fromstring(s.strip('{}'), sep=',', dtype=np.float32)

        # Apply to coordinate columns if they exist
        for col in ['vals_x', 'vals_y']:
            if col in df.columns and isinstance(df[col].iloc[0], str):
                df[col] = df[col].apply(str_to_np_array)

        return df

    except Exception as e:
        print(f"[Pushdown V3 NP] Error loading {path.name}: {str(e)}")
        return pd.DataFrame()

def load_segmented_parquet_mmap_pushdown(path: Path, bbox: Tuple[float, float, float, float]) -> pd.DataFrame:
    """ Load a segmented Parquet file with memory mapping and predicate pushdown."""
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
    
    # Convert string arrays to NumPy (only if necessary)
    for col in ['vals_x', 'vals_y']:
        if isinstance(df[col].iloc[0], str):
            df[col] = df[col].str.strip('{}').apply(
                lambda s: np.fromstring(s, sep=',', dtype=np.float32)
            )
    
    return df
