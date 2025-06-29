from pathlib import Path
import pandas as pd
import numpy as np
from shapely.geometry import Point
from typing import Tuple

def run_knn_query(df: pd.DataFrame, point: Tuple[float, float], k: int):
    """Υπολογίζει απόσταση για κάθε σημείο και επιστρέφει τα k κοντινότερα."""
    ref_lat, ref_lon = point
    distances = np.sqrt((df["lat"] - ref_lat) ** 2 + (df["lon"] - ref_lon) ** 2)
    df = df.copy()
    df["distance"] = distances
    return df.nsmallest(k, "distance")[["traj_id", "lat", "lon", "distance"]]

def run_knn_query_on_parquet(path: Path, point: Tuple[float, float], k: int):
    df = pd.read_parquet(path)
    return run_knn_query(df, point, k)

def run_knn_query_on_segments(df: pd.DataFrame, point: tuple, k: int, lat_col: str, lon_col: str):
    distances = np.sqrt((df[lat_col] - point[0]) ** 2 + (df[lon_col] - point[1]) ** 2)
    df = df.copy()
    df["distance"] = distances
    return df.nsmallest(k, "distance")[["entity_id", lat_col, lon_col, "distance"]]

