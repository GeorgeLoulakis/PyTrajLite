import numpy as np
import pandas as pd
from typing import Tuple


def memory_optimized_batch_query(df, bbox, batch_size=10000):
    min_lat, max_lat, min_lon, max_lon = bbox
    results = []
    
    # Use batching to reduce memory usage when processing large datasets
    for batch in np.array_split(df, max(1, len(df)//batch_size)):
        # Flatten the coordinate arrays across the current batch
        all_x = np.concatenate(batch['vals_x'].values)
        all_y = np.concatenate(batch['vals_y'].values)
        # Get the length of each trajectory segment in the batch
        lengths = batch['vals_x'].str.len().values
        
        # Perform a vectorized spatial filter to find all points inside the bounding box
        mask = (
            (all_x >= min_lat) & (all_x <= max_lat) &
            (all_y >= min_lon) & (all_y <= max_lon)
        )
        
        # Create a boolean mask to keep only the segments that contain at least one point inside the bounding box
        result_mask = np.zeros(len(batch), dtype=bool)
        pos = 0
        for i, length in enumerate(lengths):
            result_mask[i] = np.any(mask[pos:pos+length])
            pos += length
        
        # Append the filtered segments to the results list
        results.append(batch[result_mask])
    # Concatenate and return the filtered results
    return pd.concat(results) if results else df.iloc[:0]

def optimized_combo_query(df, bbox, batch_size=50000):
    # Unpack bounding box coordinates
    min_lat, max_lat, min_lon, max_lon = bbox
    results = []
    
    # Process the DataFrame in batches to reduce memory usage
    for i in range(0, len(df), batch_size):
        batch = df.iloc[i:i+batch_size]
        
        # Vectorized operations with direct array access
        x_arrays = batch['vals_x'].values
        y_arrays = batch['vals_y'].values
        
        batch_mask = np.zeros(len(batch), dtype=bool)
        
        for j in range(len(batch)):
            x_vals = x_arrays[j]
            y_vals = y_arrays[j]
            
            # Vectorized filtering
            mask = (
                (x_vals >= min_lat) & (x_vals <= max_lat) &
                (y_vals >= min_lon) & (y_vals <= max_lon)
            )
            
            batch_mask[j] = np.any(mask)
        
        # Append only matching segments to results
        results.append(batch[batch_mask])

    # Return the combined result
    return pd.concat(results) if results else df.iloc[:0]

def vectorized_bbox_query(df, bbox):
    """Full vectorized approach without numba, used in (Vectorized)"""
    if df.empty:
        return df
    
    min_lat, max_lat, min_lon, max_lon = bbox
    
    # Explode all points
    segments = df.explode(['vals_x', 'vals_y'])
    x_vals = segments['vals_x'].astype(float)
    y_vals = segments['vals_y'].astype(float)
    
    # Vectorized filtering
    mask = (
        (x_vals >= min_lat) & (x_vals <= max_lat) &
        (y_vals >= min_lon) & (y_vals <= max_lon)
    )
    
    # Group back to segments
    valid_segments = segments[mask].index.unique()
    return df.loc[valid_segments]
