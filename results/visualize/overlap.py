import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from matplotlib.patches import Rectangle
from matplotlib.collections import PatchCollection

# Path configuration
CURRENT_FILE = Path(__file__).resolve()
PROJECT_ROOT = CURRENT_FILE.parents[2]
PARQUET_PATH = PROJECT_ROOT / "data" / "processed" / "trajectory_segments.parquet"
OUTPUT_DIR = PROJECT_ROOT / "results" / "visualizations"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def get_grid_bounds(df, cell_size):
    all_lats = np.concatenate(df['vals_x'].values)
    all_lons = np.concatenate(df['vals_y'].values)
    min_lat, max_lat = np.min(all_lats), np.max(all_lats)
    min_lon, max_lon = np.min(all_lons), np.max(all_lons)
    return min_lat, max_lat, min_lon, max_lon

def get_cell_id(lat, lon, min_lat, min_lon, cell_size):
    i = int((lat - min_lat) / cell_size)
    j = int((lon - min_lon) / cell_size)
    return (i, j)

def visualize_overlap_points(traj_id, cell_size=0.001):
    df = pd.read_parquet(PARQUET_PATH)
    df = df[df["entity_id"] == traj_id]

    if df.empty:
        print(f"No segments found for trajectory {traj_id}")
        return

    min_lat, max_lat, min_lon, max_lon = get_grid_bounds(df, cell_size)

    fig, ax = plt.subplots(figsize=(14, 12))

    # Plot grid
    lat_steps = np.arange(np.floor(min_lat / cell_size) * cell_size,
                          np.ceil(max_lat / cell_size) * cell_size + cell_size,
                          cell_size)
    lon_steps = np.arange(np.floor(min_lon / cell_size) * cell_size,
                          np.ceil(max_lon / cell_size) * cell_size + cell_size,
                          cell_size)

    patches = []
    for lat in lat_steps:
        for lon in lon_steps:
            rect = Rectangle((lon, lat), cell_size, cell_size,
                             linewidth=0.5, edgecolor='gray', facecolor='lightblue', alpha=0.2)
            patches.append(rect)
    ax.add_collection(PatchCollection(patches, match_original=True))

    # Keep track of all plotted points to detect overlaps
    seen_points = set()
    overlap_points = []

    # Plot each segment
    colors = plt.cm.tab20.colors
    for idx, row in df.iterrows():
        segment_id = row['segment_id']
        vals_x = row['vals_x']
        vals_y = row['vals_y']

        color = colors[segment_id % len(colors)]

        # Plot all points
        for i, (lat, lon) in enumerate(zip(vals_x, vals_y)):
            rounded = (round(lat, 6), round(lon, 6))
            if rounded in seen_points:
                overlap_points.append((lon, lat))
            else:
                seen_points.add(rounded)
                ax.scatter(lon, lat, color=color, s=40, alpha=0.7, zorder=5)

    # Plot overlap points in red on top
    if overlap_points:
        overlap_points = np.array(overlap_points)
        ax.scatter(overlap_points[:, 0], overlap_points[:, 1],
                   color='red', s=100, marker='x', label='Overlap Points', zorder=10)

    ax.set_title(f"Trajectory {traj_id} with Overlap Points (in red)", fontsize=14)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_aspect("equal", adjustable="datalim")
    ax.grid(True, linestyle=":", alpha=0.5)
    ax.legend()

    # Save figure
    output_path = OUTPUT_DIR / f"overlap_vis_{traj_id}.png"
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.show()
    print(f"Saved visualization to {output_path}")

def main():
    traj_id = input("Enter trajectory ID: ").strip()
    try:
        cell_size = float(input("Enter cell size (e.g. 0.001): ").strip() or 0.001)
    except ValueError:
        cell_size = 0.001
    visualize_overlap_points(traj_id, cell_size)

if __name__ == "__main__":
    main()
