import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
from pathlib import Path
from matplotlib.patches import Rectangle
from matplotlib.collections import PatchCollection

# Path configuration
PROJECT_ROOT = Path(__file__).parent.parent.parent
PARQUET_PATH = PROJECT_ROOT / "data" / "processed" / "trajectory_segments_grid.parquet"
OUTPUT_DIR = PROJECT_ROOT / "results" / "visualizations"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def load_trajectory_segments(traj_id: str) -> pd.DataFrame:
    """Load segments for a specific trajectory ID from parquet"""
    print(f"Looking for parquet file at: {PARQUET_PATH}")
    if not PARQUET_PATH.exists():
        raise FileNotFoundError(f"Parquet file not found at: {PARQUET_PATH}")
    
    df = pd.read_parquet(PARQUET_PATH)
    return df[df['entity_id'] == traj_id]

def get_grid_cells(traj_df: pd.DataFrame, cell_size: float = 0.001):
    """Calculate grid cells that cover the trajectory"""
    all_lats = np.concatenate(traj_df['vals_x'].values)
    all_lons = np.concatenate(traj_df['vals_y'].values)
    
    min_lat, max_lat = min(all_lats), max(all_lats)
    min_lon, max_lon = min(all_lons), max(all_lons)
    
    # Calculate grid boundaries
    lat_start = np.floor(min_lat / cell_size) * cell_size
    lon_start = np.floor(min_lon / cell_size) * cell_size
    lat_end = np.ceil(max_lat / cell_size) * cell_size
    lon_end = np.ceil(max_lon / cell_size) * cell_size
    
    # Generate grid cells
    lat_steps = np.arange(lat_start, lat_end + cell_size, cell_size)
    lon_steps = np.arange(lon_start, lon_end + cell_size, cell_size)
    
    return lat_steps, lon_steps, cell_size

def visualize_trajectory(traj_id: str, cell_size: float = 0.001):
    """Visualize trajectory segments with grid cells"""
    segments_df = load_trajectory_segments(traj_id)
    
    if segments_df.empty:
        print(f"No segments found for trajectory {traj_id}")
        return
    
    # Create figure
    plt.figure(figsize=(14, 12))
    ax = plt.gca()
    
    # Get grid cells that cover this trajectory
    lat_steps, lon_steps, cell_size = get_grid_cells(segments_df, cell_size)
    
    # Plot grid cells with new colors
    patches = []
    for lat in lat_steps:
        for lon in lon_steps:
            rect = Rectangle((lon, lat), cell_size, cell_size,
                           linewidth=0.7, edgecolor='#2F4F4F',  # DarkSlateGray outline
                           facecolor='#ADD8E6', alpha=0.3)  # LightBlue fill
            patches.append(rect)
    
    pc = PatchCollection(patches, match_original=True)
    ax.add_collection(pc)
    
    # Create a colormap with distinct colors for segments
    colors = plt.cm.tab20.colors
    
    # Plot each segment with different color
    for idx, (_, row) in enumerate(segments_df.iterrows()):
        # Plot trajectory segment with thicker line
        plt.plot(row['vals_y'], row['vals_x'], 
                color=colors[idx % len(colors)],
                linewidth=3.5,  # Thicker line for better visibility
                label=f'Segment {row["segment_id"]}',
                alpha=0.9, zorder=10)  # Higher zorder to appear above grid
        
        # Enhanced markers
        plt.scatter(row['vals_y'][0], row['vals_x'][0], 
                  color=colors[idx % len(colors)],
                  marker='o', s=120, edgecolor='black',
                  linewidths=1.5, zorder=15,
                  label='Start' if idx == 0 else "")
        
        plt.scatter(row['vals_y'][-1], row['vals_x'][-1],
                  color=colors[idx % len(colors)],
                  marker='s', s=120, edgecolor='black',
                  linewidths=1.5, zorder=15,
                  label='End' if idx == 0 else "")

    # Customize plot appearance
    plt.title(f'Trajectory {traj_id}\n{len(segments_df)} segments | Grid: {cell_size}°',
             fontsize=15, pad=25, weight='bold')
    
    plt.xlabel('Longitude', fontsize=13, labelpad=10)
    plt.ylabel('Latitude', fontsize=13, labelpad=10)
    
    # Improved legend
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))  # Remove duplicates
    plt.legend(by_label.values(), by_label.keys(),
              loc='upper right', framealpha=0.9)
    
    # Grid and aspect ratio
    plt.grid(True, which='both', linestyle=':', alpha=0.5)
    ax.set_aspect('equal', adjustable='datalim')
    
    # Save and show
    output_path = OUTPUT_DIR / f'trajectory_{traj_id}_grid_{cell_size}.png'
    plt.savefig(output_path, dpi=200, bbox_inches='tight', facecolor='white')
    print(f"Visualization saved to: {output_path}")
    plt.show()


def main():
    traj_id = input("Enter trajectory ID to visualize: ").strip()
    cell_size = float(input("Enter grid cell size (in degrees, e.g. 0.001): ") or 0.001)
    visualize_trajectory(traj_id, cell_size)

if __name__ == "__main__":
    main()