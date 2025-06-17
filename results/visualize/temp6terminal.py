import sys
import os

# ➕ Πρόσθεσε το root του project (PyTrajLite) στο path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from src.models.grid import Grid
from src.fileio import load_segments_from_parquet

# 📍 Τροποποιημένο Parquet με grid-based segmentation
parquet_path = "data/processed/trajectory_segments_grid.parquet"
segments = load_segments_from_parquet(parquet_path)

# 📌 Διάλεξε entity_id
target_id = "20081023025304"
target_segments = [seg for seg in segments if seg.entity_id == target_id]

if not target_segments:
    print(f"No segments found for entity_id {target_id}")
    exit()

# 🧭 Υπολόγισε min/max για να δημιουργηθεί grid
all_x = [lat for seg in target_segments for lat in seg.vals_x]
all_y = [lon for seg in target_segments for lon in seg.vals_y]
grid = Grid(min_lat=min(all_x), max_lat=max(all_x), min_lon=min(all_y), max_lon=max(all_y), cell_size=0.01)

print(f"Grid bounds:\n  Latitude ∈ [{grid.min_lat}, {grid.max_lat}]\n  Longitude ∈ [{grid.min_lon}, {grid.max_lon}]\n")

# 🖨️ Τύπωσε τα σημεία κάθε segment
for seg in target_segments:
    print(f"\nSegment ID: {seg.segment_id}")
    for i, (lat, lon) in enumerate(zip(seg.vals_x, seg.vals_y)):
        cell = grid.get_cell_id(lat, lon)
        print(f"  Point {i:02}: (lat={lat:.6f}, lon={lon:.6f}) → cell {cell}")
