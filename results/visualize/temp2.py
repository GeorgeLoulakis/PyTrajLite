import sys
import os
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

# ➕ Βάλε στο path το src/
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from src.models.grid import Grid
from src.fileio import load_segments_from_parquet

# 📍 Αρχείο Parquet
parquet_path = "data/processed/trajectory_segments.parquet"
df = pd.read_parquet(parquet_path)

# 📌 Ζήτα entity_id
entity_id = input("Enter entity_id (e.g. 20081023025304): ").strip()
segments = df[df["entity_id"] == entity_id]

if segments.empty:
    print(f"No segments found for entity_id: {entity_id}")
    sys.exit()

# ➕ Συγκεντρωμένα σημεία όλων των segments
all_x = [lat for seg in segments["vals_x"] for lat in seg]
all_y = [lon for seg in segments["vals_y"] for lon in seg]

# 🔲 Δημιουργία grid
grid = Grid(min_lat=min(all_x), max_lat=max(all_x), min_lon=min(all_y), max_lon=max(all_y), cell_size=0.01)

# ➕ Υπολογισμός όλων των κελιών που χρησιμοποιούνται
used_cells = set()
for _, row in segments.iterrows():
    for lat, lon in zip(row["vals_x"], row["vals_y"]):
        used_cells.add(grid.get_cell_id(lat, lon))

# 🎨 Σχεδίαση
plt.figure(figsize=(14, 8))

# 🔲 Σχεδίαση μόνο των used grid cells (έντονα, μαύρα)
for (i, j) in used_cells:
    lon_start = grid.min_lon + j * grid.cell_size
    lat_start = grid.min_lat + i * grid.cell_size
    rect = Rectangle(
        (lon_start, lat_start),
        grid.cell_size, grid.cell_size,
        linewidth=1.5, edgecolor='black', facecolor='none', linestyle='-'
    )
    plt.gca().add_patch(rect)

# 🔁 Σχεδίαση segments με σημεία, βέλη και αριθμούς
for _, row in segments.iterrows():
    x = row["vals_y"]  # longitude
    y = row["vals_x"]  # latitude

    # Γραμμή διαδρομής
    plt.plot(x, y, label=f"Segment {row['segment_id']}")

    # Αρχή (πράσινο)
    plt.plot(x[0], y[0], 'go', markersize=6)

    # Τέλος (κόκκινο)
    plt.plot(x[-1], y[-1], 'ro', markersize=6)

    # Κόκκινο Χ στο τέλος
    plt.scatter(x[-1], y[-1], color='red', marker='x', s=80, linewidths=2)


    # Βέλος + Αριθμός Segment
    if len(x) >= 2:
        x0, y0 = x[-2], y[-2]
        x1, y1 = x[-1], y[-1]
        plt.annotate(
            f"{row['segment_id']}",
            xy=(x1, y1), xytext=(x0, y0),
            arrowprops=dict(arrowstyle="->", color="black", lw=1),
            fontsize=8, color='black', ha='left'
        )

# 🧭 Σχεδίαση τελική
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.title(f"Trajectory Segments with Grid Cells for {entity_id}")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
