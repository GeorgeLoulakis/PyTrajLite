import sys
import os
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

# ➕ Βάλε στο path το src/
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from src.models.grid import Grid
from src.fileio import load_segments_from_parquet  # Δεσμευμένο, αν θες μπορείς να το χρησιμοποιήσεις

# 📍 Αρχείο Parquet
parquet_path = "data/processed/trajectory_segments.parquet"
df = pd.read_parquet(parquet_path)

# 📌 Ζήτα entity_id
entity_id = input("Enter entity_id (π.χ. 20081023025304): ").strip()
segments = df[df["entity_id"] == entity_id]

if segments.empty:
    print(f"No segments found for entity_id: {entity_id}")
    sys.exit()

# ➕ Συγκέντρωση όλων των σημείων
all_x = [lat for seg in segments["vals_x"] for lat in seg]
all_y = [lon for seg in segments["vals_y"] for lon in seg]

# 🔲 Δημιουργία grid
grid = Grid(
    min_lat=min(all_x), max_lat=max(all_x),
    min_lon=min(all_y), max_lon=max(all_y),
    cell_size=0.01
)

# ➕ Υπολογισμός όλων των κελιών που χρησιμοποιούνται
used_cells = set()
for _, row in segments.iterrows():
    for lat, lon in zip(row["vals_x"], row["vals_y"]):
        used_cells.add(grid.get_cell_id(lat, lon))

# 🎨 Σχεδίαση
plt.figure(figsize=(14, 8))

# 1) Σχεδίαση μόνο των used grid cells (μαύρο περίγραμμα)
for (i, j) in used_cells:
    lon0 = grid.min_lon + j * grid.cell_size
    lat0 = grid.min_lat + i * grid.cell_size
    rect = Rectangle(
        (lon0, lat0),
        grid.cell_size, grid.cell_size,
        linewidth=1.5, edgecolor='black',
        facecolor='none'
    )
    plt.gca().add_patch(rect)

# 2) Ετοιμασία χρωμάτων: tab20 (έως 20 segments)
cmap = plt.get_cmap("tab20", len(segments))

# 3) Σχεδίαση segments με διακριτά χρώματα
for idx, row in segments.reset_index(drop=True).iterrows():
    x = row["vals_y"]  # longitude
    y = row["vals_x"]  # latitude
    color = cmap(idx)

    # γραμμή διαδρομής
    plt.plot(x, y, color=color, label=f"Seg {row['segment_id']}")

    # start (κύκλος) & end (τετράγωνο)
    plt.plot(x[0], y[0], 'o', color=color, markeredgecolor='k', markersize=6)
    plt.plot(x[-1], y[-1], 's', color=color, markeredgecolor='k', markersize=6)

    # arrow + segment_id
    if len(x) >= 2:
        plt.annotate(
            f"{row['segment_id']}",
            xy=(x[-1], y[-1]),
            xytext=(x[-2], y[-2]),
            arrowprops=dict(arrowstyle="->", color=color, lw=1),
            fontsize=8, color=color
        )

# 🧭 Τελικές ρυθμίσεις
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.title(f"Trajectory Segments with Grid Cells for {entity_id}")
plt.grid(True)
plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=8)
plt.tight_layout()
plt.show()
