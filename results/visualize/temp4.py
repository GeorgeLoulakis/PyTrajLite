import sys
import os
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

# ➕ Βάλε στο path το src/
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from src.models.grid import Grid

# 📍 Αρχείο Parquet
parquet_path = "data/processed/trajectory_segments.parquet"
df = pd.read_parquet(parquet_path)

# 📌 Ζήτα entity_id
entity_id = input("Enter entity_id (e.g. 20081023025304): ").strip()
segments = df[df["entity_id"] == entity_id]

if segments.empty:
    print(f"No segments found for entity_id: {entity_id}")
    sys.exit()

# 🔁 Συγκέντρωση όλων των σημείων όλων των segments
all_lats = []
all_lons = []

for _, row in segments.iterrows():
    all_lats.extend(row["vals_x"])  # latitude
    all_lons.extend(row["vals_y"])  # longitude

# 🔲 Δημιουργία Grid με βάση τα συνολικά όρια
grid = Grid(min_lat=min(all_lats), max_lat=max(all_lats), min_lon=min(all_lons), max_lon=max(all_lons), cell_size=0.01)

# ➕ Υπολογισμός used grid cells
used_cells = set()
for lat, lon in zip(all_lats, all_lons):
    used_cells.add(grid.get_cell_id(lat, lon))

# 🖨️ Εκτύπωση κελιών που χρησιμοποιούνται
print("\nUsed grid cells and their bounds:")
for i, j in sorted(used_cells):
    lat_min = grid.min_lat + i * grid.cell_size
    lat_max = lat_min + grid.cell_size
    lon_min = grid.min_lon + j * grid.cell_size
    lon_max = lon_min + grid.cell_size
    print(f" Cell (i={i}, j={j}): "
          f"lat ∈ [{lat_min:.6f}, {lat_max:.6f}], "
          f"lon ∈ [{lon_min:.6f}, {lon_max:.6f}]")

# 🎨 Σχεδίαση
plt.figure(figsize=(14, 8))
ax = plt.gca()

# 🔲 Σχεδίαση των used grid cells με κόκκινο χρώμα και label (i,j)
for (i, j) in used_cells:
    lon_start = grid.min_lon + j * grid.cell_size
    lat_start = grid.min_lat + i * grid.cell_size

    # Κόκκινο κελί (διαφανές)
    rect = Rectangle(
        (lon_start, lat_start),
        grid.cell_size, grid.cell_size,
        linewidth=1.5, edgecolor='black', facecolor='red', alpha=0.2
    )
    ax.add_patch(rect)

    # Ετικέτα (i, j) στο κέντρο
    center_lon = lon_start + grid.cell_size / 2
    center_lat = lat_start + grid.cell_size / 2
    ax.text(center_lon, center_lat, f"{i},{j}", ha='center', va='center', fontsize=9, color='red', fontweight='bold')

# 🔵 Σχεδίαση όλων των σημείων ως scatter
plt.scatter(all_lons, all_lats, c='blue', s=10, label="All trajectory points")

# 🟢 Πράσινο κύκλο στην αρχή
plt.plot(all_lons[0], all_lats[0], 'go', markersize=6)

# 🔴 Κόκκινο κύκλο + ❌ Χ στο τέλος
plt.plot(all_lons[-1], all_lats[-1], 'ro', markersize=6)
plt.scatter(all_lons[-1], all_lats[-1], color='red', marker='x', s=80, linewidths=2)

# 🧭 Τελικό γράφημα
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.title(f"All Points with Grid for {entity_id}")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()