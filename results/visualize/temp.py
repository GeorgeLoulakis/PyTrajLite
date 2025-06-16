import sys
import os
import pandas as pd
import matplotlib.pyplot as plt

# ✅ Ρύθμιση για να βρίσκει το src/
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from src.models.grid import Grid
from src.fileio import load_segments_from_parquet

# ✅ Διαδρομή προς Parquet αρχείο (τροποποίησε αν είναι αλλού)
parquet_path = "data/processed/trajectory_segments.parquet"
df = pd.read_parquet(parquet_path)

# Ζήτα από τον χρήστη το entity_id
entity_id = input("Enter entity_id (e.g. 20081023025304): ").strip()

segments = df[df["entity_id"] == entity_id]

if segments.empty:
    print(f"No segments found for entity_id: {entity_id}")
    sys.exit()

# Plot
plt.figure(figsize=(14, 8))

for i, row in segments.iterrows():
    x = row["vals_y"]  # longitude
    y = row["vals_x"]  # latitude

    # Γραμμή διαδρομής
    plt.plot(x, y, label=f"Segment {row['segment_id']}")

    # Πράσινος κύκλος στην αρχή
    plt.plot(x[0], y[0], 'go', markersize=6)

    # Κόκκινος κύκλος στο τέλος
    plt.plot(x[-1], y[-1], 'ro', markersize=6)

    # Βέλος κατεύθυνσης + αριθμός segment
    if len(x) >= 2:
        x0, y0 = x[-2], y[-2]
        x1, y1 = x[-1], y[-1]
        plt.annotate(
            f"{row['segment_id']}",
            xy=(x1, y1), xytext=(x0, y0),
            arrowprops=dict(arrowstyle="->", color="black", lw=1),
            fontsize=8, color='black', ha='left'
        )

# Ρυθμίσεις διαγράμματος
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.title(f"Trajectory Segments for {entity_id}")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
