import pandas as pd
import matplotlib.pyplot as plt

# Load Data
df = pd.read_parquet("data/processed/trajectory_segments.parquet")
entity_id = "20081023025304"
segments = df[df["entity_id"] == entity_id]

# Plotting Trajectory Segments
plt.figure(figsize=(10, 8))

for i, row in segments.iterrows():
    x = row["vals_y"]  # longitude
    y = row["vals_x"]  # latitude

    # 1. Line segment
    plt.plot(x, y, label=f"Segment {row['segment_id']}")

    if len(x) >= 2:
        # 2. Arrow from second last to last point
        x0, y0 = x[-2], y[-2]
        x1, y1 = x[-1], y[-1]
        plt.annotate(
            f"{row['segment_id']}",  # number of segment
            xy=(x1, y1), xytext=(x0, y0),
            arrowprops=dict(arrowstyle="->", color="black", lw=1),
            fontsize=8, color='black', ha='left'
        )

        # 3. Green circle at the start
        plt.plot(x[0], y[0], 'go', markersize=6)  # green circle

        # 4. Red circle at the end
        plt.plot(x[-1], y[-1], 'ro', markersize=6)  # red circle

# Labels and Title
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.title(f"Trajectory Segments for {entity_id}")
plt.grid(True)
plt.legend()
plt.show()
