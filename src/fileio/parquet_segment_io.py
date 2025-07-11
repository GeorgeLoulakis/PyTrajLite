import pandas as pd
from pathlib import Path
from ..models import TrajectorySegment, Grid


def save_segments_to_parquet(segments: list[TrajectorySegment], output_path: Path):
    """
    Persist a list of TrajectorySegment instances to a Parquet file.
    Each segment is serialized via its to_dict() representation.
    """
    rows = [seg.to_dict() for seg in segments]
    df = pd.DataFrame(rows)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path, index=False)

def load_segments_from_parquet(parquet_path: Path) -> list[TrajectorySegment]:
    """
    Load TrajectorySegment records from a Parquet file.
    Displays a progress indicator during loading.
    """
    df = pd.read_parquet(parquet_path)
    segments: list[TrajectorySegment] = []
    total = len(df)
    for idx, (_, row) in enumerate(df.iterrows(), start=1):
        percent = (idx / total) * 100
        print(f"\r[{percent:5.1f}%] Loading segment {row['segment_id']}...", end="")
        seg = TrajectorySegment(
            entity_id=row["entity_id"],
            segment_id=row["segment_id"],
            vals_x=row["vals_x"],
            vals_y=row["vals_y"],
            vals_t=row["vals_t"],
        )
        segments.append(seg)
    print(f"\nLoaded {len(segments)} segments from: {parquet_path}")
    return segments

def build_knn_grid_index(
    source_parquet: Path,
    target_parquet: Path,
    cell_size: float = 0.001
) -> None:
    """
    Δημιουργεί kNN-έτοιμο αρχείο (Grid segments) με στήλη `grid_cell`
    σε μορφή "i_j" (string).
    """
    import pandas as pd
    from ..models import Grid

    # 1. Διαβάζουμε το αρχείο με τα segments
    df = pd.read_parquet(source_parquet)

    # 2. Υπολογίζουμε τα global extents (ΠΡΟΣΟΧΗ στη σωστή σειρά)
    min_lat, max_lat = df["min_x"].min(), df["max_x"].max()
    min_lon, max_lon = df["min_y"].min(), df["max_y"].max()

    # 3. Φτιάχνουμε το grid
    grid = Grid(
        min_lat=min_lat,
        max_lat=max_lat,
        min_lon=min_lon,
        max_lon=max_lon,
        cell_size=cell_size
    )

    # 4. Συνάρτηση που επιστρέφει "i_j" για το πρώτο point του segment
    def cell_id_str(lat0, lon0):
        i, j = grid.get_cell_id(lat0, lon0)
        return f"{i}_{j}"

    # 5. Προσθέτουμε τη στήλη grid_cell
    df["grid_cell"] = df.apply(
        lambda row: cell_id_str(row["vals_x"][0], row["vals_y"][0]),
        axis=1
    )

    # 6. Αποθήκευση στο target αρχείο
    target_parquet.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(target_parquet, index=False)

def build_knn_fixed_index(
    source_parquet: Path,
    target_parquet: Path
) -> None:
    """
    Creates a kNN-ready file for fixed segments with additional fields:  
        - min_x, max_x, min_y, max_y: Bounding box coordinates  
        - centroid_x, centroid_y: Geometric center (centroid) of the segment  
    Used for spatial pre-filters in kNN queries.  
    """
    df = pd.read_parquet(source_parquet)

    # Calculating bounding box
    df["min_x"] = df["vals_x"].apply(min)
    df["max_x"] = df["vals_x"].apply(max)
    df["min_y"] = df["vals_y"].apply(min)
    df["max_y"] = df["vals_y"].apply(max)

    # Calculating centroid
    df["centroid_x"] = df["vals_x"].apply(lambda x: sum(x) / len(x) if len(x) > 0 else None)
    df["centroid_y"] = df["vals_y"].apply(lambda y: sum(y) / len(y) if len(y) > 0 else None)


    # Saving the modified DataFrame to the target Parquet file
    target_parquet.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(target_parquet, index=False)
