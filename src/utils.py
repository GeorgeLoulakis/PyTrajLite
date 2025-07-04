import os

def display_menu():
    print("\n" + "=" * 40)
    print("     PyTrajLite - Trajectory Engine")
    print("=" * 40)
    print("1. Generate Files (Base, Fixed, Grid, GeoParquet)")
    print("2. Compare File Formats (CSV vs Parquet)")
    print("3. BBox Query on Base / Fixed / Grid")
    print("4. BBox Query on GeoParquet")
    print("5. kNN Query on Base / Fixed / Grid")
    print("6. kNN Query on GeoParquet")
    print("0. Exit")
    print("=" * 40)

def pause_and_clear():
    try:
        input("\nPress Enter to return to the menu...")
    except KeyboardInterrupt:
        pass  # Optional: allow Ctrl+C to exit gracefully

    os.system('cls' if os.name == 'nt' else 'clear')
