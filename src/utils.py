
import os

def display_menu():
    print("\n" + "=" * 40)
    print("     PyTrajLite - Trajectory Engine")
    print("=" * 40)
    print("1. Prepare data (create Parquet file if missing)")
    print("2. Run BBox Query")
    print("3. Compare Parquet vs CSV (size and read time)")
    print("0. Exit")
    print("=" * 40)

def pause_and_clear():
    try:
        input("\nPress Enter to return to the menu...")
    except KeyboardInterrupt:
        pass  # Optional: allow Ctrl+C to exit gracefully

    os.system('cls' if os.name == 'nt' else 'clear')
