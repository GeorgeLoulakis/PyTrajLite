# PyTrajLite

PyTrajLite is a Python-based undergraduate thesis project for efficient storage, transformation, and querying of spatial trajectory data using Apache Parquet and GeoParquet.
The project is inspired by the TrajParquet approach and focuses on a lightweight, modular Python implementation for trajectory-oriented data processing. It supports trajectory parsing, columnar storage, segmentation, spatial filtering, and benchmarking across multiple storage formats.

## Overview

The main goal of PyTrajLite is to study how trajectory data can be stored and queried more efficiently in columnar formats compared to traditional flat files such as CSV. The system is designed around a complete processing pipeline that starts from raw GPS trajectory files and produces multiple output formats for evaluation.

The project includes:
* Conversion of raw trajectory data to CSV and Parquet
* Fixed-size and grid-based trajectory segmentation
* GeoParquet generation for geospatial querying
* Bounding Box (BBox) query evaluation
* k-Nearest Neighbors (k-NN) querying
* Performance comparison across CSV, Parquet, segmented Parquet, and GeoParquet

## Main Features

* Raw trajectory parsing from .plt files
* Columnar storage using Apache Parquet
* GeoParquet generation with geometry support
* Fixed-size segmentation for trajectory partitioning
* Grid-based segmentation for spatial partitioning
* k-NN-ready indexed segment files
* BBox query benchmarking across multiple storage variants
* k-NN query support for raw points, segmented data, and GeoParquet
* CSV vs Parquet comparison for file size and loading performance
* Interactive CLI workflow through a menu-driven `main.py`

## Project Structure

```text
PyTrajLite/
├── data/
│   └── raw/
│       └── Data/
├── src/
│   ├── fileio/
│   ├── models/
│   ├── queries/
│   │   ├── bbox/
│   │   ├── knn/
│   │   └── spatial_geoparquet/
│   ├── segmentation/
│   ├── raw_input_loader.py
│   └── utils.py
├── main.py
└── requirements.txt
```

## Module Summary

* **src/raw_input_loader.py:** Parses raw GeoLife-style .plt trajectory files into Python objects.
* **src/models/:** Defines the core data structures (Point, Trajectory, TrajectorySegment, Grid).
* **src/segmentation/:** Implements fixed-size segmentation and grid-based segmentation.
* **src/fileio/:** Handles reading and writing of Parquet files for trajectories and segmented trajectories, including generation of k-NN-ready files.
* **src/queries/bbox/:** Contains utilities for Bounding Box query execution and benchmarking across different storage formats.
* **src/queries/knn/:** Implements k-NN queries for raw and segmented Parquet files.
* **src/queries/spatial_geoparquet/:** Supports GeoParquet-based BBox and k-NN queries using DuckDB.

## Data Pipeline

PyTrajLite follows a simple trajectory-processing pipeline:
1. Read raw GPS trajectory files from the dataset
2. Convert trajectories into a base Parquet representation
3. Export an equivalent CSV version for comparison
4. Generate segmented trajectory files (fixed-size segments, grid-based segments)
5. Build additional Parquet files optimized for k-NN queries
6. Generate GeoParquet outputs
7. Run spatial queries and benchmarks

## Generated Files

After preprocessing, the project can generate files such as:
* `data/processed/trajectories.csv`
* `data/processed/trajectories.parquet`
* `data/processed/trajectory_segments_fixed.parquet`
* `data/processed/trajectory_segments_fixed_knn.parquet`
* `data/processed/trajectory_segments_grid.parquet`
* `data/processed/trajectory_segments_grid_knn.parquet`
* `data/processed/trajectories_geoparquet_uncompressed.parquet`
* `data/processed/trajectories_geoparquet_compressed_snappy.parquet`

## Installation

Create and activate a virtual environment, then install the required packages.

### Windows (PowerShell)

```powershell
python -m venv venv
.\venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
```

### Linux / macOS

```bash
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

## Requirements

The project uses Python libraries such as:
* pandas
* pyarrow
* geopandas
* shapely
* duckdb
* numpy
* scikit-learn
* matplotlib

All required dependencies are listed in `requirements.txt`.

## How to Run

From the project root, run:

```bash
python main.py
```

The application opens an interactive menu with the main options:
1. Generate all required Parquet and GeoParquet files
2. Compare CSV and Parquet formats
3. Run BBox evaluation on standard Parquet variants
4. Run BBox evaluation on GeoParquet files
5. Run k-NN on base, fixed, and grid Parquet formats
6. Run k-NN on GeoParquet

## Query Support

### Bounding Box (BBox)

PyTrajLite supports Bounding Box queries on:
* base Parquet
* CSV
* fixed-size segmented Parquet
* grid-based segmented Parquet
* GeoParquet

The BBox evaluation workflow measures: load time, query time, total execution time, and number of matching records.

### k-Nearest Neighbors (k-NN)

k-NN queries are supported for:
* base Parquet trajectory points
* fixed segmented Parquet
* grid segmented Parquet
* GeoParquet

This allows comparison between direct point-level querying and segmented/indexed approaches.

## Academic Context

This repository was developed as part of an undergraduate thesis on efficient storage and retrieval of spatial trajectory data using Apache Parquet and GeoParquet.
The work investigates how trajectory-oriented storage strategies, such as segmentation and columnar organization, can improve the performance of spatial queries such as BBox and k-NN. The implementation is intended as a practical Python-based study of these ideas in a modular and reproducible form.

## Dataset

The project is designed to work with raw GPS trajectory data stored under: `data/raw/Data/`
In the thesis experiments, the Microsoft GeoLife GPS Trajectories dataset was used as the primary data source.

## Notes

* The project is intended for experimental and educational use.
* It emphasizes clarity, modularity, and reproducibility.
* The implementation is lightweight and suitable for further extension in research or course projects.

## Future Improvements

Possible extensions include:
* temporal filtering in addition to spatial filtering
* improved indexing strategies
* query caching
* parallel query execution
* larger-scale benchmarking on additional datasets
* integration with distributed processing frameworks

## Author

**Georgios Loulakis** - Undergraduate thesis project on spatial trajectory storage and querying with Parquet and GeoParquet.
