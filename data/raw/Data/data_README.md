Geolife Trajectories Dataset – Data Description

Place your Geolife data folders (e.g., 000, 001, ...) inside this directory:

data/raw/Data/

Each folder represents a user and should contain a subdirectory named 'Trajectory/' with the corresponding .plt files.

Example valid path:

data/raw/Data/000/Trajectory/20081023025304.plt

---

This folder contains the raw Geolife GPS trajectory data collected by Microsoft Research Asia.

- Source: https://www.microsoft.com/en-us/download/details.aspx?id=52367
- Version: 1.3
- Published: 2012
- Total Users: 182
- Total Trajectories: 17,621
- Total Points: Over 24 million

Folder Structure

Each subfolder (e.g., 000, 001, ...) represents a unique user.
Each user folder contains a /Trajectory/ directory with .plt files.
Each .plt file represents a single trajectory.

File Format (.plt files)

Each trajectory file follows this structure:

Latitude, Longitude, 0, Altitude, Days_Since_1899, Date, Time
39.984702,116.318417,0,492,39757.53506,2008-11-26,12:50:29

Notes

- Only the fields Latitude, Longitude, Altitude, Date, Time are used.
- The first 6 lines in every .plt file are header and should be skipped.
- During preprocessing, files will be converted to a unified CSV or Parquet format for further processing.
