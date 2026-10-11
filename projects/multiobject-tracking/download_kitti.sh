#!/bin/bash
# =============================================================================
# KITTI Tracking Dataset — Download Script
# Project root: ~/MAVERIC/MOT
#
# PHASE 1 (this script): Tracking split only — camera + LiDAR (~32 GB)
#   Sequences 0000-0020 with ground truth track IDs.
#   Work with sequences 0000-0002 initially (set in kitti_dataset.py).
#
# PHASE 2 (run manually when ready for fine-tuning):
#   Detection split (~42 GB) — only needed to fine-tune YOLO11m on KITTI.
#   Commands are at the bottom of this file, commented out.
#
# Usage: bash download_kitti.sh
# =============================================================================

set -e

DATA_ROOT="$HOME/MAVERIC/MOT/data/kitti"
mkdir -p "$DATA_ROOT"
cd "$DATA_ROOT"

echo "==> Downloading KITTI Tracking split (Phase 1) to: $DATA_ROOT"
echo "    Expected total: ~32 GB. Make sure you have space."
echo ""

# -----------------------------------------------------------------------------
# PHASE 1 — Tracking split
# Camera + LiDAR + labels + calibration for sequences 0000-0020.
# This is all you need to build and validate the full MOT pipeline.
# -----------------------------------------------------------------------------

echo "[1/4] Downloading tracking LiDAR point clouds (~22 GB)..."
wget -c https://s3.eu-central-1.amazonaws.com/avg-kitti/data_tracking_velodyne.zip

echo "[2/4] Downloading tracking left camera images (~9 GB)..."
wget -c https://s3.eu-central-1.amazonaws.com/avg-kitti/data_tracking_image_2.zip

echo "[3/4] Downloading tracking labels (~2 MB)..."
wget -c https://s3.eu-central-1.amazonaws.com/avg-kitti/data_tracking_label_2.zip

echo "[4/4] Downloading tracking calibration files (~1 MB)..."
wget -c https://s3.eu-central-1.amazonaws.com/avg-kitti/data_tracking_calib.zip

# -----------------------------------------------------------------------------
# UNZIP
# -----------------------------------------------------------------------------

echo ""
echo "==> Unzipping archives..."

unzip -q data_tracking_velodyne.zip   -d tracking/
unzip -q data_tracking_image_2.zip    -d tracking/
unzip -q data_tracking_label_2.zip    -d tracking/
unzip -q data_tracking_calib.zip      -d tracking/

echo ""
echo "==> Cleaning up zip files..."
rm -f data_tracking_*.zip

echo ""
echo "Done. Final structure:"
echo ""
echo "  $DATA_ROOT/"
echo "  └── tracking/"
echo "      ├── training/     (sequences 0000-0020, with labels)"
echo "      └── testing/      (sequences 0000-0028, no labels)"
echo ""
echo "  Start with sequences [0, 1, 2] in kitti_dataset.py."
echo "  Set sequences=None when ready to use all 21 training sequences."

# -----------------------------------------------------------------------------
# PHASE 2 — Detection split (uncomment when ready to fine-tune YOLO11m)
# Only needed for training/fine-tuning the detector, not for running the tracker.
# Additional ~42 GB required.
# -----------------------------------------------------------------------------

# echo "[1/4] Downloading detection LiDAR point clouds (~29 GB)..."
# wget -c https://s3.eu-central-1.amazonaws.com/avg-kitti/data_object_velodyne.zip

# echo "[2/4] Downloading detection left camera images (~12 GB)..."
# wget -c https://s3.eu-central-1.amazonaws.com/avg-kitti/data_object_image_2.zip

# echo "[3/4] Downloading detection labels (~5 MB)..."
# wget -c https://s3.eu-central-1.amazonaws.com/avg-kitti/data_object_label_2.zip

# echo "[4/4] Downloading detection calibration files (~2 MB)..."
# wget -c https://s3.eu-central-1.amazonaws.com/avg-kitti/data_object_calib.zip

# unzip -q data_object_velodyne.zip     -d detection/
# unzip -q data_object_image_2.zip      -d detection/
# unzip -q data_object_label_2.zip      -d detection/
# unzip -q data_object_calib.zip        -d detection/
# rm -f data_object_*.zip
