#!/bin/bash
# =============================================================================
# KITTI — Extract LiDAR Only
# ~/MAVERIC/MOT · Extracts only the heaviest archive (data_tracking_velodyne.zip)
# and deletes the zip immediately after.
#
# Run after: labels, calib, and images are already extracted.
# Usage: bash extract_kitti.sh
# Run from: ~/MAVERIC/data/kitti (where your zip is)
# =============================================================================

set -e

DATA_ROOT="$HOME/MAVERIC/MOT/data/kitti"
cd "$DATA_ROOT"

echo ""
echo "==> Extracting LiDAR point clouds"
echo "    Root: $DATA_ROOT"
echo ""

FREE=$(df -h "$DATA_ROOT" | awk 'NR==2 {print $4}')
echo "  Free space at start: $FREE"
echo ""

if [ ! -f "data_tracking_velodyne.zip" ]; then
    echo "  [ERROR] data_tracking_velodyne.zip not found in $DATA_ROOT"
    exit 1
fi

SIZE=$(du -sh "data_tracking_velodyne.zip" | cut -f1)
echo "  [>>] Extracting LiDAR ($SIZE)..."
unzip -q data_tracking_velodyne.zip -d tracking/

echo "  [XX] Deleting data_tracking_velodyne.zip..."
rm -f data_tracking_velodyne.zip

FREE_FINAL=$(df -h "$DATA_ROOT" | awk 'NR==2 {print $4}')
echo "  [OK] Done.  Free space remaining: $FREE_FINAL"
echo ""
echo "  Next: python kitti_dataset.py"
