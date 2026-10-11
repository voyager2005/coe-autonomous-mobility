#!/bin/bash
# =============================================================================
# MAVERIC — Project Directory Setup
# Run once after cloning / starting the project.
# Usage: bash setup_project.sh
# =============================================================================

set -e

PROJECT_ROOT="$HOME/MAVERIC/MOT"

echo "==> Setting up project structure at: $PROJECT_ROOT"
echo ""

# -----------------------------------------------------------------------------
# Directory tree
# -----------------------------------------------------------------------------

dirs=(
    # Data
    "data/kitti/tracking/training"
    "data/kitti/tracking/testing"
    # detection/ added in Phase 2 when running download_kitti.sh Phase 2 commands

    # Pretrained + fine-tuned model weights
    "models/yolo11"
    "models/pointpillars"
    "models/osnet_reid"

    # YAML config files (one per component)
    "configs"

    # Source code modules
    "src/detection"         # YOLO11 + PointPillars wrappers
    "src/reid"              # OSNet embedding
    "src/tracking"          # Kalman filter, Hungarian, track manager
    "src/fusion"            # LiDAR-camera projection utilities
    "src/visualisation"     # BEV renderer, annotation overlay
    "src/evaluation"        # MOT metrics, KITTI eval format
    "src/utils"             # Logging, config loader, misc helpers

    # Per-run outputs
    "outputs/tracks"        # JSON track logs per sequence
    "outputs/visualisations"# Annotated video frames / mp4s
    "outputs/evaluation"    # MOTA/MOTP/IDF1 result CSVs
    "outputs/tensorrt"      # Exported TensorRT engine files

    # Notebooks for exploration / debugging
    "notebooks"

    # Checkpoints from any fine-tuning runs
    "checkpoints"

    # Logs (loguru / Python logging output)
    "logs"
)

for d in "${dirs[@]}"; do
    mkdir -p "$PROJECT_ROOT/$d"
    echo "  [+] $d"
done

# -----------------------------------------------------------------------------
# Skeleton files
# -----------------------------------------------------------------------------

echo ""
echo "==> Creating skeleton files..."

# .gitignore
cat > "$PROJECT_ROOT/.gitignore" << 'EOF'
# Data (too large for git)
data/

# Model weights
models/
checkpoints/

# Outputs
outputs/
logs/

# Python
__pycache__/
*.py[cod]
*.egg-info/
.venv/
.env

# Jupyter
.ipynb_checkpoints/
notebooks/.ipynb_checkpoints/

# TensorRT engines (binary, machine-specific)
outputs/tensorrt/

# OS
.DS_Store
Thumbs.db
EOF
echo "  [+] .gitignore"

# .env template — for any API keys or paths that vary per machine
cat > "$PROJECT_ROOT/.env.example" << 'EOF'
# Copy this to .env and fill in your values
PROJECT_ROOT=$HOME/MAVERIC/MOT
KITTI_ROOT=$HOME/MAVERIC/MOT/data/kitti
CUDA_VISIBLE_DEVICES=0
EOF
echo "  [+] .env.example"

# Master config YAML
cat > "$PROJECT_ROOT/configs/pipeline.yaml" << 'EOF'
# =============================================================
# MAVERIC — Master Pipeline Config
# =============================================================

project_root: ~/MAVERIC/MOT

data:
  kitti_root:       ~/MAVERIC/MOT/data/kitti
  split:            training
  sequences:        null          # null = all sequences
  classes:          [Car, Pedestrian, Cyclist]
  img_size:         [1242, 375]

detection:
  camera:
    model:          yolo11m
    weights:        ~/MAVERIC/MOT/models/yolo11/yolo11m.pt
    conf_thresh:    0.25
    iou_thresh:     0.45
    device:         cuda:0
    use_tensorrt:   true
    precision:      int8
  lidar:
    model:          pointpillars
    weights:        ~/MAVERIC/MOT/models/pointpillars/pointpillars_kitti.pth
    voxel_size:     [0.05, 0.05, 0.1]
    point_cloud_range: [0, -39.68, -3, 69.12, 39.68, 1]
    device:         cuda:0

reid:
  model:            osnet_x0_25
  weights:          ~/MAVERIC/MOT/models/osnet_reid/osnet_x0_25.pth
  embed_dim:        512
  ema_alpha:        0.9            # embedding smoothing factor
  device:           cuda:0

tracking:
  max_age:          30             # frames before a track is deleted
  min_hits:         2              # frames before a track is confirmed
  iou_weight:       0.5            # weight of IoU in cost matrix
  reid_weight:      0.5            # weight of ReID cosine distance
  iou_threshold:    0.3            # gate: discard matches above this IoU distance
  reid_threshold:   0.4            # gate: discard matches above this cosine distance
  high_conf_thresh: 0.6            # ByteTrack: first-pass match threshold
  low_conf_thresh:  0.1            # ByteTrack: second-pass match threshold

output:
  tracks_dir:       ~/MAVERIC/MOT/outputs/tracks
  vis_dir:          ~/MAVERIC/MOT/outputs/visualisations
  eval_dir:         ~/MAVERIC/MOT/outputs/evaluation
  save_video:       true
  video_fps:        10
  show_lidar_overlay: true
EOF
echo "  [+] configs/pipeline.yaml"

# src/__init__.py files so each module is importable
for pkg in detection reid tracking fusion visualisation evaluation utils; do
    touch "$PROJECT_ROOT/src/$pkg/__init__.py"
done
touch "$PROJECT_ROOT/src/__init__.py"
echo "  [+] src/**/__init__.py"

# Minimal README
cat > "$PROJECT_ROOT/README.md" << 'EOF'
# MAVERIC
Multi-Object Tracking pipeline — Camera + LiDAR fusion on KITTI.

## Stack
- **Detector (2D):** YOLO11m (TensorRT INT8)
- **Detector (3D):** PointPillars (OpenPCDet)
- **ReID:** OSNet-x0.25 (torchreid)
- **Tracker:** ByteTrack-style Hungarian + Kalman filter
- **Dataset:** KITTI Tracking split
- **Hardware:** RTX 4050 · CUDA 13 · Python 3.12

## Setup
```bash
bash setup_project.sh       # this script — run once
bash download_kitti.sh      # download tracking split Phase 1 (~32 GB)
pip install -r requirements.txt
```

## Project Structure
```
MAVERIC/
└── MOT/
    ├── data/           KITTI tracking split (Phase 1)
    #                   detection split added in Phase 2
    ├── models/         Pretrained weights
    ├── configs/        YAML configs
    ├── src/            Pipeline source code
    ├── outputs/        Track logs, visualisations, eval results
    ├── checkpoints/    Fine-tuning checkpoints
    ├── notebooks/      Exploration notebooks
    └── logs/           Runtime logs
```
EOF
echo "  [+] README.md"

# -----------------------------------------------------------------------------
# Done
# -----------------------------------------------------------------------------

echo ""
echo "================================================================"
echo "  Project ready at: $PROJECT_ROOT"
echo ""
echo "  Next steps:"
echo "    1. bash download_kitti.sh          # pull the dataset"
echo "    2. cp .env.example .env            # set your env vars"
echo "    3. pip install -r requirements.txt # install dependencies"
echo "================================================================"
