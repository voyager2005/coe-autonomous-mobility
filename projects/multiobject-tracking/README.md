# 🚗 MAVERIC - Multi-Object Tracking Pipeline

<div align="center">

![Python](https://img.shields.io/badge/Python-3.12-3776AB?style=flat-square&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.6-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)
![CUDA](https://img.shields.io/badge/CUDA-12.8-76B900?style=flat-square&logo=nvidia&logoColor=white)
![YOLO](https://img.shields.io/badge/YOLO11m-Ultralytics-00BFFF?style=flat-square)
![PointPillars](https://img.shields.io/badge/PointPillars-OpenPCDet-FF6B35?style=flat-square)
![License](https://img.shields.io/badge/License-MIT-yellow?style=flat-square)
![Status](https://img.shields.io/badge/Status-In%20Development-orange?style=flat-square)

**Camera + LiDAR fusion multi-object tracking for autonomous vehicles. YOLO11m + PointPillars + OSNet ReID + ByteTrack-style persistent ID assignment.**

*Part of the MAVRO Autonomous Vehicle Project · MAHE Bangalore*

</div>

---

## Overview

This repo implements an end-to-end multi-object tracking (MOT) pipeline as part of the autonomous buggy perception stack. The system accepts synchronised camera frames and LiDAR point clouds, and returns persistent per-object track IDs across frames - a prerequisite for downstream path planning and collision avoidance.

The pipeline is built around three principles:

- **Sensor fusion** - 2D camera detections and 3D LiDAR detections are fused per frame via calibration-matrix projection, giving each track both a 2D bounding box and a 3D oriented box.
- **Persistent identity** - ByteTrack-style two-pass Hungarian assignment, Kalman filter motion prediction, and OSNet appearance embeddings ensure objects retain the same ID across occlusions and missed detections.
- **Portability** - TensorRT INT8 export for YOLO11m is supported from day one. The same pipeline runs on a development RTX 4050 and deploys to a Jetson Orin with only a config change.

---

## Table of Contents

- [Architecture](#architecture)
- [Repository Structure](#repository-structure)
- [Setup](#setup)
- [Dataset](#dataset)
- [Running the Pipeline](#running-the-pipeline)
- [Evaluation](#evaluation)
- [Results](#results)
- [Export to TensorRT](#export-to-tensorrt)
- [Development Plan](#development-plan)
- [Known Issues](#known-issues)

---

## Architecture

```
Camera Frame (1242 × 375)          LiDAR Point Cloud (N × 4)
        │                                    │
        ▼                                    ▼
┌───────────────┐               ┌─────────────────────┐
│  YOLO11m      │               │  PointPillars        │
│  2D Detector  │               │  3D Detector         │
│  (TensorRT    │               │  (OpenPCDet)         │
│   INT8)       │               │                      │
└──────┬────────┘               └──────────┬───────────┘
       │ [{bbox2d, conf, class}]            │ [{bbox3d, conf, class}]
       │                                    │
       └──────────────┬─────────────────────┘
                      │  Calibration-matrix fusion
                      │  (project_3d_to_2d)
                      ▼
          ┌─────────────────────┐
          │  OSNet-x0.25        │
          │  ReID Embedder      │
          │  → (N, 512) L2-norm │
          └──────────┬──────────┘
                     │
                     ▼
          ┌─────────────────────────────────────┐
          │  MAVERICTracker                      │
          │                                      │
          │  1. Kalman predict all tracks        │
          │  2. Split: high-conf / low-conf      │
          │  3. Hungarian (IoU + cosine dist)    │
          │     - first pass:  high-conf dets    │
          │     - second pass: low-conf dets     │
          │  4. Birth tentative / kill dead      │
          │  5. Confirm after 2 consecutive hits │
          └──────────┬──────────────────────────┘
                     │  [Track(id, bbox, class, conf, bbox_3d)]
                     ▼
          ┌─────────────────────┐
          │  Renderer           │
          │  · Colour-coded IDs │
          │  · 3D wireframes    │
          │  · LiDAR overlay    │
          │  · BEV mini-panel   │
          └─────────────────────┘
                     │
                     ▼
         outputs/visualisations/
         outputs/tracks/*.json
```

| Component | Detail |
|---|---|
| 2D Detector | YOLO11m, COCO pretrained, TensorRT INT8 export supported |
| 3D Detector | PointPillars via OpenPCDet, KITTI pretrained |
| ReID | OSNet-x0.25, ImageNet pretrained, 512-d L2-normalised embeddings |
| Tracker | ByteTrack-style two-pass Hungarian + Kalman (constant velocity) |
| Association cost | `0.5 × IoU distance + 0.5 × cosine distance` |
| Dataset | KITTI Tracking split - sequences 0–2 for development |
| Input resolution | 1242 × 375 (KITTI native) |
| Hardware | RTX 4050 · CUDA 12.8 · Python 3.12 |

---

## Repository Structure

```
MAVERIC/MOT/
│
├── src/
│   ├── detection/
│   │   ├── yolo_detector.py         # YOLO11m 2D detection wrapper
│   │   └── pointpillars_detector.py # PointPillars 3D detection wrapper
│   │
│   ├── reid/
│   │   └── embedder.py              # OSNet-x0.25 appearance embedder
│   │
│   ├── tracking/
│   │   ├── kalman.py                # Pure-numpy Kalman filter
│   │   ├── track.py                 # Track dataclass (state machine)
│   │   └── tracker.py               # MAVERICTracker (ByteTrack-style)
│   │
│   ├── fusion/
│   │   └── projector.py             # LiDAR ↔ camera coordinate transforms
│   │
│   ├── visualisation/
│   │   └── renderer.py              # Frame annotation + BEV panel
│   │
│   ├── evaluation/
│   │   └── evaluate.py              # MOTA / IDF1 / MOTP via motmetrics
│   │
│   └── run_pipeline.py              # End-to-end pipeline runner
│
├── configs/
│   ├── pipeline.yaml                # All runtime configuration
│   └── kitti_tracking.yaml          # KITTI class definitions for YOLO
│
├── data/
│   └── kitti/
│       └── tracking/                # KITTI tracking split (~31 GB)
│           ├── training/
│           └── testing/
│
├── models/
│   ├── yolo11/                      # yolo11m.pt (auto-downloaded)
│   ├── pointpillars/                # pointpillars_kitti.pth
│   └── osnet_reid/                  # osnet_x0_25_imagenet.pth (auto-downloaded)
│
├── outputs/
│   ├── visualisations/              # Annotated frames + MP4 videos
│   ├── tracks/                      # Per-sequence JSON track logs
│   └── evaluation/                  # results.csv (MOTA, IDF1, MOTP)
│
├── OpenPCDet/                       # Cloned from open-mmlab (source install)
│
├── kitti_dataset.py                 # KITTI tracking dataset loader
├── requirements.txt
└── README.md
```

---

## Setup

### Prerequisites

- Python 3.12
- CUDA 12.8 · cuDNN 9.x
- Linux (Ubuntu 22.04 / 24.04 recommended)

> **Note:** This pipeline does **not** require WSL on Windows. It is developed and tested on Ubuntu native.

### Install

```bash
# 1. Create virtual environment (no conda required)
cd ~/MAVERIC/MOT
python3.12 -m venv maveric
source maveric/bin/activate

# 2. Upgrade pip
python -m pip install --upgrade pip setuptools wheel packaging

# 3. PyTorch (cu128 - stable on CUDA 12.8)
pip install torch torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/cu128

# 4. Verify GPU
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"

# 5. YOLO11 (single line - no flash-attn, no nvcc required)
pip install ultralytics

# 6. OpenPCDet / PointPillars (from source)
git clone https://github.com/open-mmlab/OpenPCDet
cd OpenPCDet
python -m pip install -r requirements.txt
python -m pip install -e . --no-build-isolation
cd ..

# 7. torchreid / OSNet
pip install git+https://github.com/KaiyangZhou/deep-person-reid.git \
    --no-build-isolation

# 8. ByteTrack
pip install git+https://github.com/ifzhang/ByteTrack.git \
    --no-build-isolation

# 9. Everything else
pip install -r requirements.txt
```

### requirements.txt

```
tensorrt>=10.9.0
torch-tensorrt>=2.6.0
numpy>=1.26.0
opencv-python>=4.11.0
Pillow>=10.4.0
scipy>=1.13.0
open3d>=0.18.0
SharedArray>=3.2.4
llvmlite>=0.43.0
numba>=0.60.0
gdown>=5.2.0
pykalman>=0.9.7
lapx>=0.5.10
faiss-gpu-cu12>=1.13.0
onnx>=1.16.0
onnxruntime-gpu>=1.22.0
pyyaml>=6.0.1
ujson>=5.10.0
pandas>=2.2.0
matplotlib>=3.9.0
motmetrics>=1.4.0
tqdm>=4.66.0
loguru>=0.7.2
rich>=13.7.0
python-dotenv>=1.0.1
```

---

## Dataset

MAVERIC uses the **KITTI Tracking split** - 21 training sequences with persistent ground truth track IDs, synchronised camera images and LiDAR point clouds.

### Download (~31 GB)

```bash
cd ~/MAVERIC/MOT/data/kitti
bash download_kitti.sh      # tracking split only (Phase 1)
```

Or extract sequentially if space is tight:

```bash
bash extract_kitti.sh       # extracts one zip at a time, deletes each after
```

### Structure after extraction

```
data/kitti/tracking/
├── training/
│   ├── image_02/     0000/ … 0020/   (camera frames)
│   ├── velodyne/     0000/ … 0020/   (LiDAR .bin files)
│   ├── label_02/     0000.txt … 0020.txt
│   └── calib/        0000.txt … 0020.txt
└── testing/
    └── image_02/     0000/ … 0028/   (no labels)
```

### Verify

```bash
python kitti_dataset.py
# Expected output:
# KITTITrackingDataset [training]: 3 sequences, 834 frames total.
# Image  : torch.Size([3, 375, 1242])
# LiDAR  : (62250, 4) points
# Labels : N objects
# Sanity check image saved → kitti_sanity_check.png
```

---

## Running the Pipeline

All configuration lives in `configs/pipeline.yaml`. Edit there - do not hardcode paths or hyperparameters in scripts.

### Full pipeline (camera + LiDAR)

```bash
python src/run_pipeline.py
```

### Camera-only mode (skip PointPillars)

```bash
python src/run_pipeline.py --no-lidar
```

### Specific sequences

```bash
python src/run_pipeline.py --sequences 0 1
```

### Skip video output

```bash
python src/run_pipeline.py --no-video
```

### Key config options

```yaml
# configs/pipeline.yaml

data:
  sequences:    null          # null = [0, 1, 2]; or specify e.g. [0]
  classes:      [Car, Van, Truck, Pedestrian, Cyclist]

tracking:
  max_age:          30        # frames before a track is deleted
  min_hits:         2         # frames before a track is confirmed
  iou_weight:       0.5       # spatial vs appearance balance
  reid_weight:      0.5
  high_conf_thresh: 0.6       # ByteTrack first-pass threshold
  ema_alpha:        0.9       # embedding smoothing

output:
  show_lidar_overlay: true
  save_video:         true
  video_fps:          10
```

### Outputs

```
outputs/
├── visualisations/
│   ├── seq_0000/
│   │   ├── frame_000000.png   # annotated frames
│   │   └── …
│   └── seq_0000.mp4           # rendered video
├── tracks/
│   └── seq_0000.json          # per-frame track log
└── evaluation/
    └── results.csv            # MOT metrics
```

---

## Evaluation

```bash
python src/evaluation/evaluate.py --sequences 0 1 2
```

Computes MOTA, MOTP, IDF1, ID switches, mostly tracked / mostly lost against KITTI ground truth labels. Results are printed as a rich table and saved to `outputs/evaluation/results.csv`.

### Target metrics

| Metric | Minimum | Good | Excellent |
|---|---|---|---|
| MOTA ↑ | 0.55 | 0.65 | 0.75+ |
| IDF1 ↑ | 0.60 | 0.70 | 0.80+ |
| MOTP ↓ | 0.25 | 0.20 | 0.15 |
| FPS (camera only) | 15 | 25 | 40+ |
| FPS (camera + LiDAR) | 8 | 12 | 18+ |

---

## Results

> Results on KITTI Tracking sequences 0–2 (in development).

| Sequence | Frames | MOTA | IDF1 | ID Switches | FPS |
|---|---|---|---|---|---|
| 0000 | 154 | - | - | - | ~10 |
| 0001 | 447 | - | - | - | ~10 |
| 0002 | 233 | - | - | - | ~10 |
| **Overall** | **834** | **-** | **-** | **-** | **~10** |

*Full evaluation in progress. Table will be updated as pipeline matures.*

---

## Export to TensorRT

Export YOLO11m to TensorRT INT8 for maximum inference speed (~3–5× faster than PyTorch).

```python
from ultralytics import YOLO

model = YOLO("models/yolo11/yolo11m.pt")
model.export(
    format="engine",
    int8=True,
    device=0,
    data="configs/kitti_tracking.yaml",
)
```

Then enable in config:

```yaml
detection:
  camera:
    use_tensorrt: true   # loads .engine file automatically
```

Expected speedup on RTX 4050:

| Mode | Latency | FPS |
|---|---|---|
| PyTorch FP32 | ~18ms | ~55 |
| TensorRT INT8 | ~6ms | ~165 |

---

## Development Plan

| Phase | Description | Status |
|---|---|---|
| 1 | Environment setup + KITTI data validation | ✅ Complete |
| 2 | Detection - YOLO11m + PointPillars + projector | ✅ Complete |
| 3 | ReID - OSNet embedding + similarity validation | ✅ Complete |
| 4 | Tracking core - Kalman + Track + MAVERICTracker | ✅ Complete |
| 5 | Visualisation + pipeline runner + evaluation | ✅ Complete |
| 6 | TensorRT INT8 export + speed optimisation | 🔵 In Progress |

Full interactive development plan: [`MAVERIC_dev_plan.html`](mot_architecture.html)

---

## Known Issues

**PointPillars - CUDA version mismatch during build**

OpenPCDet's CUDA extensions must be compiled against the same CUDA toolkit version used to build PyTorch. If `nvcc --version` reports a different version from `torch.version.cuda`, set `CUDA_HOME` explicitly before building:

```bash
export CUDA_HOME=/usr/local/cuda-12.8
export PATH=$CUDA_HOME/bin:$PATH
cd OpenPCDet && python -m pip install -e . --no-build-isolation
```

**KITTI tracking calibration format**

The tracking split uses `Tr_velo_cam` and `R_rect` (no underscores/suffixes) instead of the detection split's `Tr_velo_to_cam` and `R0_rect`. The `kitti_dataset.py` loader handles both formats automatically.

**First-frame latency**

The first frame of each run is slow (~700ms) due to CUDA JIT compilation for YOLO11m and PointPillars kernels. All subsequent frames run at normal speed (~100ms with LiDAR, ~30ms camera-only).

---

## Acknowledgements

- [Ultralytics YOLO](https://github.com/ultralytics/ultralytics) - YOLO11m detector
- [OpenPCDet](https://github.com/open-mmlab/OpenPCDet) - PointPillars 3D detector
- [deep-person-reid (torchreid)](https://github.com/KaiyangZhou/deep-person-reid) - OSNet ReID
- [ByteTrack](https://github.com/ifzhang/ByteTrack) - Two-pass association strategy
- [KITTI Vision Benchmark](https://www.cvlibs.net/datasets/kitti/) - Dataset

---

<div align="center">
<sub>MAVRO Autonomous Vehicle Project · MAHE Bangalore</sub>
</div>
