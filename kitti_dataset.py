"""
kitti_dataset.py
================
PyTorch Dataset for the KITTI Tracking split.

Yields synchronized (image, point_cloud, labels, calib) tuples per frame,
ready to feed directly into the MOT pipeline.

Usage
-----
    from kitti_dataset import KITTITrackingDataset
    from torch.utils.data import DataLoader

    dataset = KITTITrackingDataset(
        root="/data/kitti/tracking",
        split="training",
        sequences=[0, 1, 2],   # None = all sequences
    )
    loader = DataLoader(dataset, batch_size=1, num_workers=4,
                        collate_fn=KITTITrackingDataset.collate_fn)

    for batch in loader:
        image   = batch["image"]        # (1, 3, H, W)  float32 [0,1]
        points  = batch["points"][0]    # (N, 4)        float32 [x, y, z, intensity]
        labels  = batch["labels"][0]    # list of dicts  — one per object
        calib   = batch["calib"][0]     # KITTICalib namedtuple
        meta    = batch["meta"][0]      # {"seq", "frame", "timestamp"}
"""

from __future__ import annotations

import os
import glob
from pathlib import Path
from collections import namedtuple
from typing import Optional

import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


# ---------------------------------------------------------------------------
# Calibration
# ---------------------------------------------------------------------------

KITTICalib = namedtuple("KITTICalib", [
    "P0", "P1", "P2", "P3",   # 3x4 projection matrices for cameras 0-3
    "R0_rect",                  # 3x3 rectification matrix
    "Tr_velo_to_cam",           # 3x4 velodyne → camera transform
    "Tr_imu_to_velo",           # 3x4 IMU → velodyne transform
])


def load_calib(calib_path: str) -> KITTICalib:
    """Parse a KITTI calibration .txt file into a KITTICalib namedtuple."""
    data = {}
    with open(calib_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            # Handle both "Key: values" (detection) and "Key values" (tracking)
            if ":" in line:
                key, values = line.split(":", 1)
            else:
                parts = line.split()
                key, values = parts[0], " ".join(parts[1:])
            if not values.strip():
                continue
            data[key.strip()] = np.array(
                [float(v) for v in values.split()], dtype=np.float32
            )

    def mat(key, shape, *aliases):
        """Try key and any aliases — tracking split uses different names."""
        for k in (key,) + aliases:
            if k in data:
                return data[k].reshape(shape)
        return np.zeros(shape, dtype=np.float32)

    return KITTICalib(
        P0=mat("P0", (3, 4)),
        P1=mat("P1", (3, 4)),
        P2=mat("P2", (3, 4)),
        P3=mat("P3", (3, 4)),
        # Detection split: R0_rect  |  Tracking split: R_rect
        R0_rect=mat("R0_rect", (3, 3), "R_rect"),
        # Detection split: Tr_velo_to_cam  |  Tracking split: Tr_velo_cam
        Tr_velo_to_cam=mat("Tr_velo_to_cam", (3, 4), "Tr_velo_cam"),
        # Detection split: Tr_imu_to_velo  |  Tracking split: Tr_imu_velo
        Tr_imu_to_velo=mat("Tr_imu_to_velo", (3, 4), "Tr_imu_velo"),
    )


# ---------------------------------------------------------------------------
# Label parsing
# ---------------------------------------------------------------------------

# KITTI tracking label columns (space-separated per line):
# frame  track_id  type  truncated  occluded  alpha
# left   top  right  bottom        (2D bbox in camera image)
# height width length              (3D object dimensions in metres)
# x  y  z                         (3D location in camera coords)
# rotation_y                       (yaw around Y-axis in camera coords)

LABEL_FIELDS = [
    "frame", "track_id", "type",
    "truncated", "occluded", "alpha",
    "left", "top", "right", "bottom",
    "height", "width", "length",
    "x", "y", "z",
    "rotation_y",
]


def parse_label_file(label_path: str) -> dict[int, list[dict]]:
    """
    Parse a KITTI tracking label file.

    Returns
    -------
    dict mapping frame_index → list of object dicts for that frame.
    """
    frame_labels: dict[int, list[dict]] = {}

    with open(label_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if not parts:
                continue

            obj: dict = {}
            obj["frame"]      = int(parts[0])
            obj["track_id"]   = int(parts[1])
            obj["type"]       = parts[2]          # Car, Pedestrian, Cyclist, etc.
            obj["truncated"]  = float(parts[3])
            obj["occluded"]   = int(parts[4])
            obj["alpha"]      = float(parts[5])
            obj["bbox_2d"]    = np.array([float(x) for x in parts[6:10]], dtype=np.float32)   # [l, t, r, b]
            obj["dimensions"] = np.array([float(x) for x in parts[10:13]], dtype=np.float32)  # [h, w, l]
            obj["location"]   = np.array([float(x) for x in parts[13:16]], dtype=np.float32)  # [x, y, z]
            obj["rotation_y"] = float(parts[16])

            frame_idx = obj["frame"]
            frame_labels.setdefault(frame_idx, []).append(obj)

    return frame_labels


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class KITTITrackingDataset(Dataset):
    """
    KITTI Tracking Dataset loader for camera-LiDAR synchronized data.

    Parameters
    ----------
    root : str
        Path to the tracking split root, e.g. /data/kitti/tracking/training
    split : str
        'training' or 'testing'. Testing has no labels.
    sequences : list[int] | None
        Sequence indices to include. None = all.
    classes : list[str] | None
        Filter labels to these class names. None = all.
        Common choices: ["Car"], ["Car", "Pedestrian", "Cyclist"]
    img_size : tuple[int, int] | None
        Resize images to (width, height). None = original (~1242×375).
    """

    VALID_CLASSES = {"Car", "Van", "Truck", "Pedestrian",
                     "Person_sitting", "Cyclist", "Tram", "Misc"}

    def __init__(
        self,
        root: str,
        split: str = "training",
        sequences: Optional[list[int]] = None,
        classes: Optional[list[str]] = None,
        img_size: Optional[tuple[int, int]] = None,
    ):
        self.root = Path(root) / split
        self.split = split
        self.classes = set(classes) if classes else self.VALID_CLASSES
        self.img_size = img_size
        self.has_labels = (split == "training")

        # Discover available sequences
        all_seq_dirs = sorted((self.root / "image_02").glob("*"))
        all_seqs = [int(d.name) for d in all_seq_dirs if d.is_dir()]

        if sequences is not None:
            missing = set(sequences) - set(all_seqs)
            if missing:
                raise ValueError(f"Sequences not found: {missing}. Available: {all_seqs}")
            self.sequences = sorted(sequences)
        else:
            self.sequences = all_seqs

        # Build flat index: list of (seq, frame) tuples
        self.index: list[tuple[int, int]] = []
        self._label_cache: dict[int, dict] = {}

        for seq in self.sequences:
            img_dir = self.root / "image_02" / f"{seq:04d}"
            frames = sorted(int(p.stem) for p in img_dir.glob("*.png"))
            self.index.extend((seq, f) for f in frames)

            # Pre-load labels into cache (they're tiny text files)
            if self.has_labels:
                label_path = self.root / "label_02" / f"{seq:04d}.txt"
                if label_path.exists():
                    self._label_cache[seq] = parse_label_file(str(label_path))
                else:
                    self._label_cache[seq] = {}

        print(
            f"KITTITrackingDataset [{split}]: "
            f"{len(self.sequences)} sequences, {len(self.index)} frames total."
        )

    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, idx: int) -> dict:
        seq, frame = self.index[idx]

        image   = self._load_image(seq, frame)
        points  = self._load_velodyne(seq, frame)
        labels  = self._load_labels(seq, frame)
        calib   = self._load_calib(seq)

        return {
            "image":  image,
            "points": points,
            "labels": labels,
            "calib":  calib,
            "meta":   {"seq": seq, "frame": frame},
        }

    # ------------------------------------------------------------------
    # Loaders
    # ------------------------------------------------------------------

    def _load_image(self, seq: int, frame: int) -> torch.Tensor:
        path = self.root / "image_02" / f"{seq:04d}" / f"{frame:06d}.png"
        img = cv2.imread(str(path))
        if img is None:
            raise FileNotFoundError(f"Image not found: {path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        if self.img_size is not None:
            img = cv2.resize(img, self.img_size, interpolation=cv2.INTER_LINEAR)

        # HWC uint8 → CHW float32 [0, 1]
        img = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
        return img

    def _load_velodyne(self, seq: int, frame: int) -> np.ndarray:
        path = self.root / "velodyne" / f"{seq:04d}" / f"{frame:06d}.bin"
        if not path.exists():
            return np.zeros((0, 4), dtype=np.float32)
        # Each point: [x, y, z, intensity]  — stored as binary float32
        points = np.fromfile(str(path), dtype=np.float32).reshape(-1, 4)

        # Remove points behind the sensor (x < 0) — common preprocessing step
        points = points[points[:, 0] > 0]
        return points

    def _load_labels(self, seq: int, frame: int) -> list[dict]:
        if not self.has_labels:
            return []
        frame_dict = self._label_cache.get(seq, {})
        objects = frame_dict.get(frame, [])
        # Filter by class and exclude DontCare
        return [
            obj for obj in objects
            if obj["type"] in self.classes and obj["type"] != "DontCare"
        ]

    def _load_calib(self, seq: int) -> KITTICalib:
        calib_path = self.root / "calib" / f"{seq:04d}.txt"
        return load_calib(str(calib_path))

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def project_lidar_to_image(
        self,
        points: np.ndarray,
        calib: KITTICalib,
        img_shape: tuple[int, int],
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Project LiDAR points onto the camera image plane.

        Returns
        -------
        pts_img : (M, 2) pixel coordinates of points that fall inside the image
        depth   : (M,)  depth values (metres) for those points
        """
        H, W = img_shape

        # Velodyne → camera (homogeneous)
        pts_hom = np.hstack([points[:, :3], np.ones((len(points), 1), dtype=np.float32)])
        Tr = np.vstack([calib.Tr_velo_to_cam, [0, 0, 0, 1]])   # 4×4
        R0 = np.eye(4, dtype=np.float32)
        R0[:3, :3] = calib.R0_rect
        P2 = calib.P2                                            # 3×4

        cam_pts = (R0 @ Tr @ pts_hom.T)[:3]  # 3×N
        # Keep only points in front of camera
        mask_front = cam_pts[2] > 0
        cam_pts = cam_pts[:, mask_front]
        depths = cam_pts[2]

        # Project
        proj = P2 @ np.vstack([cam_pts, np.ones((1, cam_pts.shape[1]))])
        proj[:2] /= proj[2]
        u, v = proj[0], proj[1]

        mask_img = (u >= 0) & (u < W) & (v >= 0) & (v < H)
        return np.stack([u[mask_img], v[mask_img]], axis=1), depths[mask_img]

    @staticmethod
    def collate_fn(batch: list[dict]) -> dict:
        """
        Custom collate: images are stacked as tensors;
        points/labels/calib/meta stay as lists (variable length per frame).
        """
        return {
            "image":  torch.stack([b["image"] for b in batch]),
            "points": [b["points"] for b in batch],
            "labels": [b["labels"] for b in batch],
            "calib":  [b["calib"]  for b in batch],
            "meta":   [b["meta"]   for b in batch],
        }


# ---------------------------------------------------------------------------
# Quick sanity check
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    import matplotlib.pyplot as plt

    root = sys.argv[1] if len(sys.argv) > 1 else os.path.expanduser("~/MAVERIC/data/kitti/tracking")

    ds = KITTITrackingDataset(
        root=root,
        split="training",
        sequences=[0],                        # just sequence 0 for the check
        classes=["Car", "Pedestrian", "Cyclist"],
        img_size=(1242, 375),
    )

    sample = ds[0]
    img     = sample["image"].permute(1, 2, 0).numpy()
    points  = sample["points"]
    labels  = sample["labels"]
    calib   = sample["calib"]
    meta    = sample["meta"]

    print(f"\nSeq {meta['seq']} | Frame {meta['frame']}")
    print(f"  Image  : {sample['image'].shape}  dtype={sample['image'].dtype}")
    print(f"  LiDAR  : {points.shape} points")
    print(f"  Labels : {len(labels)} objects")
    for obj in labels[:3]:
        print(f"    [{obj['track_id']:3d}] {obj['type']:<12s}  "
              f"bbox={obj['bbox_2d'].astype(int)}  "
              f"loc=({obj['location'][0]:.1f}, {obj['location'][1]:.1f}, {obj['location'][2]:.1f})m")

    # Draw 2D bboxes on image
    img_vis = (img * 255).astype(np.uint8).copy()
    img_vis = cv2.cvtColor(img_vis, cv2.COLOR_RGB2BGR)
    for obj in labels:
        l, t, r, b = obj["bbox_2d"].astype(int)
        color = {"Car": (0, 255, 0), "Pedestrian": (255, 0, 0), "Cyclist": (0, 165, 255)}.get(obj["type"], (200, 200, 200))
        cv2.rectangle(img_vis, (l, t), (r, b), color, 2)
        cv2.putText(img_vis, f"{obj['type']}#{obj['track_id']}", (l, t - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)

    out_path = "kitti_sanity_check.png"
    cv2.imwrite(out_path, img_vis)
    print(f"\n  Sanity check image saved → {out_path}")
    print("  Green=Car  Blue=Pedestrian  Orange=Cyclist")