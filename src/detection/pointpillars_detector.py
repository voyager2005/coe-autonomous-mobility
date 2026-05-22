"""
PointPillarsDetector — LiDAR-based 3-D object detector for the MAVERIC pipeline.

Wraps OpenPCDet's PointPillars model using its own DemoDataset infrastructure
to avoid brittle stub implementations. Takes a raw point cloud (N, 4) and
returns a list of 3-D oriented bounding boxes with class labels and scores.

Loads all config from configs/pipeline.yaml.
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml


class PointPillarsDetector:
    """Detect objects in LiDAR point clouds using PointPillars (OpenPCDet).

    Parameters
    ----------
    config_path : str | Path
        Path to pipeline YAML config. Defaults to ``configs/pipeline.yaml``
        resolved from the project root.

    Example
    -------
    >>> det = PointPillarsDetector()
    >>> detections = det.detect(points)  # points: (N, 4) float32
    [{"bbox_3d": [x, y, z, w, h, l, yaw], "conf": 0.82, "class": "Car"}, ...]
    """

    def __init__(self, config_path: str | Path = "configs/pipeline.yaml") -> None:
        # ------------------------------------------------------------------
        # 1. Load pipeline config
        # ------------------------------------------------------------------
        config_path = Path(config_path).expanduser()
        if not config_path.is_absolute():
            project_root = Path(__file__).resolve().parents[2]
            config_path = project_root / config_path

        with open(config_path, "r") as f:
            cfg: dict[str, Any] = yaml.safe_load(f)

        lid_cfg: dict[str, Any] = cfg["detection"]["lidar"]

        self.score_thresh: float = float(lid_cfg.get("score_thresh", 0.3))
        self.device: str        = lid_cfg.get("device", "cuda:0")

        # ------------------------------------------------------------------
        # 2. Resolve paths
        # ------------------------------------------------------------------
        project_root    = Path(__file__).resolve().parents[2]
        openpcdet_root  = project_root / "OpenPCDet"

        if not openpcdet_root.exists():
            raise FileNotFoundError(
                f"OpenPCDet not found at {openpcdet_root}.\n"
                "Clone it with: git clone https://github.com/open-mmlab/OpenPCDet"
            )

        weights_path = Path(lid_cfg["weights"]).expanduser()
        if not weights_path.is_absolute():
            weights_path = project_root / weights_path

        if not weights_path.exists():
            raise FileNotFoundError(
                f"PointPillars weights not found at {weights_path}.\n"
                "Download a pretrained KITTI checkpoint from OpenPCDet model zoo\n"
                "and update detection.lidar.weights in configs/pipeline.yaml."
            )

        pcdet_cfg_path = Path(lid_cfg.get(
            "pcdet_config",
            "OpenPCDet/tools/cfgs/kitti_models/pointpillar.yaml"
        )).expanduser()
        if not pcdet_cfg_path.is_absolute():
            pcdet_cfg_path = project_root / pcdet_cfg_path

        if not pcdet_cfg_path.exists():
            raise FileNotFoundError(
                f"OpenPCDet config not found at {pcdet_cfg_path}."
            )

        # ------------------------------------------------------------------
        # 3. Add OpenPCDet to path and chdir to tools/ (required by pcdet)
        # ------------------------------------------------------------------
        if str(openpcdet_root) not in sys.path:
            sys.path.insert(0, str(openpcdet_root))
            
        tools_path = str(openpcdet_root / "tools")
        if tools_path not in sys.path:
            sys.path.insert(0, tools_path)

        os.chdir(tools_path)

        # ------------------------------------------------------------------
        # 4. Load model using OpenPCDet's own DemoDataset infrastructure
        # ------------------------------------------------------------------
        self.model, self.class_names = self._build_model(
            pcdet_cfg_path, weights_path
        )

    # ------------------------------------------------------------------
    # Model building — uses OpenPCDet's DemoDataset to avoid stub issues
    # ------------------------------------------------------------------
    def _build_model(self, cfg_path: Path, weights_path: Path):
        from pcdet.config import cfg as pcdet_cfg, cfg_from_yaml_file
        from pcdet.models import build_network, load_data_to_gpu
        from pcdet.utils import common_utils

        # cfg_from_yaml_file accepts absolute paths directly
        cfg_from_yaml_file(str(cfg_path), pcdet_cfg)

        logger = common_utils.create_logger()

        # Use DemoDataset — OpenPCDet's official lightweight inference dataset
        from pcdet.datasets import build_dataloader
        from tools.demo import DemoDataset  # noqa: F401 — import path check

        demo_dataset = DemoDataset(
            dataset_cfg=pcdet_cfg.DATA_CONFIG,
            class_names=pcdet_cfg.CLASS_NAMES,
            training=False,
            root_path=Path("."),   # placeholder — we feed points manually
            ext=".bin",
            logger=logger,
        )

        model = build_network(
            model_cfg=pcdet_cfg.MODEL,
            num_class=len(pcdet_cfg.CLASS_NAMES),
            dataset=demo_dataset,
        )

        model.load_params_from_file(
            filename=str(weights_path),
            logger=logger,
            to_cpu=False,
        )
        model.cuda()
        model.eval()

        return model, pcdet_cfg.CLASS_NAMES

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def detect(self, points: np.ndarray) -> list[dict[str, Any]]:
        """Run 3-D object detection on a LiDAR point cloud.

        Parameters
        ----------
        points : np.ndarray
            Shape ``(N, 4)`` — ``[x, y, z, intensity]`` in velodyne frame.
            Points behind the sensor (x ≤ 0) should already be removed.

        Returns
        -------
        list[dict]
            Each dict contains:
            - ``bbox_3d`` : ``[x, y, z, w, h, l, yaw]`` in camera coordinates.
            - ``conf``    : detection score (float, 0–1).
            - ``class``   : class name string (e.g. ``"Car"``).
        """
        if len(points) == 0:
            return []

        from pcdet.models import load_data_to_gpu
        from pcdet.utils import common_utils

        # Build input dict in OpenPCDet format
        data_dict = {
            "points": points.astype(np.float32),
            "frame_id": 0,
            "calib": None,
        }

        # Use the demo dataset's preprocessing pipeline
        demo_dataset = self._get_demo_dataset()
        data_dict = demo_dataset.prepare_data(data_dict=data_dict)
        data_dict = demo_dataset.collate_batch([data_dict])
        load_data_to_gpu(data_dict)

        with torch.no_grad():
            pred_dicts, _ = self.model.forward(data_dict)

        return self._parse_predictions(pred_dicts)

    def _get_demo_dataset(self):
        """Return cached demo dataset instance for preprocessing."""
        if not hasattr(self, "_demo_dataset"):
            from pcdet.config import cfg as pcdet_cfg
            from pcdet.utils import common_utils
            from tools.demo import DemoDataset

            logger = common_utils.create_logger()
            self._demo_dataset = DemoDataset(
                dataset_cfg=pcdet_cfg.DATA_CONFIG,
                class_names=pcdet_cfg.CLASS_NAMES,
                training=False,
                root_path=Path("."),
                ext=".bin",
                logger=logger,
            )
        return self._demo_dataset

    def _parse_predictions(
        self, pred_dicts: list[dict]
    ) -> list[dict[str, Any]]:
        """Convert OpenPCDet output to MAVERIC detection format."""
        detections: list[dict[str, Any]] = []

        for pred in pred_dicts:
            boxes  = pred["pred_boxes"].cpu().numpy()    # (M, 7)
            scores = pred["pred_scores"].cpu().numpy()   # (M,)
            labels = pred["pred_labels"].cpu().numpy()   # (M,) 1-indexed

            for box, score, label in zip(boxes, scores, labels):
                if score < self.score_thresh:
                    continue
                detections.append({
                    "bbox_3d": box.tolist(),
                    "conf":    float(score),
                    "class":   self.class_names[int(label) - 1],
                })

        return detections

    def __repr__(self) -> str:
        return (
            f"PointPillarsDetector("
            f"score_thresh={self.score_thresh}, "
            f"classes={self.class_names}, "
            f"device={self.device})"
        )


# ----------------------------------------------------------------------
# Quick demo — run with:
#   python src/detection/pointpillars_detector.py
# ----------------------------------------------------------------------
if __name__ == "__main__":
    det = PointPillarsDetector()
    print(det)
    print()

    project_root = Path(__file__).resolve().parents[2]
    lidar_dir    = project_root / "data/kitti/tracking/training/velodyne/0000"
    test_frames  = ["000000.bin", "000010.bin", "000020.bin"]

    for fname in test_frames:
        path = lidar_dir / fname
        if not path.exists():
            print(f"SKIP  {fname} (not found)")
            continue

        points = np.fromfile(str(path), dtype=np.float32).reshape(-1, 4)
        points = points[points[:, 0] > 0]

        t0 = time.perf_counter()
        results = det.detect(points)
        dt = (time.perf_counter() - t0) * 1000

        print(f"--- {fname} ---")
        print(f"  Input points : {len(points):,}")
        print(f"  Detections   : {len(results)}  ({dt:.1f} ms)")
        for r in results:
            box = [round(v, 2) for v in r["bbox_3d"]]
            print(f"    {r['class']:12s}  conf={r['conf']:.3f}  box={box}")
        print()

    print("Done.")