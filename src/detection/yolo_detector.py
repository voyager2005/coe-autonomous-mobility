"""
YOLODetector — YOLO11m-based 2-D object detector for the MAVERIC pipeline.

Loads model weights and detection thresholds from configs/pipeline.yaml.
NMS runs on GPU via Ultralytics built-in (no manual post-processing).
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from ultralytics import YOLO


class YOLODetector:
    """Detect objects in BGR frames using YOLO11m.

    Parameters
    ----------
    config_path : str | Path
        Path to the pipeline YAML config (default: ``configs/pipeline.yaml``
        resolved relative to the project root).

    Example
    -------
    >>> det = YOLODetector()
    >>> detections = det.detect(frame_bgr)
    [{"bbox": [100, 200, 300, 400], "conf": 0.87, "class": "car"}, ...]
    """

    def __init__(self, config_path: str | Path = "configs/pipeline.yaml") -> None:
        # ------------------------------------------------------------------
        # 1. Load pipeline config
        # ------------------------------------------------------------------
        config_path = Path(config_path).expanduser()
        if not config_path.is_absolute():
            # Resolve relative paths from the project root (two levels up
            # from this file: src/detection/yolo_detector.py → MAVERIC/)
            project_root = Path(__file__).resolve().parents[2]
            config_path = project_root / config_path

        with open(config_path, "r") as f:
            cfg: dict[str, Any] = yaml.safe_load(f)

        cam_cfg: dict[str, Any] = cfg["detection"]["camera"]

        # ------------------------------------------------------------------
        # 2. Read detection thresholds & device
        # ------------------------------------------------------------------
        self.conf_thresh: float = float(cam_cfg.get("conf_thresh", 0.25))
        self.iou_thresh: float = float(cam_cfg.get("iou_thresh", 0.45))
        self.device: str = cam_cfg.get("device", "cuda:0")

        # ------------------------------------------------------------------
        # 3. Resolve weights path & load model
        # ------------------------------------------------------------------
        weights_path = Path(cam_cfg["weights"]).expanduser()
        if not weights_path.is_absolute():
            project_root = Path(__file__).resolve().parents[2]
            weights_path = project_root / weights_path

        if not weights_path.exists():
            raise FileNotFoundError(
                f"YOLO weights not found at {weights_path}. "
                "Download yolo11m.pt or update detection.camera.weights in "
                "configs/pipeline.yaml."
            )

        use_trt: bool = cam_cfg.get("use_tensorrt", False)
        engine_path = weights_path.with_suffix(".engine")
        if use_trt and engine_path.exists():
            self.model = YOLO(str(engine_path))
        else:
            self.model = YOLO(str(weights_path))

        # Cache class-name mapping from the model
        self._names: dict[int, str] = self.model.names  # {0: 'person', …}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def detect(self, frame: np.ndarray) -> list[dict[str, Any]]:
        """Run detection on a single BGR frame.

        Parameters
        ----------
        frame : np.ndarray
            Image of shape ``(H, W, 3)`` in **BGR** colour order
            (standard OpenCV convention).

        Returns
        -------
        list[dict]
            Each dict contains:
            - ``bbox``  : ``[x1, y1, x2, y2]`` pixel coordinates (float).
            - ``conf``  : detection confidence (float, 0–1).
            - ``class`` : human-readable class name (str).
        """
        results = self.model.predict(
            source=frame,
            conf=self.conf_thresh,
            iou=self.iou_thresh,
            device=self.device,
            verbose=False,
        )

        detections: list[dict[str, Any]] = []
        for result in results:
            boxes = result.boxes
            if boxes is None or len(boxes) == 0:
                continue
            # Move to CPU once for the whole batch of boxes
            xyxy = boxes.xyxy.cpu().numpy()
            confs = boxes.conf.cpu().numpy()
            cls_ids = boxes.cls.cpu().numpy().astype(int)

            for bbox, conf, cls_id in zip(xyxy, confs, cls_ids):
                class_name = self._names.get(cls_id, str(cls_id)).lower()
                detections.append(
                    {
                        "bbox": bbox.tolist(),       # [x1, y1, x2, y2]
                        "conf": float(conf),
                        "class": class_name,
                    }
                )

        return detections

    def __repr__(self) -> str:
        return (
            f"YOLODetector(conf={self.conf_thresh}, "
            f"iou={self.iou_thresh}, device={self.device})"
        )


# ----------------------------------------------------------------------
# Quick demo — run with:  python src/detection/yolo_detector.py
# ----------------------------------------------------------------------
if __name__ == "__main__":
    import cv2

    det = YOLODetector()
    print(det)
    print()

    # Output directory for annotated images
    out_dir = Path(__file__).resolve().parents[2] / "outputs" / "detection_demo"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Use a few frames from KITTI sequence 0000
    image_dir = Path(__file__).resolve().parents[2] / "data/kitti/tracking/training/image_02/0000"
    test_frames = ["000000.png", "000050.png", "000100.png"]

    # Colour palette for classes (cycle if more classes appear)
    COLORS = [
        (0, 255, 0), (0, 165, 255), (0, 0, 255), (255, 255, 0),
        (255, 0, 255), (255, 128, 0), (128, 255, 0), (0, 255, 255),
    ]
    _class_colors: dict[str, tuple[int, ...]] = {}

    for fname in test_frames:
        path = image_dir / fname
        frame = cv2.imread(str(path))
        if frame is None:
            print(f"SKIP  {fname} (not found)")
            continue

        results = det.detect(frame)
        print(f"--- {fname} ({frame.shape[1]}x{frame.shape[0]}) ---")
        print(f"  Total detections: {len(results)}")

        # Draw bounding boxes and labels
        vis = frame.copy()
        for r in results:
            x1, y1, x2, y2 = [int(v) for v in r["bbox"]]
            label = f"{r['class']} {r['conf']:.2f}"

            # Assign a consistent colour per class
            if r["class"] not in _class_colors:
                _class_colors[r["class"]] = COLORS[len(_class_colors) % len(COLORS)]
            color = _class_colors[r["class"]]

            cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)

            # Label background
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(vis, (x1, y1 - th - 6), (x1 + tw + 4, y1), color, -1)
            cv2.putText(vis, label, (x1 + 2, y1 - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

            bbox = [round(v, 1) for v in r["bbox"]]
            print(f"    {r['class']:15s}  conf={r['conf']:.3f}  bbox={bbox}")

        # Save annotated image
        save_path = out_dir / f"det_{fname}"
        cv2.imwrite(str(save_path), vis)
        print(f"  Saved → {save_path}")
        print()

        # Show image (press any key to advance, 'q' to quit)
        cv2.imshow(f"YOLODetector — {fname}", vis)
        key = cv2.waitKey(0) & 0xFF
        if key == ord("q"):
            break

    cv2.destroyAllWindows()
    print("Done.")