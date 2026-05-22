"""
run_pipeline.py — MAVERIC end-to-end MOT pipeline runner.

For each frame in the configured KITTI sequences:
  1. Load image + point cloud + calibration via KITTITrackingDataset
  2. Detect 2-D boxes  → YOLODetector
  3. Detect 3-D boxes  → PointPillarsDetector  (optional, skip if weights missing)
  4. Embed crops       → OSNetEmbedder
  5. Track             → MAVERICTracker  (persistent IDs across frames)
  6. Render            → renderer.render_frame
  7. Save frame to     outputs/visualisations/seq_XXXX/
  8. Write track log   outputs/tracks/seq_XXXX.json

Run with:
    python src/run_pipeline.py
    python src/run_pipeline.py --config configs/pipeline.yaml --sequences 0 1 2
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import yaml
from loguru import logger

# ── Path setup ───────────────────────────────────────────────────────────────
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
for _p in [str(_PROJECT_ROOT), str(_PROJECT_ROOT / "src")]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from kitti_dataset import KITTITrackingDataset
from detection.yolo_detector import YOLODetector
from reid.embedder import OSNetEmbedder
from tracking.kalman import KalmanBoxTracker
from tracking.track import Track
from tracking.tracker import MAVERICTracker
from visualisation.renderer import render_frame


# ── Track dataclass extension ─────────────────────────────────────────────────
# The base Track dataclass doesn't store class_name / score / location.
# We extend it here so the renderer can access those fields.

@dataclass
class RichTrack(Track):
    """Track with extra metadata for the renderer."""
    class_name: str  = "Object"
    score:      float = 0.0
    location:   Any  = None   # [x, y, z] in camera coords from PointPillars
    bbox_3d:    Any  = None   # [x, y, z, w, h, l, yaw]


# ── Helpers ───────────────────────────────────────────────────────────────────

def _load_config(path: str | Path) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def _detections_to_arrays(
    detections: list[dict],
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Split YOLO detection dicts into parallel arrays."""
    if not detections:
        return np.empty((0, 4), np.float32), np.empty(0, np.float32), []
    bboxes  = np.array([d["bbox"]  for d in detections], dtype=np.float32)
    scores  = np.array([d["conf"]  for d in detections], dtype=np.float32)
    classes = [d["class"] for d in detections]
    return bboxes, scores, classes


def _match_3d_to_2d(
    det2d: list[dict],
    det3d: list[dict],
    calib: Any,
    img_shape: tuple[int, int],
) -> list[Any]:
    """
    For each 2-D detection, find the best matching 3-D detection by IoU
    of the projected 3-D box vs the 2-D box. Returns a list parallel to
    det2d where each entry is either a 3-D dict or None.
    """
    if not det3d:
        return [None] * len(det2d)

    try:
        from fusion.projector import project_3d_to_2d
    except ImportError:
        return [None] * len(det2d)

    matched = [None] * len(det2d)
    H, W = img_shape

    for j, d3 in enumerate(det3d):
        try:
            _, bbox2d_proj = project_3d_to_2d(d3["bbox_3d"], calib, (H, W))
        except Exception:
            continue

        best_iou, best_i = 0.0, -1
        x1p, y1p, x2p, y2p = bbox2d_proj
        for i, d2 in enumerate(det2d):
            x1, y1, x2, y2 = d2["bbox"]
            ix1 = max(x1, x1p); iy1 = max(y1, y1p)
            ix2 = min(x2, x2p); iy2 = min(y2, y2p)
            iw = max(0, ix2 - ix1); ih = max(0, iy2 - iy1)
            inter = iw * ih
            union = (x2-x1)*(y2-y1) + (x2p-x1p)*(y2p-y1p) - inter + 1e-6
            iou = inter / union
            if iou > best_iou:
                best_iou, best_i = iou, i
        if best_i >= 0 and best_iou > 0.3:
            matched[best_i] = d3

    return matched


def _build_rich_track(
    track_id: int,
    bbox: np.ndarray,
    embedding: np.ndarray,
    class_name: str,
    score: float,
    d3: dict | None = None,
) -> RichTrack:
    """Construct a RichTrack from raw detection data."""
    kalman = KalmanBoxTracker(bbox)
    location = d3["bbox_3d"][:3] if d3 is not None else None
    bbox_3d  = d3["bbox_3d"]     if d3 is not None else None

    return RichTrack(
        id=track_id,
        kalman=kalman,
        embedding=embedding,
        class_name=class_name,
        score=score,
        location=location,
        bbox_3d=bbox_3d,
    )


# ── Per-sequence runner ───────────────────────────────────────────────────────

def run_sequence(
    seq_id: int,
    dataset: KITTITrackingDataset,
    yolo: YOLODetector,
    embedder: OSNetEmbedder,
    lidar_detector: Any | None,
    cfg: dict,
    vis_dir: Path,
    track_log_dir: Path,
    save_video: bool = True,
    show_lidar: bool = True,
) -> dict:
    """Run the full pipeline on a single KITTI tracking sequence."""

    tracker = MAVERICTracker(
        iou_weight=cfg["tracking"].get("iou_weight", 0.5),
        reid_weight=cfg["tracking"].get("reid_weight", 0.5),
        max_age=cfg["tracking"].get("max_age", 30),
    )

    # Rich track registry: tracker returns base Track objects,
    # we shadow them with RichTrack for renderer metadata
    rich_registry: dict[int, RichTrack] = {}

    # Frame indices for this sequence
    seq_indices = [i for i, (s, _) in enumerate(dataset.index) if s == seq_id]

    seq_vis_dir = vis_dir / f"seq_{seq_id:04d}"
    seq_vis_dir.mkdir(parents=True, exist_ok=True)

    track_log: list[dict] = []

    fps_times: list[float] = []
    video_writer = None

    logger.info(f"Sequence {seq_id:04d} — {len(seq_indices)} frames")

    for frame_num, ds_idx in enumerate(seq_indices):
        t_frame_start = time.perf_counter()

        sample   = dataset[ds_idx]
        frame_id = sample["meta"]["frame"]
        calib    = sample["calib"]
        points   = sample["points"]

        # Convert CHW float tensor → HWC BGR uint8
        img_rgb = (sample["image"].permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        frame_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
        H, W = frame_bgr.shape[:2]

        # ── 2-D Detection ──────────────────────────────────────────────
        det2d_raw = yolo.detect(frame_bgr)
        # Filter to AV-relevant classes only
        allowed = set(c.lower() for c in cfg["data"].get(
            "classes", ["car", "van", "truck", "pedestrian", "cyclist", "person"]
        ))
        coco_to_kitti = {
            "car": "Car", "truck": "Truck", "van": "Van",
            "person": "Pedestrian", "bicycle": "Cyclist", "motorcycle": "Cyclist",
        }
        coco_exclude = {"bus", "train", "traffic light", "stop sign",
                        "parking meter", "bench", "cat", "dog",
                        "handbag", "backpack", "umbrella", "suitcase"}
        det2d = []
        for d in det2d_raw:
            cn = d["class"].lower()
            if cn in coco_exclude:
                continue
            if cn in coco_to_kitti:
                d = dict(d)
                d["class"] = coco_to_kitti[cn]
                det2d.append(d)
        bboxes, scores, classes = _detections_to_arrays(det2d)

        # ── 3-D Detection (optional) ───────────────────────────────────
        det3d: list[dict] = []
        if lidar_detector is not None and len(points) > 0:
            try:
                pts = points[points[:, 0] > 0]
                det3d = lidar_detector.detect(pts)
            except Exception as e:
                logger.warning(f"  LiDAR detection failed frame {frame_id}: {e}")

        matched_3d = _match_3d_to_2d(det2d, det3d, calib, (H, W))

        # ── ReID Embedding ─────────────────────────────────────────────
        if len(bboxes) > 0:
            embeddings = embedder.embed(frame_bgr, bboxes.tolist())
        else:
            embeddings = np.empty((0, 512), dtype=np.float32)

        # ── Tracking ───────────────────────────────────────────────────
        active_base_tracks = tracker.update(bboxes, scores, embeddings)

        # Sync rich_registry with tracker state
        # For new tracks (no entry in registry), create a RichTrack
        active_ids = set()
        for bt in active_base_tracks:
            active_ids.add(bt.id)
            if bt.id not in rich_registry:
                # Find the detection that spawned this track
                bbox_est = bt.kalman.get_bbox()
                best_i, best_iou = 0, 0.0
                for i, bb in enumerate(bboxes):
                    x1,y1,x2,y2 = bb
                    bx1,by1,bx2,by2 = bbox_est
                    ix1=max(x1,bx1); iy1=max(y1,by1)
                    ix2=min(x2,bx2); iy2=min(y2,by2)
                    iw=max(0,ix2-ix1); ih=max(0,iy2-iy1)
                    inter=iw*ih
                    union=(x2-x1)*(y2-y1)+(bx2-bx1)*(by2-by1)-inter+1e-6
                    iou=inter/union
                    if iou > best_iou:
                        best_iou, best_i = iou, i

                cls  = classes[best_i] if len(classes) > best_i else "object"
                scr  = float(scores[best_i]) if len(scores) > best_i else 0.0
                d3   = matched_3d[best_i] if best_i < len(matched_3d) else None
                emb  = embeddings[best_i] if best_i < len(embeddings) else np.zeros(512)

                rich_registry[bt.id] = RichTrack(
                    id=bt.id,
                    kalman=bt.kalman,
                    embedding=bt.embedding,
                    state=bt.state,
                    hits=bt.hits,
                    age=bt.age,
                    time_since_update=bt.time_since_update,
                    class_name=cls.capitalize(),
                    score=scr,
                    location=d3["bbox_3d"][:3] if d3 else None,
                    bbox_3d=d3["bbox_3d"]      if d3 else None,
                )
            else:
                # Sync state fields from base track
                rt = rich_registry[bt.id]
                rt.kalman           = bt.kalman
                rt.embedding        = bt.embedding
                rt.state            = bt.state
                rt.hits             = bt.hits
                rt.age              = bt.age
                rt.time_since_update = bt.time_since_update
                # Score is kept from initial assignment — don't overwrite with scores[0]
                pass

        # Prune dead tracks from registry
        for tid in list(rich_registry.keys()):
            if tid not in active_ids:
                del rich_registry[tid]

        render_tracks = list(rich_registry.values())

        # ── Render ─────────────────────────────────────────────────────
        if frame_num == 0:
            logger.info(f"  LiDAR debug: points={len(points)}, show_lidar={show_lidar}, "
                       f"points_type={type(points)}, points_shape={points.shape if hasattr(points, 'shape') else 'N/A'}")
        vis = render_frame(
            frame=frame_bgr,
            tracks=render_tracks,
            calib=calib,
            points=points if show_lidar else None,
            show_lidar=show_lidar,
            show_bev=True,
            show_3d_boxes=(lidar_detector is not None),
        )

        # ── Save frame ─────────────────────────────────────────────────
        out_path = seq_vis_dir / f"frame_{frame_id:06d}.png"
        cv2.imwrite(str(out_path), vis)

        # ── Video writer ───────────────────────────────────────────────
        if save_video and video_writer is None:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            video_path = vis_dir / f"seq_{seq_id:04d}.mp4"
            video_writer = cv2.VideoWriter(str(video_path), fourcc, 10, (W, H))
        if video_writer is not None:
            video_writer.write(vis)

        # ── Track log ──────────────────────────────────────────────────
        track_log.append({
            "frame": frame_id,
            "tracks": [
                {
                    "id":         t.id,
                    "class":      t.class_name,
                    "conf":       round(t.score, 3),
                    "bbox":       [round(v, 1) for v in t.kalman.get_bbox().tolist()],
                    "state":      t.state,
                }
                for t in render_tracks
            ],
        })

        dt = (time.perf_counter() - t_frame_start) * 1000
        fps_times.append(dt)

        if frame_num % 50 == 0 or frame_num == 0:
            avg_fps = 1000.0 / (sum(fps_times[-50:]) / len(fps_times[-50:]))
            logger.info(
                f"  [{frame_id:04d}] {len(render_tracks):2d} tracks | "
                f"{dt:5.1f}ms | {avg_fps:.1f} FPS"
            )

    if video_writer is not None:
        video_writer.release()

    # Save track log
    log_path = track_log_dir / f"seq_{seq_id:04d}.json"
    with open(log_path, "w") as f:
        json.dump(track_log, f, indent=2)
    logger.info(f"  Track log → {log_path}")

    avg_ms  = sum(fps_times) / len(fps_times) if fps_times else 0
    avg_fps = 1000.0 / avg_ms if avg_ms > 0 else 0
    return {"seq": seq_id, "frames": len(seq_indices), "avg_ms": avg_ms, "avg_fps": avg_fps}


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="MAVERIC MOT Pipeline")
    parser.add_argument("--config",    default="configs/pipeline.yaml")
    parser.add_argument("--sequences", nargs="+", type=int, default=None,
                        help="Sequence IDs to run (default: from config)")
    parser.add_argument("--no-lidar",  action="store_true",
                        help="Skip LiDAR detection (camera-only mode)")
    parser.add_argument("--no-video",  action="store_true",
                        help="Skip MP4 video output")
    args = parser.parse_args()

    cfg_path = Path(args.config).expanduser()
    if not cfg_path.is_absolute():
        cfg_path = _PROJECT_ROOT / cfg_path
    cfg = _load_config(cfg_path)

    # ── Output dirs ───────────────────────────────────────────────────
    vis_dir       = Path(cfg["output"]["vis_dir"]).expanduser()
    track_log_dir = Path(cfg["output"]["tracks_dir"]).expanduser()
    vis_dir.mkdir(parents=True, exist_ok=True)
    track_log_dir.mkdir(parents=True, exist_ok=True)

    # ── Dataset ───────────────────────────────────────────────────────
    sequences = args.sequences or cfg["data"].get("sequences") or [0, 1, 2]
    dataset = KITTITrackingDataset(
        root=str(Path(cfg["data"]["kitti_root"]).expanduser() / "tracking"),
        split=cfg["data"].get("split", "training"),
        sequences=sequences,
        classes=cfg["data"].get("classes", ["Car", "Van", "Truck", "Pedestrian", "Cyclist"]),
        img_size=tuple(cfg["data"].get("img_size", [1242, 375])) if cfg["data"].get("img_size") else None,
    )

    # ── Models ────────────────────────────────────────────────────────
    logger.info("Loading YOLODetector...")
    yolo = YOLODetector(cfg_path)

    logger.info("Loading OSNetEmbedder...")
    embedder = OSNetEmbedder(cfg_path)

    lidar_detector = None
    if not args.no_lidar:
        try:
            from detection.pointpillars_detector import PointPillarsDetector
            logger.info("Loading PointPillarsDetector...")
            lidar_detector = PointPillarsDetector(cfg_path)
        except Exception as e:
            logger.warning(f"PointPillars unavailable ({e}), running camera-only.")

    # ── Run sequences ─────────────────────────────────────────────────
    results = []
    for seq_id in sequences:
        logger.info(f"\n{'='*60}")
        logger.info(f"Sequence {seq_id:04d}")
        logger.info(f"{'='*60}")
        result = run_sequence(
            seq_id=seq_id,
            dataset=dataset,
            yolo=yolo,
            embedder=embedder,
            lidar_detector=lidar_detector,
            cfg=cfg,
            vis_dir=vis_dir,
            track_log_dir=track_log_dir,
            save_video=not args.no_video,
            show_lidar=cfg["output"].get("show_lidar_overlay", True),
        )
        results.append(result)

    # ── Summary ───────────────────────────────────────────────────────
    logger.info(f"\n{'='*60}")
    logger.info("PIPELINE SUMMARY")
    logger.info(f"{'='*60}")
    for r in results:
        logger.info(
            f"  Seq {r['seq']:04d} | {r['frames']:4d} frames | "
            f"{r['avg_ms']:.1f}ms/frame | {r['avg_fps']:.1f} FPS"
        )
    logger.info(f"\nVisuals → {vis_dir}")
    logger.info(f"Logs    → {track_log_dir}")


if __name__ == "__main__":
    main()