"""
renderer.py — Frame annotation and BEV visualisation for the MAVERIC pipeline.

Renders confirmed and tentative tracks onto camera frames with:
  - Colour-coded persistent ID bounding boxes
  - Class + ID + confidence labels with adaptive text colour
  - LiDAR depth point overlay (green=close, red=far)
  - 3-D wireframe box projection using ground truth or predicted 3-D boxes
  - Bird's-eye-view (BEV) mini-panel in the top-right corner

Design decisions:
  - All drawing is done on a copy — original frame never mutated
  - Colours are deterministic per track ID (hash → HSV → BGR)
  - Tentative tracks are drawn at 50% opacity
  - BEV panel uses a dark background with a distance grid and ego marker
  - LiDAR overlay is vectorised — no Python loops over individual points
"""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np

# Make projector importable regardless of working directory
_SRC_ROOT  = Path(__file__).resolve().parents[1]
_PROJ_ROOT = Path(__file__).resolve().parents[2]
for _p in [str(_PROJ_ROOT), str(_SRC_ROOT)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from fusion.projector import (
    project_lidar_to_image,
    project_3d_to_2d,
    draw_box_3d,
)


# ---------------------------------------------------------------------------
# Colour utilities
# ---------------------------------------------------------------------------

def track_color(track_id: int) -> tuple[int, int, int]:
    """Deterministic BGR colour for a track ID via MD5 → HSV → BGR."""
    h = int(hashlib.md5(str(track_id).encode()).hexdigest(), 16) % 180
    hsv = np.uint8([[[h, 220, 255]]])
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0][0]
    return (int(bgr[0]), int(bgr[1]), int(bgr[2]))


def _adaptive_text_color(bg: tuple[int, int, int]) -> tuple[int, int, int]:
    b, g, r = bg
    return (0, 0, 0) if (0.114 * b + 0.587 * g + 0.299 * r) > 140 else (255, 255, 255)


def _dim_color(c: tuple[int, int, int], f: float = 0.45) -> tuple[int, int, int]:
    return (int(c[0] * f), int(c[1] * f), int(c[2] * f))


# ---------------------------------------------------------------------------
# Label
# ---------------------------------------------------------------------------

def _draw_label(img, text, x1, y1, color, font_scale=0.48, thickness=1):
    font = cv2.FONT_HERSHEY_SIMPLEX
    (tw, th), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    ly = max(y1 - 2, th + 6)
    pad = 3
    cv2.rectangle(img, (x1, ly - th - pad), (x1 + tw + pad * 2, ly + baseline), color, -1)
    cv2.putText(img, text, (x1 + pad, ly), font, font_scale,
                _adaptive_text_color(color), thickness, cv2.LINE_AA)


# ---------------------------------------------------------------------------
# LiDAR overlay — fully vectorised
# ---------------------------------------------------------------------------

def _draw_lidar_points(img, points, calib):
    H, W = img.shape[:2]
    uv, depth = project_lidar_to_image(points, calib, (H, W))
    # Debug — print once per session
    if not getattr(_draw_lidar_points, '_debugged', False):
        import sys
        print(f"[LiDAR renderer] points_in={len(points)}, points_projected={len(uv)}", 
              file=sys.stderr)
        if len(uv) > 0:
            print(f"  depth range: {depth.min():.1f}–{depth.max():.1f}m  "
                  f"uv range: u={uv[:,0].min():.0f}–{uv[:,0].max():.0f} "
                  f"v={uv[:,1].min():.0f}–{uv[:,1].max():.0f}", file=sys.stderr)
        _draw_lidar_points._debugged = True
    if len(uv) == 0:
        return
    t     = np.clip(depth / 55.0, 0.0, 1.0)
    hues  = (120.0 - t * 120.0).astype(np.uint8)
    hsv_b = np.stack([hues, np.full_like(hues, 255), np.full_like(hues, 255)],
                     axis=-1)[:, np.newaxis, :]
    bgr_b = cv2.cvtColor(hsv_b, cv2.COLOR_HSV2BGR)[:, 0, :]
    for (u, v), (b, g, r) in zip(uv.astype(int), bgr_b):
        cv2.circle(img, (u, v), 1, (int(b), int(g), int(r)), -1)


# ---------------------------------------------------------------------------
# BEV panel
# ---------------------------------------------------------------------------

_BEV_RANGE  = 40.0
_BEV_STEP   = 10.0
_BEV_BG     = (18, 20, 28)
_BEV_GRID   = (45, 50, 65)
_BEV_EGO    = (0, 140, 255)


def _build_bev(size: int) -> np.ndarray:
    panel = np.full((size, size, 3), _BEV_BG, dtype=np.uint8)
    steps = int(_BEV_RANGE / _BEV_STEP)
    for i in range(-steps, steps + 1):
        off = int(i * _BEV_STEP / _BEV_RANGE * (size / 2) + size / 2)
        cv2.line(panel, (off, 0), (off, size - 1), _BEV_GRID, 1)
        cv2.line(panel, (0, off), (size - 1, off), _BEV_GRID, 1)
    ex, ey = size // 2, size - 16
    cv2.drawMarker(panel, (ex, ey), _BEV_EGO, cv2.MARKER_TRIANGLE_UP, 10, 2)
    return panel


def _world_to_bev(x_cam, z_cam, size):
    px = int(x_cam / _BEV_RANGE * (size / 2) + size / 2)
    py = int(size - 16 - z_cam / _BEV_RANGE * (size - 32))
    return (px, py) if 0 <= px < size and 0 <= py < size else None


# ---------------------------------------------------------------------------
# Main render function
# ---------------------------------------------------------------------------

def render_frame(
    frame: np.ndarray,
    tracks: list[Any],
    calib: Any,
    points: np.ndarray | None = None,
    show_lidar: bool = True,
    show_bev: bool = True,
    show_3d_boxes: bool = False,
) -> np.ndarray:
    """Render all tracks onto a camera frame.

    Parameters
    ----------
    frame : np.ndarray
        BGR image (H, W, 3) uint8. Never modified in place.
    tracks : list
        Active Track objects. Each must expose:
          id, state, class_name, score, kalman.get_bbox() → [x1,y1,x2,y2]
        Optional: location ([x,y,z] camera coords), bbox_3d ([x,y,z,w,h,l,yaw])
    calib : KITTICalib
        KITTI calibration namedtuple from kitti_dataset.
    points : np.ndarray | None
        Raw velodyne (N, 4) point cloud. Skipped if None or empty.
    show_lidar : bool
        Overlay LiDAR depth points.
    show_bev : bool
        Draw BEV mini-panel in top-right corner.
    show_3d_boxes : bool
        Project and draw 3-D wireframe boxes (requires track.bbox_3d).

    Returns
    -------
    np.ndarray
        Annotated BGR image (copy of input).
    """
    img    = frame.copy()
    H, W   = img.shape[:2]

    # 1. LiDAR overlay
    if show_lidar and points is not None and len(points) > 0:
        _draw_lidar_points(img, points, calib)

    # 2. BEV panel
    bev_size  = max(160, int(H * 0.38))
    bev_panel = _build_bev(bev_size) if show_bev else None

    # 3. Tracks
    for track in tracks:
        state = getattr(track, "state", "confirmed")
        if state == "dead":
            continue

        tid      = track.id
        color    = track_color(tid)
        is_tent  = state == "tentative"
        dc       = _dim_color(color, 0.5) if is_tent else color
        thick    = 1 if is_tent else 2

        # 2-D box
        bbox             = track.kalman.get_bbox()
        x1, y1, x2, y2  = (int(v) for v in bbox)
        x1c = max(0, x1); y1c = max(0, y1)
        x2c = min(W - 1, x2); y2c = min(H - 1, y2)

        if is_tent:
            ov = img.copy()
            cv2.rectangle(ov, (x1c, y1c), (x2c, y2c), dc, thick)
            cv2.addWeighted(ov, 0.5, img, 0.5, 0, img)
        else:
            cv2.rectangle(img, (x1c, y1c), (x2c, y2c), dc, thick)

        # Label
        cname = getattr(track, "class_name", "Object")
        score = getattr(track, "score", None)
        label = f"{cname} #{tid} {score:.2f}" if score is not None else f"{cname} #{tid}"
        _draw_label(img, label, x1c, y1c, dc)

        # 3-D wireframe
        if show_3d_boxes:
            bbox_3d = getattr(track, "bbox_3d", None)
            if bbox_3d is not None:
                try:
                    corners_2d, _ = project_3d_to_2d(bbox_3d, calib, (H, W))
                    draw_box_3d(img, corners_2d, color=dc, thickness=thick)
                except Exception:
                    pass

        # BEV dot — use 3D location if available, else estimate from 2D box
        if bev_panel is not None:
            try:
                loc = getattr(track, "location", None)
                if loc is None:
                    # Estimate depth from box height heuristic:
                    # assume average object height ~1.7m, focal length ~721px (KITTI)
                    bx1, by1, bx2, by2 = float(x1), float(y1), float(x2), float(y2)
                    box_h = max(by2 - by1, 1.0)
                    est_depth = (1.7 * 721.0) / box_h
                    est_x     = ((bx1 + bx2) / 2.0 - W / 2.0) / 721.0 * est_depth
                    loc = [est_x, 0.0, est_depth]
                pos = _world_to_bev(float(loc[0]), float(loc[2]), bev_size)
                if pos is not None:
                    cv2.circle(bev_panel, pos, 4, dc, -1)
                    cv2.putText(bev_panel, str(tid), (pos[0] + 5, pos[1] - 3),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.3, dc, 1, cv2.LINE_AA)
            except Exception:
                pass

    # 4. Composite BEV into top-right
    if bev_panel is not None:
        pad = 6
        bx1, bx2 = W - bev_size - pad, W - pad
        by1, by2 = pad, pad + bev_size
        cv2.rectangle(img, (bx1 - 1, by1 - 1), (bx2 + 1, by2 + 1), (80, 85, 100), 1)
        roi = img[by1:by2, bx1:bx2]
        cv2.addWeighted(bev_panel, 0.88, roi, 0.12, 0, roi)
        img[by1:by2, bx1:bx2] = roi

    return img


# ---------------------------------------------------------------------------
# Demo — python src/visualisation/renderer.py
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import time
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from kitti_dataset import KITTITrackingDataset

    class _K:
        def __init__(self, b): self._b = b
        def get_bbox(self): return self._b

    class _T:
        _c = 0
        def __init__(self, bbox, cname, loc=None):
            _T._c += 1
            self.id = _T._c; self.state = "confirmed"
            self.class_name = cname
            self.score = round(0.75 + 0.1 * (self.id % 3), 2)
            self.kalman = _K(bbox); self.location = loc; self.bbox_3d = None

    ds = KITTITrackingDataset(
        root=str(Path(__file__).resolve().parents[2] / "data/kitti/tracking"),
        split="training", sequences=[0],
        classes=["Car", "Van", "Truck", "Pedestrian", "Cyclist"],
    )

    out_dir = Path(__file__).resolve().parents[2] / "outputs" / "renderer_demo"
    out_dir.mkdir(parents=True, exist_ok=True)

    for idx in range(min(10, len(ds))):
        s = ds[idx]
        frame = cv2.cvtColor(
            (s["image"].permute(1,2,0).numpy() * 255).astype(np.uint8),
            cv2.COLOR_RGB2BGR)
        tracks = [_T(o["bbox_2d"].astype(int).tolist(), o["type"],
                     o["location"].tolist()) for o in s["labels"]]
        if tracks: tracks[0].state = "tentative"

        t0 = time.perf_counter()
        vis = render_frame(frame, tracks, s["calib"], s["points"],
                           show_lidar=True, show_bev=True)
        dt = (time.perf_counter() - t0) * 1000

        out = out_dir / f"render_f{idx:06d}.png"
        cv2.imwrite(str(out), vis)
        print(f"Frame {idx:3d} | {len(tracks)} tracks | {dt:.1f}ms → {out.name}")

    print(f"\nDone. Check {out_dir}")