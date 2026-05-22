"""
projector.py — LiDAR ↔ Camera coordinate projection utilities.

Handles all coordinate system transforms needed by the MAVERIC pipeline:
  - Velodyne sensor frame  →  Camera frame
  - Camera frame           →  Image plane (pixel coordinates)
  - 3-D bounding box       →  8-corner polygon on image
  - 3-D bounding box       →  Tight 2-D axis-aligned bounding box

All transforms use the calibration matrices loaded by KITTICalib.

Coordinate conventions (KITTI):
  Velodyne : +x forward,  +y left,   +z up
  Camera   : +x right,   +y down,   +z forward
  Image    : origin top-left, u = col (right), v = row (down)
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from kitti_dataset import KITTICalib


# ---------------------------------------------------------------------------
# Core transform helpers
# ---------------------------------------------------------------------------

def velo_to_cam(points_velo: np.ndarray, calib: "KITTICalib") -> np.ndarray:
    """Transform points from velodyne frame to (rectified) camera frame.

    Parameters
    ----------
    points_velo : np.ndarray
        Shape ``(N, 3)`` or ``(N, 4)`` — xyz or xyzi. Only xyz is used.
    calib : KITTICalib
        Calibration namedtuple from ``kitti_dataset.load_calib()``.

    Returns
    -------
    np.ndarray
        Shape ``(N, 3)`` in rectified camera coordinates.
    """
    pts = points_velo[:, :3]  # drop intensity if present
    N = len(pts)

    # Homogeneous velodyne coords: (4, N)
    pts_h = np.vstack([pts.T, np.ones((1, N), dtype=np.float32)])

    # Tr_velo_to_cam is (3, 4) → extend to (4, 4)
    Tr = np.vstack([calib.Tr_velo_to_cam, [0.0, 0.0, 0.0, 1.0]])  # (4, 4)

    # R0_rect is (3, 3) → extend to (4, 4)
    R0 = np.eye(4, dtype=np.float32)
    R0[:3, :3] = calib.R0_rect

    # Rectified camera coords: (3, N)
    cam_pts = (R0 @ Tr @ pts_h)[:3]
    return cam_pts.T  # (N, 3)


def cam_to_image(points_cam: np.ndarray, calib: "KITTICalib") -> tuple[np.ndarray, np.ndarray]:
    """Project camera-frame points onto the image plane using P2.

    Parameters
    ----------
    points_cam : np.ndarray
        Shape ``(N, 3)`` in rectified camera coordinates.

    Returns
    -------
    uv : np.ndarray
        Shape ``(N, 2)`` pixel coordinates ``[u, v]`` (col, row).
    depth : np.ndarray
        Shape ``(N,)`` depth values in metres (Z in camera frame).
    """
    N = len(points_cam)
    pts_h = np.vstack([points_cam.T, np.ones((1, N), dtype=np.float32)])  # (4, N)

    proj = calib.P2 @ pts_h          # (3, N)
    depth = proj[2]                   # Z before normalisation
    proj = proj[:2] / (depth + 1e-8)  # (2, N)  normalise

    uv = proj.T  # (N, 2)
    return uv, depth


def project_lidar_to_image(
    points: np.ndarray,
    calib: "KITTICalib",
    img_shape: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray]:
    """Project a full point cloud onto the image, returning only visible points.

    Parameters
    ----------
    points : np.ndarray
        Shape ``(N, 4)`` — raw velodyne points ``[x, y, z, intensity]``.
    calib : KITTICalib
        KITTI calibration matrices.
    img_shape : tuple[int, int]
        ``(height, width)`` of the camera image.

    Returns
    -------
    uv : np.ndarray
        Shape ``(M, 2)`` pixel coordinates of points that project inside the image.
    depth : np.ndarray
        Shape ``(M,)`` depth values for those points.
    """
    H, W = img_shape

    # Velodyne → camera
    pts_cam = velo_to_cam(points, calib)

    # Keep only points in front of camera (Z > 0)
    mask_front = pts_cam[:, 2] > 0
    pts_cam = pts_cam[mask_front]

    if len(pts_cam) == 0:
        return np.zeros((0, 2), dtype=np.float32), np.zeros(0, dtype=np.float32)

    # Project to image
    uv, depth = cam_to_image(pts_cam, calib)

    # Keep only points that fall within image bounds
    mask_img = (
        (uv[:, 0] >= 0) & (uv[:, 0] < W) &
        (uv[:, 1] >= 0) & (uv[:, 1] < H)
    )
    return uv[mask_img], depth[mask_img]


# ---------------------------------------------------------------------------
# 3-D bounding box utilities
# ---------------------------------------------------------------------------

def bbox3d_corners(bbox_3d: list[float] | np.ndarray) -> np.ndarray:
    """Compute the 8 corners of a 3-D bounding box in its LOCAL frame.

    Parameters
    ----------
    bbox_3d : array-like
        ``[x, y, z, w, h, l, yaw]`` in camera coordinates.
        x/y/z = centre, w = width (left-right), h = height (up-down),
        l = length (front-back), yaw = rotation around Y-axis (radians).

    Returns
    -------
    np.ndarray
        Shape ``(8, 3)`` — 8 corner points in camera frame.

    Corner ordering (viewed from above)::

        4 ── 5
        │    │   (front)
        7 ── 6
        │    │   (back)
        0 ── 1
        │    │
        3 ── 2
    """
    x, y, z, w, h, l, yaw = bbox_3d

    # Half-extents
    hw, hh, hl = w / 2.0, h / 2.0, l / 2.0

    # 8 corners in object frame (before rotation + translation)
    corners = np.array([
        [-hw, -hh, -hl],
        [ hw, -hh, -hl],
        [ hw, -hh,  hl],
        [-hw, -hh,  hl],
        [-hw,  hh, -hl],
        [ hw,  hh, -hl],
        [ hw,  hh,  hl],
        [-hw,  hh,  hl],
    ], dtype=np.float32)

    # Rotation matrix around Y-axis (KITTI convention)
    # KITTI yaw is counter-clockwise when viewed from above
    cos_y, sin_y = np.cos(yaw), np.sin(yaw)
    R = np.array([
        [ cos_y, 0.0, sin_y],
        [   0.0, 1.0,   0.0],
        [-sin_y, 0.0, cos_y],
    ], dtype=np.float32)

    # Rotate then translate
    corners = (R @ corners.T).T + np.array([x, y, z], dtype=np.float32)
    return corners  # (8, 3) in camera frame


def project_3d_to_2d(
    bbox_3d: list[float] | np.ndarray,
    calib: "KITTICalib",
    img_shape: tuple[int, int] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Project a 3-D bounding box onto the image plane.

    Parameters
    ----------
    bbox_3d : array-like
        ``[x, y, z, w, h, l, yaw]`` in rectified camera coordinates.
    calib : KITTICalib
        KITTI calibration matrices.
    img_shape : tuple[int, int] | None
        If given as ``(H, W)``, corners outside the image are clipped.

    Returns
    -------
    corners_2d : np.ndarray
        Shape ``(8, 2)`` — projected pixel coordinates of the 8 box corners.
    bbox_2d : np.ndarray
        Shape ``(4,)`` — tight axis-aligned 2-D box ``[x1, y1, x2, y2]``.
    """
    # Compute 3-D corners in camera frame
    corners_3d = bbox3d_corners(bbox_3d)  # (8, 3)

    # Project onto image
    uv, depth = cam_to_image(corners_3d, calib)  # (8, 2), (8,)

    # Clip to image bounds if shape provided
    if img_shape is not None:
        H, W = img_shape
        uv[:, 0] = np.clip(uv[:, 0], 0, W - 1)
        uv[:, 1] = np.clip(uv[:, 1], 0, H - 1)

    # Tight 2-D bounding box from projected corners
    x1, y1 = uv[:, 0].min(), uv[:, 1].min()
    x2, y2 = uv[:, 0].max(), uv[:, 1].max()
    bbox_2d = np.array([x1, y1, x2, y2], dtype=np.float32)

    return uv, bbox_2d


def draw_box_3d(
    image: "np.ndarray",
    corners_2d: np.ndarray,
    color: tuple[int, int, int] = (0, 255, 0),
    thickness: int = 2,
) -> "np.ndarray":
    """Draw a 3-D bounding box on an image given its 8 projected corners.

    Parameters
    ----------
    image : np.ndarray
        BGR image to draw on (modified in place).
    corners_2d : np.ndarray
        Shape ``(8, 2)`` — output of ``project_3d_to_2d()``.
    color : tuple[int, int, int]
        BGR colour.
    thickness : int
        Line thickness in pixels.

    Returns
    -------
    np.ndarray
        The annotated image (same object as input).
    """
    import cv2

    pts = corners_2d.astype(int)

    # Bottom face (indices 0-3)
    for i, j in [(0, 1), (1, 2), (2, 3), (3, 0)]:
        cv2.line(image, tuple(pts[i]), tuple(pts[j]), color, thickness)

    # Top face (indices 4-7)
    for i, j in [(4, 5), (5, 6), (6, 7), (7, 4)]:
        cv2.line(image, tuple(pts[i]), tuple(pts[j]), color, thickness)

    # Vertical pillars connecting bottom to top
    for i, j in [(0, 4), (1, 5), (2, 6), (3, 7)]:
        cv2.line(image, tuple(pts[i]), tuple(pts[j]), color, thickness)

    # Highlight front face in brighter colour so orientation is visible
    front_color = tuple(min(c + 80, 255) for c in color)
    for i, j in [(4, 5), (0, 1)]:
        cv2.line(image, tuple(pts[i]), tuple(pts[j]), front_color, thickness + 1)

    return image


# ---------------------------------------------------------------------------
# Quick demo — run with:  python src/fusion/projector.py
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import sys
    import cv2
    from pathlib import Path

    # Add project root to path so kitti_dataset is importable
    project_root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(project_root))

    from kitti_dataset import KITTITrackingDataset, load_calib

    ds = KITTITrackingDataset(
        root=str(project_root / "data/kitti/tracking"),
        split="training",
        sequences=[0],
        classes=["Car", "Van", "Truck", "Pedestrian", "Cyclist"],
    )

    out_dir = project_root / "outputs" / "projection_demo"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Test on first 5 frames
    for idx in range(min(5, len(ds))):
        sample = ds[idx]
        frame   = sample["image"].permute(1, 2, 0).numpy()
        frame   = (frame * 255).astype(np.uint8)
        frame   = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        points  = sample["points"]
        labels  = sample["labels"]
        calib   = sample["calib"]
        meta    = sample["meta"]

        H, W = frame.shape[:2]
        vis = frame.copy()

        # 1. Overlay LiDAR points coloured by depth
        if len(points) > 0:
            uv, depth = project_lidar_to_image(points, calib, (H, W))
            max_depth = 50.0
            for (u, v), d in zip(uv.astype(int), depth):
                t = min(d / max_depth, 1.0)
                # Heatmap: close=green, far=red
                r = int(255 * t)
                g = int(255 * (1 - t))
                cv2.circle(vis, (u, v), 1, (0, g, r), -1)

        # 2. Project ground truth 3-D boxes from labels
        for obj in labels:
            # KITTI label dimensions: [height, width, length]
            # KITTI location: bottom centre of box in camera coords
            # bbox3d_corners expects centre, so shift y up by h/2
            kitti_h, kitti_w, kitti_l = obj["dimensions"]
            x, y, z = obj["location"]
            yaw      = obj["rotation_y"]
            # Shift centre from bottom to middle along Y (height axis)
            y_centre = y - kitti_h / 2.0
            bbox_3d  = [x, y_centre, z, kitti_w, kitti_h, kitti_l, yaw]

            try:
                corners_2d, bbox_2d = project_3d_to_2d(bbox_3d, calib, (H, W))
                draw_box_3d(vis, corners_2d, color=(0, 255, 0), thickness=2)

                # Label
                label_txt = f"{obj['type']} #{obj['track_id']}"
                x1, y1 = int(bbox_2d[0]), int(bbox_2d[1])
                cv2.putText(vis, label_txt, (x1, y1 - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)
            except Exception as e:
                print(f"  Projection failed for {obj['type']}: {e}")

        save_path = out_dir / f"proj_seq{meta['seq']:04d}_f{meta['frame']:06d}.png"
        cv2.imwrite(str(save_path), vis)
        print(f"Saved → {save_path}  ({len(labels)} objects)")

    print("\nDone. Check outputs/projection_demo/")