"""
kalman.py — Kalman filter for bounding box tracking.

Pure numpy implementation — no filterpy dependency (incompatible with Python 3.12).
State vector: [cx, cy, w, h, vx, vy, vw, vh]
Measurement:  [cx, cy, w, h]
"""

from __future__ import annotations
import numpy as np


class KalmanBoxTracker:
    """
    Constant velocity Kalman filter for a single bounding box track.
    State: [cx, cy, w, h, vx, vy, vw, vh]
    Measurement: [cx, cy, w, h]
    """

    def __init__(self, bbox: list | np.ndarray) -> None:
        """
        Parameters
        ----------
        bbox : [x1, y1, x2, y2]
        """
        # ── Dimensions ────────────────────────────────────────────────
        self.dim_x = 8   # state
        self.dim_z = 4   # measurement

        # ── State transition matrix F (constant velocity) ─────────────
        self.F = np.eye(self.dim_x, dtype=float)
        for i in range(4):
            self.F[i, i + 4] = 1.0          # pos += vel

        # ── Measurement matrix H ──────────────────────────────────────
        self.H = np.zeros((self.dim_z, self.dim_x), dtype=float)
        self.H[:4, :4] = np.eye(4)

        # ── Measurement noise R ───────────────────────────────────────
        self.R = np.eye(self.dim_z, dtype=float)
        self.R[2:, 2:] *= 10.0              # more noise on w/h

        # ── Process noise Q ───────────────────────────────────────────
        self.Q = np.eye(self.dim_x, dtype=float)
        self.Q[4:, 4:] *= 0.01              # small velocity noise
        self.Q[-1, -1] *= 0.01

        # ── State covariance P ────────────────────────────────────────
        self.P = np.eye(self.dim_x, dtype=float)
        self.P *= 10.0
        self.P[4:, 4:] *= 1000.0           # high uncertainty on initial vel

        # ── Initial state ─────────────────────────────────────────────
        self.x = np.zeros((self.dim_x, 1), dtype=float)
        self.x[:4] = self._bbox_to_z(bbox)

        # ── Bookkeeping ───────────────────────────────────────────────
        self.time_since_update = 0
        self.hits = 0
        self.hit_streak = 0
        self.age = 0
        self.history: list[np.ndarray] = []

    # ──────────────────────────────────────────────────────────────────
    # Public interface
    # ──────────────────────────────────────────────────────────────────

    def predict(self) -> np.ndarray:
        """Advance state by one frame. Returns predicted [x1, y1, x2, y2]."""
        # Prevent width/height from going negative due to velocity
        if (self.x[6] + self.x[2]) <= 0:
            self.x[6] *= 0.0
        if (self.x[7] + self.x[3]) <= 0:
            self.x[7] *= 0.0

        # Predict step: x = F·x,  P = F·P·F^T + Q
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q

        self.age += 1
        if self.time_since_update > 0:
            self.hit_streak = 0
        self.time_since_update += 1

        self.history.append(self._x_to_bbox(self.x))
        return self.history[-1]

    def update(self, bbox: list | np.ndarray) -> None:
        """Correct state with a new measurement [x1, y1, x2, y2]."""
        self.time_since_update = 0
        self.history = []
        self.hits += 1
        self.hit_streak += 1

        z = self._bbox_to_z(bbox)

        # Innovation: y = z - H·x
        y = z - self.H @ self.x

        # Innovation covariance: S = H·P·H^T + R
        S = self.H @ self.P @ self.H.T + self.R

        # Kalman gain: K = P·H^T·S^-1
        K = self.P @ self.H.T @ np.linalg.inv(S)

        # Update state and covariance
        self.x = self.x + K @ y
        I_KH = np.eye(self.dim_x) - K @ self.H
        self.P = I_KH @ self.P

    def get_state(self) -> np.ndarray:
        """Return current bounding box estimate as [x1, y1, x2, y2]."""
        return self._x_to_bbox(self.x)

    def get_bbox(self) -> np.ndarray:
        """Alias for get_state() — used by renderer."""
        return self.get_state()

    # ──────────────────────────────────────────────────────────────────
    # Static conversion helpers
    # ──────────────────────────────────────────────────────────────────

    @staticmethod
    def _bbox_to_z(bbox) -> np.ndarray:
        """[x1, y1, x2, y2] → [[cx], [cy], [w], [h]]"""
        x1, y1, x2, y2 = float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])
        w  = x2 - x1
        h  = y2 - y1
        cx = x1 + w / 2.0
        cy = y1 + h / 2.0
        return np.array([[cx], [cy], [w], [h]], dtype=float)

    @staticmethod
    def _x_to_bbox(x) -> np.ndarray:
        """State vector → [x1, y1, x2, y2]"""
        cx = float(x[0])
        cy = float(x[1])
        w  = float(x[2])
        h  = float(x[3])
        return np.array([cx - w / 2.0, cy - h / 2.0,
                         cx + w / 2.0, cy + h / 2.0], dtype=float)

    # Keep public aliases matching the old filterpy-style API so
    # track.py / tracker.py don't need changes
    convert_bbox_to_z = _bbox_to_z.__func__   # type: ignore[attr-defined]
    convert_x_to_bbox = _x_to_bbox.__func__   # type: ignore[attr-defined]
