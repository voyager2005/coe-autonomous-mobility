"""
OSNetEmbedder — Appearance embedding module for the MAVERIC pipeline.

Wraps torchreid's OSNet-x0.25 model. Takes a BGR frame and a list of 2-D
bounding boxes, crops and resizes each region, and returns a batch of
L2-normalised 512-d embedding vectors.

All crops are batched into a single GPU forward pass — no per-box looping.
Loads config from configs/pipeline.yaml.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
import yaml


# ImageNet normalisation constants (torchreid uses these by default)
_MEAN = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32)
_STD  = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32)

# OSNet expected input size: (width=128, height=256)
_CROP_W = 128
_CROP_H = 256


class OSNetEmbedder:
    """Extract L2-normalised appearance embeddings for bounding box crops.

    Parameters
    ----------
    config_path : str | Path
        Path to pipeline YAML config. Defaults to ``configs/pipeline.yaml``
        resolved from the project root.

    Example
    -------
    >>> emb = OSNetEmbedder()
    >>> embeddings = emb.embed(frame_bgr, [[x1,y1,x2,y2], ...])
    # embeddings: np.ndarray of shape (N, 512), L2-normalised
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

        reid_cfg: dict[str, Any] = cfg["reid"]

        self.device: str    = reid_cfg.get("device", "cuda:0")
        self.embed_dim: int = int(reid_cfg.get("embed_dim", 512))

        # ------------------------------------------------------------------
        # 2. Resolve weights path
        # ------------------------------------------------------------------
        project_root = Path(__file__).resolve().parents[2]
        weights_path = Path(reid_cfg["weights"]).expanduser()
        if not weights_path.is_absolute():
            weights_path = project_root / weights_path

        # ------------------------------------------------------------------
        # 3. Build OSNet model via torchreid
        # ------------------------------------------------------------------
        import torchreid

        # pretrained=True: torchreid auto-downloads weights from its own
        # mirror into ~/.cache/torch/ on first run (~3 MB, very fast).
        # Once downloaded, subsequent runs load from cache instantly.
        self.model = torchreid.models.build_model(
            name="osnet_x0_25",
            num_classes=1,          # num_classes irrelevant for feature extraction
            loss="softmax",
            pretrained=True,        # auto-downloads on first run
        )

        self.model = self.model.to(self.device)
        self.model.eval()

        # Move normalisation tensors to device once
        self._mean = _MEAN.view(3, 1, 1).to(self.device)
        self._std  = _STD.view(3, 1, 1).to(self.device)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def embed(
        self,
        frame: np.ndarray,
        bboxes: list[list[float]] | np.ndarray,
    ) -> np.ndarray:
        """Extract embeddings for a list of bounding boxes in a frame.

        Parameters
        ----------
        frame : np.ndarray
            BGR image of shape ``(H, W, 3)`` uint8 (standard OpenCV format).
        bboxes : list or np.ndarray
            Each row is ``[x1, y1, x2, y2]`` in pixel coordinates.
            Out-of-bounds values are clamped automatically.

        Returns
        -------
        np.ndarray
            Shape ``(N, 512)`` float32, L2-normalised.
            Returns empty array of shape ``(0, 512)`` if bboxes is empty.
        """
        if len(bboxes) == 0:
            return np.zeros((0, self.embed_dim), dtype=np.float32)

        H, W = frame.shape[:2]
        crops = self._extract_crops(frame, bboxes, H, W)  # (N, 3, 256, 128)

        if crops is None or len(crops) == 0:
            return np.zeros((0, self.embed_dim), dtype=np.float32)

        # Move to GPU, normalise
        crops = crops.to(self.device)
        crops = (crops - self._mean) / self._std

        # Single batched forward pass
        with torch.no_grad():
            features = self.model(crops)           # (N, embed_dim)

        # L2 normalise — cosine similarity then equals dot product
        features = F.normalize(features, p=2, dim=1)

        return features.cpu().numpy().astype(np.float32)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _extract_crops(
        self,
        frame: np.ndarray,
        bboxes: list[list[float]] | np.ndarray,
        H: int,
        W: int,
    ) -> torch.Tensor | None:
        """Crop, resize, and stack all bounding box regions into a tensor.

        Returns
        -------
        torch.Tensor or None
            Shape ``(N, 3, _CROP_H, _CROP_W)`` float32 in ``[0, 1]``,
            RGB channel order. Returns None if all crops are degenerate.
        """
        import cv2

        # Convert BGR → RGB once for the whole frame
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        crop_tensors: list[torch.Tensor] = []

        for bbox in bboxes:
            x1, y1, x2, y2 = bbox

            # Clamp to image bounds
            x1 = max(0, int(x1))
            y1 = max(0, int(y1))
            x2 = min(W, int(x2))
            y2 = min(H, int(y2))

            # Skip degenerate boxes
            if x2 <= x1 or y2 <= y1:
                # Use a zero crop as placeholder so indices stay aligned
                crop_tensors.append(
                    torch.zeros(3, _CROP_H, _CROP_W, dtype=torch.float32)
                )
                continue

            crop = frame_rgb[y1:y2, x1:x2]                        # (h, w, 3)
            crop = cv2.resize(crop, (_CROP_W, _CROP_H),
                              interpolation=cv2.INTER_LINEAR)       # (256, 128, 3)
            crop_f = torch.from_numpy(crop).permute(2, 0, 1).float() / 255.0
            crop_tensors.append(crop_f)                             # (3, 256, 128)

        if not crop_tensors:
            return None

        return torch.stack(crop_tensors, dim=0)  # (N, 3, 256, 128)

    def __repr__(self) -> str:
        return (
            f"OSNetEmbedder("
            f"model=osnet_x0_25, "
            f"embed_dim={self.embed_dim}, "
            f"device={self.device})"
        )


# ----------------------------------------------------------------------
# Quick demo — run with:  python src/reid/embedder.py
# ----------------------------------------------------------------------
if __name__ == "__main__":
    import sys
    import time
    import cv2
    from pathlib import Path

    project_root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(project_root))
    from kitti_dataset import KITTITrackingDataset

    emb = OSNetEmbedder()
    print(emb)
    print()

    ds = KITTITrackingDataset(
        root=str(project_root / "data/kitti/tracking"),
        split="training",
        sequences=[0],
        classes=["Car", "Van", "Truck", "Pedestrian", "Cyclist"],
    )

    # Test on frames 0 and 1 — check same-object similarity
    sample0 = ds[0]
    sample1 = ds[1]

    for idx, sample in enumerate([sample0, sample1]):
        frame = sample["image"].permute(1, 2, 0).numpy()
        frame = (frame * 255).astype(np.uint8)
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        labels = sample["labels"]

        if not labels:
            print(f"Frame {idx}: no labels, skipping.")
            continue

        bboxes = [obj["bbox_2d"].tolist() for obj in labels]

        t0 = time.perf_counter()
        embeddings = emb.embed(frame, bboxes)
        dt = (time.perf_counter() - t0) * 1000

        print(f"--- Frame {idx} ---")
        print(f"  Crops embedded : {len(embeddings)}")
        print(f"  Embedding shape: {embeddings.shape}")
        print(f"  Inference time : {dt:.2f} ms")
        print(f"  L2 norms       : {np.linalg.norm(embeddings, axis=1).round(4)}")
        print()

    # Cross-frame similarity check
    print("--- Cross-frame similarity (same track IDs) ---")
    labels0 = sample0["labels"]
    labels1 = sample1["labels"]

    frame0 = sample0["image"].permute(1, 2, 0).numpy()
    frame0 = (frame0 * 255).astype(np.uint8)
    frame0 = cv2.cvtColor(frame0, cv2.COLOR_RGB2BGR)

    frame1 = sample1["image"].permute(1, 2, 0).numpy()
    frame1 = (frame1 * 255).astype(np.uint8)
    frame1 = cv2.cvtColor(frame1, cv2.COLOR_RGB2BGR)

    embs0 = emb.embed(frame0, [o["bbox_2d"].tolist() for o in labels0])
    embs1 = emb.embed(frame1, [o["bbox_2d"].tolist() for o in labels1])

    # Find matching track IDs across frames
    ids0 = {obj["track_id"]: i for i, obj in enumerate(labels0)}
    ids1 = {obj["track_id"]: i for i, obj in enumerate(labels1)}
    shared_ids = set(ids0.keys()) & set(ids1.keys())

    if shared_ids:
        for tid in sorted(shared_ids):
            i0, i1 = ids0[tid], ids1[tid]
            sim = float(np.dot(embs0[i0], embs1[i1]))
            print(f"  Track #{tid:3d} ({labels0[ids0[tid]]['type']:12s})  "
                  f"cosine similarity = {sim:.4f}  "
                  f"{'✓ PASS' if sim > 0.5 else '✗ LOW'}")
    else:
        print("  No shared track IDs between frame 0 and frame 1.")

    print("\nDone.")