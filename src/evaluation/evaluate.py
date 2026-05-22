"""
evaluate.py — MOT evaluation for the MAVERIC pipeline.

Loads KITTI ground truth labels and predicted track logs from
outputs/tracks/, computes per-sequence and aggregate metrics using
motmetrics, and saves results to outputs/evaluation/results.csv.

Run with:
    python src/evaluation/evaluate.py
    python src/evaluation/evaluate.py --sequences 0 1 2
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from rich.console import Console
from rich.table import Table

import motmetrics as mm

# ── Path setup ────────────────────────────────────────────────────────────────
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
for _p in [str(_PROJECT_ROOT), str(_PROJECT_ROOT / "src")]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from kitti_dataset import KITTITrackingDataset, parse_label_file

console = Console()


# ── IoU distance for motmetrics ───────────────────────────────────────────────

def _iou_dist(gt_boxes: np.ndarray, pred_boxes: np.ndarray) -> np.ndarray:
    """
    Compute pairwise IoU distance matrix between gt and pred boxes.
    Both inputs: (N, 4) as [x1, y1, x2, y2].
    Returns (N_gt, N_pred) distance matrix where dist = 1 - IoU.
    """
    if len(gt_boxes) == 0 or len(pred_boxes) == 0:
        return np.empty((len(gt_boxes), len(pred_boxes)))

    ix1 = np.maximum(gt_boxes[:, 0:1], pred_boxes[:, 0])
    iy1 = np.maximum(gt_boxes[:, 1:2], pred_boxes[:, 1])
    ix2 = np.minimum(gt_boxes[:, 2:3], pred_boxes[:, 2])
    iy2 = np.minimum(gt_boxes[:, 3:4], pred_boxes[:, 3])

    iw = np.maximum(ix2 - ix1, 0)
    ih = np.maximum(iy2 - iy1, 0)
    inter = iw * ih

    area_gt   = (gt_boxes[:, 2] - gt_boxes[:, 0]) * (gt_boxes[:, 3] - gt_boxes[:, 1])
    area_pred = (pred_boxes[:, 2] - pred_boxes[:, 0]) * (pred_boxes[:, 3] - pred_boxes[:, 1])
    union = area_gt[:, None] + area_pred[None, :] - inter + 1e-6

    iou = inter / union
    return 1.0 - iou


# ── Load ground truth for one sequence ───────────────────────────────────────

def _load_gt(kitti_root: Path, seq_id: int) -> dict[int, list[dict]]:
    """Load KITTI ground truth labels for a sequence. Returns {frame: [objs]}."""
    label_path = kitti_root / "tracking/training/label_02" / f"{seq_id:04d}.txt"
    if not label_path.exists():
        console.print(f"[yellow]Warning: GT labels not found at {label_path}[/]")
        return {}
    return parse_label_file(str(label_path))


# ── Load predictions for one sequence ────────────────────────────────────────

def _load_pred(track_log_dir: Path, seq_id: int) -> dict[int, list[dict]]:
    """Load pipeline track log for a sequence. Returns {frame: [tracks]}."""
    log_path = track_log_dir / f"seq_{seq_id:04d}.json"
    if not log_path.exists():
        console.print(f"[yellow]Warning: Track log not found at {log_path}[/]")
        return {}
    with open(log_path) as f:
        data = json.load(f)
    result: dict[int, list[dict]] = {}
    for entry in data:
        result[entry["frame"]] = entry["tracks"]
    return result


# ── Evaluate one sequence ─────────────────────────────────────────────────────

def evaluate_sequence(
    seq_id: int,
    kitti_root: Path,
    track_log_dir: Path,
    classes: list[str],
    max_iou_dist: float = 0.5,
) -> mm.MOTAccumulator:
    """Build a motmetrics accumulator for one sequence."""
    gt_frames   = _load_gt(kitti_root, seq_id)
    pred_frames = _load_pred(track_log_dir, seq_id)

    acc = mm.MOTAccumulator(auto_id=True)

    all_frames = sorted(set(gt_frames.keys()) | set(pred_frames.keys()))

    for frame_id in all_frames:
        gt_objs   = gt_frames.get(frame_id, [])
        pred_objs = pred_frames.get(frame_id, [])

        # Filter GT to relevant classes, exclude DontCare
        gt_filtered = [
            o for o in gt_objs
            if o["type"].capitalize() in [c.capitalize() for c in classes]
            and o["type"] != "DontCare"
        ]

        # GT: ids and boxes
        gt_ids   = [o["track_id"]          for o in gt_filtered]
        gt_boxes = np.array([o["bbox_2d"]  for o in gt_filtered], dtype=np.float32) \
                   if gt_filtered else np.empty((0, 4), dtype=np.float32)

        # Pred: ids and boxes
        pred_ids   = [t["id"]              for t in pred_objs]
        pred_boxes = np.array([t["bbox"]   for t in pred_objs], dtype=np.float32) \
                     if pred_objs else np.empty((0, 4), dtype=np.float32)

        # Distance matrix
        if len(gt_boxes) > 0 and len(pred_boxes) > 0:
            dist_matrix = _iou_dist(gt_boxes, pred_boxes)
            dist_matrix[dist_matrix > max_iou_dist] = np.nan
        else:
            dist_matrix = mm.distances.iou_matrix(
                gt_boxes, pred_boxes, max_iou=1.0 - max_iou_dist
            ) if len(gt_boxes) + len(pred_boxes) > 0 else np.empty((0, 0))

        acc.update(gt_ids, pred_ids, dist_matrix)

    return acc


# ── Format metrics dict for display ──────────────────────────────────────────

_METRIC_NAMES = [
    "num_frames", "mota", "motp", "idf1",
    "num_switches", "mostly_tracked", "partially_tracked", "mostly_lost",
    "num_false_positives", "num_misses",
]

_DISPLAY_NAMES = {
    "num_frames":          "Frames",
    "mota":                "MOTA ↑",
    "motp":                "MOTP ↓",
    "idf1":                "IDF1 ↑",
    "num_switches":        "ID Switches ↓",
    "mostly_tracked":      "MT ↑",
    "partially_tracked":   "PT",
    "mostly_lost":         "ML ↓",
    "num_false_positives": "FP ↓",
    "num_misses":          "FN ↓",
}


def _fmt(val) -> str:
    if isinstance(val, float):
        return f"{val:.4f}" if abs(val) < 10 else f"{val:.1f}"
    return str(val)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="MAVERIC MOT Evaluation")
    parser.add_argument("--config",    default="configs/pipeline.yaml")
    parser.add_argument("--sequences", nargs="+", type=int, default=None)
    parser.add_argument("--max-iou-dist", type=float, default=0.5,
                        help="Max IoU distance for GT↔pred matching (default 0.5)")
    args = parser.parse_args()

    cfg_path = Path(args.config).expanduser()
    if not cfg_path.is_absolute():
        cfg_path = _PROJECT_ROOT / cfg_path
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)

    kitti_root    = Path(cfg["data"]["kitti_root"]).expanduser()
    track_log_dir = Path(cfg["output"]["tracks_dir"]).expanduser()
    eval_dir      = Path(cfg["output"]["eval_dir"]).expanduser()
    eval_dir.mkdir(parents=True, exist_ok=True)

    sequences = args.sequences or cfg["data"].get("sequences") or [0, 1, 2]
    classes   = cfg["data"].get("classes", ["Car", "Van", "Truck", "Pedestrian", "Cyclist"])

    # ── Per-sequence evaluation ───────────────────────────────────────
    accs   = []
    names  = []

    for seq_id in sequences:
        console.print(f"[cyan]Evaluating sequence {seq_id:04d}...[/]")
        acc = evaluate_sequence(
            seq_id, kitti_root, track_log_dir, classes, args.max_iou_dist
        )
        accs.append(acc)
        names.append(f"seq_{seq_id:04d}")

    # ── Compute metrics ───────────────────────────────────────────────
    mh = mm.metrics.create()

    summary_per_seq = mh.compute_many(
        accs, metrics=_METRIC_NAMES, names=names, generate_overall=False
    )
    summary_overall = mh.compute_many(
        accs, metrics=_METRIC_NAMES, names=["OVERALL"], generate_overall=True
    )
    summary = pd.concat([summary_per_seq, summary_overall.loc[["OVERALL"]]])

    # ── Rich table ────────────────────────────────────────────────────
    table = Table(title="MAVERIC — MOT Evaluation Results", show_lines=True)
    table.add_column("Sequence", style="cyan", no_wrap=True)
    for m in _METRIC_NAMES:
        table.add_column(_DISPLAY_NAMES.get(m, m), justify="right")

    for idx in summary.index:
        row_vals = [_fmt(summary.loc[idx, m]) for m in _METRIC_NAMES]
        style = "bold green" if idx == "OVERALL" else None
        table.add_row(str(idx), *row_vals, style=style)

    console.print()
    console.print(table)

    # ── Pass / Fail ───────────────────────────────────────────────────
    overall = summary.loc["OVERALL"]
    mota = float(overall["mota"])
    idf1 = float(overall["idf1"])

    console.print()
    console.print(f"[bold]MOTA:[/] {mota:.4f}  ", end="")
    if mota >= 0.65:
        console.print("[bold green]✓ GOOD (≥ 0.65)[/]")
    elif mota >= 0.55:
        console.print("[bold yellow]~ MINIMUM (≥ 0.55)[/]")
    else:
        console.print("[bold red]✗ BELOW MINIMUM (< 0.55)[/]")

    console.print(f"[bold]IDF1:[/] {idf1:.4f}  ", end="")
    if idf1 >= 0.70:
        console.print("[bold green]✓ GOOD (≥ 0.70)[/]")
    elif idf1 >= 0.60:
        console.print("[bold yellow]~ MINIMUM (≥ 0.60)[/]")
    else:
        console.print("[bold red]✗ BELOW MINIMUM (< 0.60)[/]")

    # ── Save CSV ──────────────────────────────────────────────────────
    csv_path = eval_dir / "results.csv"
    summary.to_csv(str(csv_path))
    console.print(f"\n[dim]Results saved → {csv_path}[/]")


if __name__ == "__main__":
    main()
