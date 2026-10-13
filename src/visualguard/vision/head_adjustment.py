"""A fixed box adjustment fitted on training data, never a confidence recalibration."""

import math

from visualguard.domain import Box


def merge_head_duplicates(predictions: list[dict], iou: float = 0.5) -> list[dict]:
    """Fuse duplicate geometry without increasing the anchor's confidence."""
    result = [p for p in predictions if p["class"] != "head"]
    pending = sorted(
        (dict(p) for p in predictions if p["class"] == "head"),
        key=lambda p: p["confidence"],
        reverse=True,
    )
    while pending:
        anchor = pending.pop(0)
        group, remaining = [anchor], []
        for item in pending:
            target = group if Box(*anchor["box"]).iou(Box(*item["box"])) >= iou else remaining
            target.append(item)
        pending = remaining
        total = sum(item["confidence"] for item in group)
        anchor["box"] = [
            sum(item["confidence"] * item["box"][i] for item in group) / total for i in range(4)
        ]
        result.append(anchor)
    return result


def suppress_head_duplicates(predictions: list[dict], iou: float = 0.5) -> list[dict]:
    """Retain the highest-confidence box for strongly overlapping head predictions."""
    other = [p for p in predictions if p["class"] != "head"]
    heads = []
    for item in sorted(
        (p for p in predictions if p["class"] == "head"),
        key=lambda p: p["confidence"],
        reverse=True,
    ):
        if not any(Box(*item["box"]).iou(Box(*kept["box"])) >= iou for kept in heads):
            heads.append(item)
    return other + heads


def adjust_head(box: Box, coefficients: dict, width: int, height: int) -> Box:
    dx, dy, log_w, log_h = (float(coefficients[key]) for key in ("dx", "dy", "log_w", "log_h"))
    if not all(math.isfinite(value) for value in (dx, dy, log_w, log_h)):
        raise ValueError("Head adjustment coefficients must be finite")
    if abs(dx) > 0.5 or abs(dy) > 0.5 or abs(log_w) > 1.4 or abs(log_h) > 1.4:
        raise ValueError("Head adjustment exceeds supported geometry limits")
    w, h = box.x2 - box.x1, box.y2 - box.y1
    cx, cy = (box.x1 + box.x2) / 2 + dx * w, (box.y1 + box.y2) / 2 + dy * h
    w, h = w * math.exp(log_w), h * math.exp(log_h)
    return Box(
        max(0, min(width, cx - w / 2)),
        max(0, min(height, cy - h / 2)),
        max(0, min(width, cx + w / 2)),
        max(0, min(height, cy + h / 2)),
    )
