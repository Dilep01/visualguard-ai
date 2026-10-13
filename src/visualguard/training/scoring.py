from collections import Counter

from visualguard.domain import Box


def score_detections(predictions: list[dict], references: list[dict], iou_threshold=0.5) -> dict:
    """Confidence-ordered, one-to-one class-aware matching at a fixed operating point."""
    matched = set()
    totals = {name: Counter(tp=0, fp=0, fn=0) for name in ("person", "head", "helmet", "vest")}
    for prediction in sorted(predictions, key=lambda item: item["confidence"], reverse=True):
        name = prediction["class"]
        candidates = [
            (Box(*prediction["box"]).iou(Box(*reference["box"])), index)
            for index, reference in enumerate(references)
            if index not in matched and reference["class"] == name
        ]
        best = max(candidates, default=(0, -1))
        if best[0] >= iou_threshold:
            matched.add(best[1])
            totals[name]["tp"] += 1
        else:
            totals[name]["fp"] += 1
    for index, reference in enumerate(references):
        if index not in matched:
            totals[reference["class"]]["fn"] += 1
    return {name: dict(counts) for name, counts in totals.items()}


def summarize_counts(counts: dict) -> dict:
    return {
        name: {
            **row,
            "precision": row["tp"] / (row["tp"] + row["fp"]) if row["tp"] + row["fp"] else None,
            "recall": row["tp"] / (row["tp"] + row["fn"]) if row["tp"] + row["fn"] else None,
        }
        for name, row in counts.items()
    }
