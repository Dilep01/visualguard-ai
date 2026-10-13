"""Rescore cached baseline validation predictions at the candidate head threshold."""

import json
from collections import Counter

from visualguard.settings import project_path
from visualguard.training.scoring import score_detections, summarize_counts

if __name__ == "__main__":
    source = project_path("reports/evaluation/construction-ppe-v1-validation")
    output = project_path("reports/evaluation/construction-ppe-v1-validation-head025")
    output.mkdir(parents=True, exist_ok=False)
    report = json.loads((source / "detection_metrics.json").read_text())
    rows = json.loads((source / "predictions.json").read_text())
    counts = {n: Counter(tp=0, fp=0, fn=0) for n in ("person", "head", "helmet", "vest")}
    for row in rows:
        row["predictions"] = [
            p
            for p in row["predictions"]
            if p["confidence"] >= (0.25 if p["class"] == "head" else 0.4)
        ]
        row["counts"] = score_detections(row["predictions"], row["reference"])
        for name in counts:
            counts[name].update(row["counts"][name])
    report["fixed_operating_point"] = {
        "confidence": 0.4,
        "head_confidence": 0.25,
        "iou": 0.5,
        "per_class": summarize_counts(counts),
    }
    report["scope"] = (
        "Cached baseline validation predictions rescored at head confidence 0.25; "
        "other classes 0.4. Original baseline preprocessing retained."
    )
    (output / "detection_metrics.json").write_text(json.dumps(report, indent=2))
    (output / "predictions.json").write_text(json.dumps(rows, indent=2))
