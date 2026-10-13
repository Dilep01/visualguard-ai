"""Evaluate the actual application detector on validation, including head postprocessing."""

import hashlib
import json
from collections import Counter

import cv2

from visualguard.settings import Settings, configure_vision, project_path
from visualguard.training.scoring import score_detections, summarize_counts
from visualguard.vision.detector import YoloDetector

if __name__ == "__main__":
    configure_vision()
    import torch

    torch.set_num_threads(4)
    source = project_path("reports/evaluation/construction-ppe-v3-validation")
    output = project_path("reports/evaluation/construction-ppe-v3-profile-validation")
    output.mkdir(parents=True, exist_ok=False)
    settings = Settings(
        backend="yolo",
        weights="reports/training/construction-ppe-cpu-v3/weights/best-head.pt",
        imgsz=320,
        confidence=0.4,
        head_confidence=0.25,
        head_profile="reports/training/head-adjustment-v3/inference_profile.json",
    )
    detector = YoloDetector(settings)
    rows = json.loads((source / "predictions.json").read_text())
    counts = {name: Counter(tp=0, fp=0, fn=0) for name in ("person", "head", "helmet", "vest")}
    for index, row in enumerate(rows):
        frame = cv2.imread(
            str(project_path("data/processed/construction-ppe-v1/images/val") / row["image"])
        )
        row["predictions"] = [
            {
                "class": d.label,
                "confidence": d.confidence,
                "box": [d.box.x1, d.box.y1, d.box.x2, d.box.y2],
            }
            for d in detector.predict(frame)
        ]
        row["counts"] = score_detections(row["predictions"], row["reference"])
        for name in counts:
            counts[name].update(row["counts"][name])
        if (index + 1) % 40 == 0:
            print(f"Evaluated {index + 1}/{len(rows)}", flush=True)
    report = json.loads((source / "detection_metrics.json").read_text())
    report["raw_detector_metrics"] = report.pop("metrics")
    report["raw_detector_per_class"] = report.pop("per_class")
    report.pop("validation_confidence_sweep", None)
    report.pop("validation_timing", None)
    report["metrics"] = {"metrics/mAP50(B)": None, "metrics/mAP50-95(B)": None}
    report["fixed_operating_point"] = {
        "confidence": 0.4,
        "head_confidence": 0.25,
        "iou": 0.5,
        "per_class": summarize_counts(counts),
    }
    report["head_profile"] = settings.head_profile
    report["head_profile_sha256"] = hashlib.sha256(
        project_path(settings.head_profile).read_bytes()
    ).hexdigest()
    report["scope"] = (
        "Actual application detector, validation only. Head boxes adjusted and fused. "
        "AP was not recomputed for postprocessed boxes; raw detector AP is separately labeled."
    )
    report["rect"] = False
    (output / "detection_metrics.json").write_text(json.dumps(report, indent=2))
    (output / "predictions.json").write_text(json.dumps(rows, indent=2))
    print(json.dumps(report["fixed_operating_point"], indent=2))
