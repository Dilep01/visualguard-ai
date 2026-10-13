"""Fit head box residuals on training images and record all fitted pairs."""

import hashlib
import json
import math

import numpy as np

from visualguard.domain import Box
from visualguard.settings import configure_vision, project_path

if __name__ == "__main__":
    configure_vision()
    import torch
    from ultralytics import YOLO

    torch.set_num_threads(4)
    output = project_path("reports/training/head-adjustment-v3")
    output.mkdir(parents=True, exist_ok=False)
    manifest_path = project_path("data/processed/construction-ppe-v1/split_manifest.json")
    manifest = json.loads(manifest_path.read_text())
    records = [
        r
        for r in manifest["records"]
        if r["split"] == "train" and any(box[0] == 1 for box in r["labels"])
    ]
    weights = project_path("reports/training/construction-ppe-cpu-v3/weights/best-head.pt")
    model = YOLO(str(weights))
    residuals, pairs = [], []
    results = model.predict(
        [str(project_path(r["prepared_image"])) for r in records],
        conf=0.1,
        imgsz=320,
        batch=8,
        device="cpu",
        stream=True,
        verbose=False,
    )
    for record, result in zip(records, results, strict=True):
        height, width = result.orig_shape
        references = [
            Box(
                (x - w / 2) * width, (y - h / 2) * height, (x + w / 2) * width, (y + h / 2) * height
            )
            for label, x, y, w, h in record["labels"]
            if label == 1
        ]
        matched = set()
        for prediction in sorted(result.boxes, key=lambda p: float(p.conf.item()), reverse=True):
            if int(prediction.cls.item()) != 1:
                continue
            box = Box(*prediction.xyxy[0].tolist())
            overlap, index = max(
                ((box.iou(g), i) for i, g in enumerate(references) if i not in matched),
                default=(0, -1),
            )
            if overlap < 0.1:
                continue
            matched.add(index)
            target = references[index]
            w, h = box.x2 - box.x1, box.y2 - box.y1
            values = [
                ((target.x1 + target.x2) - (box.x1 + box.x2)) / (2 * w),
                ((target.y1 + target.y2) - (box.y1 + box.y2)) / (2 * h),
                math.log((target.x2 - target.x1) / w),
                math.log((target.y2 - target.y1) / h),
            ]
            residuals.append(values)
            pairs.append(
                {
                    "image": record["prepared_image"],
                    "split": "train",
                    "confidence": float(prediction.conf.item()),
                    "iou_before": overlap,
                    "prediction": vars(box),
                    "reference": vars(target),
                    "residual": values,
                }
            )
    if len(pairs) < 30:
        raise ValueError("Insufficient training pairs to fit the adjustment")
    coefficients = dict(
        zip(("dx", "dy", "log_w", "log_h"), np.median(residuals, axis=0).tolist(), strict=True)
    )
    report = {
        "split": "train",
        "images": len(records),
        "pairs": len(pairs),
        "coefficients": coefficients,
        "weights_sha256": hashlib.sha256(weights.read_bytes()).hexdigest(),
        "split_manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        "imgsz": 320,
        "fit_prediction_confidence": 0.1,
        "fit_min_iou": 0.1,
        "method": "Median normalized center and log-size residuals from training pairs",
        "scope": "Box geometry only; does not calibrate confidence probabilities.",
    }
    (output / "adjustment.json").write_text(json.dumps(report, indent=2))
    (output / "training_pairs.json").write_text(json.dumps(pairs, indent=2))
    print(json.dumps(report, indent=2))
