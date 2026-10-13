"""Validation-only probe: re-inspect detected people's upper bodies for small heads."""

import json
from collections import Counter

import cv2

from visualguard.domain import Box
from visualguard.settings import configure_vision, project_path
from visualguard.training.scoring import score_detections, summarize_counts

if __name__ == "__main__":
    configure_vision()
    import torch
    from ultralytics import YOLO

    torch.set_num_threads(4)
    output = project_path("reports/evaluation/construction-ppe-v3-crop-probe")
    output.mkdir(parents=True, exist_ok=False)
    model = YOLO(str(project_path("reports/training/construction-ppe-cpu-v3/weights/best-head.pt")))
    rows = json.loads(
        project_path(
            "reports/evaluation/construction-ppe-v3-validation/predictions.json"
        ).read_text()
    )
    for index, row in enumerate(rows):
        frame = cv2.imread(
            str(project_path("data/processed/construction-ppe-v1/images/val") / row["image"])
        )
        height, width = frame.shape[:2]
        crops, offsets = [], []
        for item in row["predictions"]:
            if item["class"] != "person" or item["confidence"] < 0.4:
                continue
            x1, y1, x2, y2 = item["box"]
            w, h = x2 - x1, y2 - y1
            if w < 20 or h < 40:
                continue
            left, top = max(0, int(x1 - 0.1 * w)), max(0, int(y1 - 0.1 * h))
            right, bottom = min(width, int(x2 + 0.1 * w)), min(height, int(y1 + 0.5 * h))
            crops.append(frame[top:bottom, left:right])
            offsets.append((left, top))
        heads = [item for item in row["predictions"] if item["class"] == "head"]
        if crops:
            predictions = model.predict(
                crops, imgsz=320, conf=0.05, classes=[1], device="cpu", verbose=False, batch=8
            )
            for prediction, (left, top) in zip(predictions, offsets, strict=True):
                for box in prediction.boxes:
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    heads.append(
                        {
                            "class": "head",
                            "confidence": float(box.conf.item()),
                            "box": [x1 + left, y1 + top, x2 + left, y2 + top],
                        }
                    )
        kept = []
        for item in sorted(heads, key=lambda item: item["confidence"], reverse=True):
            if not any(Box(*item["box"]).iou(Box(*other["box"])) >= 0.5 for other in kept):
                kept.append(item)
        row["predictions"] = [item for item in row["predictions"] if item["class"] != "head"] + kept
        if (index + 1) % 40 == 0:
            print(f"Processed {index + 1}/{len(rows)} images", flush=True)
    sweep = []
    for threshold in (0.2, 0.3, 0.4, 0.5):
        counts = {name: Counter(tp=0, fp=0, fn=0) for name in ("person", "head", "helmet", "vest")}
        for row in rows:
            selected = [
                p
                for p in row["predictions"]
                if p["confidence"] >= (threshold if p["class"] == "head" else 0.4)
            ]
            scores = score_detections(selected, row["reference"])
            for name in counts:
                counts[name].update(scores[name])
        sweep.append({"head_confidence": threshold, "per_class": summarize_counts(counts)})
    (output / "predictions.json").write_text(json.dumps(rows, indent=2))
    report = {
        "split": "val",
        "images": len(rows),
        "scope": "Development crop inference probe; no test data and no independent accuracy claim",
        "sweep": sweep,
    }
    (output / "probe.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
