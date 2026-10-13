import argparse
import hashlib
import importlib.metadata
import json
import platform
from collections import Counter

import yaml

from visualguard.settings import configure_vision, project_path
from visualguard.training.dataset import NAMES, verified_config
from visualguard.training.scoring import score_detections, summarize_counts

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Record held-out detection evaluation")
    parser.add_argument("--data", default="data/processed/construction-ppe-v1/dataset.yaml")
    parser.add_argument("--weights", required=True)
    parser.add_argument("--split", choices=("val", "test"), default="test")
    parser.add_argument("--sweep", action="store_true", help="Validation-only confidence sweep")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--imgsz", type=int, default=320)
    parser.add_argument("--confidence", type=float, default=0.4)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--output", default="reports/evaluation/construction-ppe-v1")
    args = parser.parse_args()
    if args.sweep and args.split != "val":
        parser.error("Confidence sweeps are restricted to validation data")
    configure_vision()
    data = verified_config(project_path(args.data))
    weights, output = project_path(args.weights), project_path(args.output)
    if not weights.is_file():
        parser.error("Local trained weights are required")
    if output.exists():
        parser.error("Use a new output directory to preserve evaluation history")
    import torch
    from ultralytics import YOLO

    torch.set_num_threads(args.threads)
    model = YOLO(str(weights))
    if model.names != NAMES:
        parser.error("Checkpoint class mapping differs from the prepared dataset")
    metrics = model.val(
        data=str(data),
        split=args.split,
        device=args.device,
        imgsz=args.imgsz,
        batch=8,
        conf=0.001,
        project=str(output.parent),
        name=output.name,
        workers=0,
        plots=False,
    )
    per_class = []
    for index, class_id in enumerate(metrics.box.ap_class_index.tolist()):
        precision, recall, ap50, ap = metrics.box.class_result(index)
        per_class.append(
            {
                "class": model.names[class_id],
                "precision_at_best_f1": float(precision),
                "recall_at_best_f1": float(recall),
                "mAP50": float(ap50),
                "mAP50_95": float(ap),
            }
        )
    config = yaml.safe_load(data.read_text())
    root = project_path(config["path"])
    images = sorted(
        path
        for path in (root / config[args.split]).iterdir()
        if path.suffix.lower() in {".jpg", ".jpeg", ".png"}
    )
    counts = {name: Counter(tp=0, fp=0, fn=0) for name in NAMES.values()}
    predictions_log = []
    for prediction in model.predict(
        [str(image) for image in images],
        conf=min(0.05, args.confidence) if args.sweep else args.confidence,
        imgsz=args.imgsz,
        device=args.device,
        stream=True,
        batch=8,
        verbose=False,
    ):
        image_path = project_path(prediction.path)
        height, width = prediction.orig_shape
        reference = []
        label_path = root / "labels" / args.split / (image_path.stem + ".txt")
        for line in label_path.read_text().splitlines():
            label, x, y, w, h = map(float, line.split())
            reference.append(
                {
                    "class": NAMES[int(label)],
                    "box": [
                        (x - w / 2) * width,
                        (y - h / 2) * height,
                        (x + w / 2) * width,
                        (y + h / 2) * height,
                    ],
                }
            )
        predicted = [
            {
                "class": NAMES[int(box.cls.item())],
                "confidence": float(box.conf.item()),
                "box": box.xyxy[0].tolist(),
            }
            for box in prediction.boxes
        ]
        scores = score_detections(
            [item for item in predicted if item["confidence"] >= args.confidence], reference
        )
        for name in counts:
            counts[name].update(scores[name])
        predictions_log.append(
            {
                "image": image_path.name,
                "predictions": predicted,
                "reference": reference,
                "counts": scores,
            }
        )
    report = {
        "split": args.split,
        "images": len(images),
        "metrics": metrics.results_dict,
        "per_class": per_class,
        "fixed_operating_point": {
            "confidence": args.confidence,
            "iou": 0.5,
            "per_class": summarize_counts(counts),
        },
        "weights_sha256": hashlib.sha256(weights.read_bytes()).hexdigest(),
        "dataset_config_sha256": hashlib.sha256(data.read_bytes()).hexdigest(),
        "split_manifest_sha256": hashlib.sha256(
            (root / "split_manifest.json").read_bytes()
        ).hexdigest(),
        "ultralytics_version": importlib.metadata.version("ultralytics"),
        "platform": platform.platform(),
        "device": args.device,
        "imgsz": args.imgsz,
        "validation_timing": {
            "batch_size": 8,
            "milliseconds_per_image": metrics.speed,
            "scope": "Ultralytics validation batches; excludes model loading, "
            "API overhead and video encoding. Not end-to-end latency.",
        },
        "scope": f"Exploratory image-detection {args.split} evaluation; "
        "scene independence and video-event accuracy unverified.",
    }
    if args.sweep:
        sweep = []
        for threshold in (0.05, 0.1, 0.2, 0.3, 0.4, 0.5):
            totals = {name: Counter(tp=0, fp=0, fn=0) for name in NAMES.values()}
            for row in predictions_log:
                scores = score_detections(
                    [item for item in row["predictions"] if item["confidence"] >= threshold],
                    row["reference"],
                )
                for name in totals:
                    totals[name].update(scores[name])
            sweep.append({"confidence": threshold, "per_class": summarize_counts(totals)})
        report["validation_confidence_sweep"] = sweep
    (output / "detection_metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (output / "predictions.json").write_text(
        json.dumps(predictions_log, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, indent=2))
