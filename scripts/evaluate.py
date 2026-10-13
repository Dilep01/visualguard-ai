import argparse
import hashlib
import importlib.metadata
import json
import platform

from visualguard.settings import local_caches, project_path
from visualguard.training.dataset import verified_config

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Record held-out detection evaluation")
    parser.add_argument("--data", default="data/processed/ppe/dataset.yaml")
    parser.add_argument("--weights", default="models/ppe.pt")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    local_caches()
    data = verified_config(project_path(args.data))
    weights = project_path(args.weights)
    if not weights.is_file():
        parser.error("Local trained weights are required")
    from ultralytics import YOLO

    model = YOLO(str(weights))
    metrics = model.val(
        data=str(data),
        split="test",
        device=args.device,
        project=str(project_path("reports/evaluation")),
        name="test",
        workers=0,
    )
    per_class = []
    for index, class_id in enumerate(metrics.box.ap_class_index.tolist()):
        precision, recall, ap50, ap = metrics.box.class_result(index)
        per_class.append(
            {
                "class": model.names[class_id],
                "precision": float(precision),
                "recall": float(recall),
                "mAP50": float(ap50),
                "mAP50_95": float(ap),
            }
        )
    report = {
        "split": "test",
        "metrics": metrics.results_dict,
        "per_class": per_class,
        "weights_sha256": hashlib.sha256(weights.read_bytes()).hexdigest(),
        "dataset_config_sha256": hashlib.sha256(data.read_bytes()).hexdigest(),
        "ultralytics_version": importlib.metadata.version("ultralytics"),
        "platform": platform.platform(),
        "device": args.device,
        "scope": "Object detection only. Does not evaluate tracking or temporal alerts.",
    }
    destination = project_path("reports/evaluation/detection_metrics.json")
    destination.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(destination)
