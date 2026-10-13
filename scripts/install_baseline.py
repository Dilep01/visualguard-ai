"""Install the evaluated research checkpoint and prepare a traceable validation sample."""

import csv
import hashlib
import json
import shutil

from visualguard.settings import project_path

if __name__ == "__main__":
    evaluation = json.loads(
        project_path(
            "reports/evaluation/construction-ppe-v1-complete/detection_metrics.json"
        ).read_text()
    )
    run = project_path("reports/training/construction-ppe-cpu-v1")
    checkpoint = run / "weights/best.pt"
    digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    if digest != evaluation["weights_sha256"]:
        raise ValueError("Evaluation does not match the checkpoint")
    target = project_path("models/construction-ppe-cpu-v1.pt")
    if target.exists() and hashlib.sha256(target.read_bytes()).hexdigest() != digest:
        raise ValueError("Refusing to overwrite a different checkpoint")
    shutil.copy2(checkpoint, target)
    manifest = json.loads(
        project_path("data/processed/construction-ppe-v1/split_manifest.json").read_text()
    )
    candidates = sorted(
        (
            row
            for row in manifest["records"]
            if row["split"] == "val" and {0, 2, 3}.issubset({box[0] for box in row["labels"]})
        ),
        key=lambda row: row["sha256"],
    )
    sample = candidates[
        0
    ]  # Fixed deterministic selection, not selected for successful predictions.
    destination = project_path("data/demo/real-ppe-sample" + project_path(sample["image"]).suffix)
    shutil.copy2(project_path(sample["image"]), destination)
    project_path("data/demo/real-ppe-sample.provenance.json").write_text(
        json.dumps(
            {
                "source": sample["image"],
                "sha256": sample["sha256"],
                "split": "val",
                "license": "AGPL-3.0 as supplied in the Construction-PPE archive",
                "selection": "First validation image by SHA256 "
                "containing person, helmet, vest labels",
                "use": "UI demonstration only, not additional independent evaluation",
            },
            indent=2,
        )
    )
    with (run / "results.csv").open() as handle:
        epochs = len(list(csv.DictReader(handle)))
    report = {
        "name": "Construction-PPE CPU baseline v1",
        "weights_sha256": digest,
        "weights": "models/construction-ppe-cpu-v1.pt",
        "epochs_completed": epochs,
        "summary": "A locally trained YOLO11n baseline. At confidence 0.4, it detected none "
        "of 58 uncovered-head annotations in the exploratory test partition.",
        "evaluation": evaluation,
        "missing_helmet_monitor_ready": False,
        "sample": str(destination.relative_to(project_path("."))).replace("\\", "/"),
        "limitations": [
            "Uncovered-head recall is 0% at confidence 0.4; "
            "missing-helmet monitoring is not ready.",
            "Scene independence is unverified. "
            "Similar-image grouping is only a leakage reduction measure.",
            "Visual inspection found non-workplace head examples and inconsistent person boxes.",
            "Vest absence remains unknown; image metrics do not measure temporal event accuracy.",
        ],
    }
    project_path("reports/baseline/model_status.json").write_text(json.dumps(report, indent=2))
    env = project_path(".env")
    if not env.exists():
        env.write_text(
            "VG_BACKEND=yolo\nVG_WEIGHTS=models/construction-ppe-cpu-v1.pt\n"
            "VG_DEVICE=cpu\nVG_IMGSZ=320\nVG_CONFIDENCE=0.4\n"
        )
    print(
        json.dumps(
            {
                "checkpoint": str(target),
                "sample": str(destination),
                "epochs_completed": epochs,
                "monitor_ready": False,
            },
            indent=2,
        )
    )
