"""Resume an interrupted project-local run, preserving its pre-resume evidence."""

import argparse
import csv
import hashlib
import json
import shutil
import time
from datetime import UTC, datetime

import yaml

from visualguard.settings import configure_vision, project_path
from visualguard.training.checkpoints import HeadCheckpoint
from visualguard.training.dataset import verified_config

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    run = project_path(args.run)
    checkpoint = run / "weights/last.pt"
    config = yaml.safe_load((run / "args.yaml").read_text())
    data = verified_config(project_path(config["data"]))
    if not checkpoint.is_file() or (run / "run_metadata.json").exists():
        parser.error("Expected an interrupted run with last.pt and no completion metadata")
    configure_vision()
    import torch
    from ultralytics import YOLO

    torch.set_num_threads(args.threads)
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    backup = run / f"interrupted-{stamp}"
    backup.mkdir(exist_ok=False)
    for name in ("args.yaml", "results.csv", "weights/last.pt", "weights/best.pt"):
        source = run / name
        if source.exists():
            shutil.copy2(source, backup / source.name)
    with (run / "results.csv").open() as handle:
        previous_epochs = len(list(csv.DictReader(handle)))
    record = {
        "started_utc": stamp,
        "resume_checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        "previous_completed_epochs": previous_epochs,
        "original_arguments": config,
        "backup": str(backup.relative_to(project_path("."))),
        "scope": "Resumed interrupted training. Historical CSV time includes interruption; "
        "do not interpret it as continuous training time. RNG replay is not guaranteed.",
    }
    (backup / "resume.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
    started = time.perf_counter()
    model = YOLO(str(checkpoint))
    if (run / "head_selection.json").exists():
        shutil.copy2(run / "head_selection.json", backup / "head_selection.json")
        if (run / "weights/best-head.pt").exists():
            shutil.copy2(run / "weights/best-head.pt", backup / "best-head.pt")
        model.add_callback("on_model_save", HeadCheckpoint(run))
    model.train(
        resume=True,
        data=str(data),
        save_dir=str(run),
        device=config["device"],
        workers=0,
        plots=False,
    )
    record["resumed_segment_wall_seconds"] = time.perf_counter() - started
    with (run / "results.csv").open() as handle:
        record["epochs_completed"] = len(list(csv.DictReader(handle)))
    record["split_manifest_sha256"] = hashlib.sha256(
        (data.parent / "split_manifest.json").read_bytes()
    ).hexdigest()
    (run / "run_metadata.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
