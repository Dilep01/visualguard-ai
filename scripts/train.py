import argparse
import csv
import hashlib
import json
import platform
import time

from visualguard.settings import configure_vision, project_path
from visualguard.training.checkpoints import HeadCheckpoint
from visualguard.training.dataset import verified_config

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train PPE detector from local model weights")
    parser.add_argument("--data", default="data/processed/ppe/dataset.yaml")
    parser.add_argument(
        "--model", required=True, help="Local trusted .pt file beneath project root"
    )
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--imgsz", type=int, default=320)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--freeze", type=int, default=10)
    parser.add_argument("--name", default="ppe-baseline")
    parser.add_argument("--time-hours", type=float, default=None)
    parser.add_argument("--lr0", type=float, default=0.001)
    parser.add_argument("--class-weight-power", type=float, default=0.0)
    parser.add_argument("--mosaic", type=float, default=1.0)
    parser.add_argument("--warmup-epochs", type=float, default=3.0)
    parser.add_argument("--warmup-bias-lr", type=float, default=0.1)
    parser.add_argument("--select-head", action="store_true")
    args = parser.parse_args()
    configure_vision()
    model_path = project_path(args.model)
    if not model_path.is_file() or model_path.suffix != ".pt":
        parser.error("Supply a local .pt checkpoint; automatic downloads are disabled")
    data = verified_config(project_path(args.data))
    import torch
    from ultralytics import YOLO

    torch.set_num_threads(args.threads)
    output = project_path("reports/training") / args.name
    if not output.resolve().is_relative_to(project_path("reports/training")) or output.exists():
        parser.error("Choose a new run name inside reports/training; existing runs are preserved")
    started = time.perf_counter()

    model = YOLO(str(model_path))
    if args.select_head:
        model.add_callback("on_model_save", HeadCheckpoint(output))
    model.train(
        data=str(data),
        epochs=args.epochs,
        device=args.device,
        seed=args.seed,
        imgsz=args.imgsz,
        batch=args.batch,
        freeze=args.freeze,
        time=args.time_hours,
        optimizer="AdamW",
        lr0=args.lr0,
        cls_pw=args.class_weight_power,
        mosaic=args.mosaic,
        warmup_epochs=args.warmup_epochs,
        warmup_bias_lr=args.warmup_bias_lr,
        cache=False,
        close_mosaic=0,
        deterministic=True,
        workers=0,
        amp=False,
        project=str(project_path("reports/training")),
        name=args.name,
        exist_ok=False,
        plots=False,
    )
    metadata = {
        "arguments": vars(args),
        "wall_seconds": time.perf_counter() - started,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "base_sha256": hashlib.sha256(model_path.read_bytes()).hexdigest(),
        "split_manifest_sha256": hashlib.sha256(
            (data.parent / "split_manifest.json").read_bytes()
        ).hexdigest(),
        "scope": "Exploratory CPU transfer-learning baseline; not production validation",
    }
    with (output / "results.csv").open() as handle:
        metadata["epochs_completed"] = len(list(csv.DictReader(handle)))
    (output / "run_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
