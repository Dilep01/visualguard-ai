import argparse
from pathlib import Path

from visualguard.training.dataset import prepare

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Validate YOLO labels and split by scene")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("data/processed/ppe"))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    report = prepare(args.manifest, args.output, args.seed)
    print(f"Prepared {len(report['images'])} images from {len(report['scenes'])} scenes")
