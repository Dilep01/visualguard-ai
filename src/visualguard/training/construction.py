"""Audit and project Construction-PPE labels into the four-class experiment."""

import hashlib
import json
import random
import shutil
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np
import yaml
from PIL import Image, ImageDraw

from visualguard.settings import project_path
from visualguard.training.dataset import NAMES

SOURCE_NAMES = {
    0: "helmet",
    1: "gloves",
    2: "vest",
    3: "boots",
    4: "goggles",
    5: "none",
    6: "Person",
    7: "no_helmet",
    8: "no_goggle",
    9: "no_gloves",
    10: "no_boots",
}
CLASS_MAP = {6: 0, 7: 1, 0: 2, 2: 3}


def perceptual_hash(image: Image.Image) -> int:
    pixels = np.asarray(image.convert("L").resize((32, 32)), dtype=np.float32)
    coefficients = cv2.dct(pixels)[:8, :8].flatten()[1:]
    return sum(
        int(value > np.median(coefficients)) << index for index, value in enumerate(coefficients)
    )


def family_name(filename: str) -> str:
    # Roboflow augmentation/export suffixes share their original-image prefix.
    return filename.split(".rf.")[0].lower()


def normalized_labels(text: str) -> tuple[list[list[float]], int]:
    rows, corrections = [], 0
    for line in text.splitlines():
        if not line.strip():
            continue
        values = list(map(float, line.split()))
        if len(values) != 5 or not all(np.isfinite(values)) or values[0] not in SOURCE_NAMES:
            raise ValueError("Invalid class or box")
        label, x, y, width, height = values
        if width <= 0 or height <= 0:
            raise ValueError("Empty box")
        edges = [x - width / 2, y - height / 2, x + width / 2, y + height / 2]
        if min(edges) < -0.001 or max(edges) > 1.001:
            raise ValueError("Box exceeds image boundaries beyond rounding tolerance")
        clipped = [min(1, max(0, value)) for value in edges]
        corrections += int(clipped != edges)
        x1, y1, x2, y2 = clipped
        rows.append([int(label), (x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1])
    return rows, corrections


def audit_prepare(source: Path, destination: Path, report_dir: Path, seed=42) -> dict:
    source, destination, report_dir = map(project_path, (source, destination, report_dir))
    if destination.exists() and any(destination.iterdir()):
        raise ValueError("Prepared destination must be empty; use a new versioned directory")
    declared = yaml.safe_load((source / "data.yaml").read_text(encoding="utf-8"))["names"]
    if declared != SOURCE_NAMES:
        raise ValueError(
            "Source class definitions differ from the reviewed Construction-PPE mapping"
        )
    records, rejected = [], []
    raw_counts = Counter()
    for split in ("train", "val", "test"):
        images = sorted((source / "images" / split).glob("*"))
        for image_path in images:
            if image_path.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
                continue
            label_path = source / "labels" / split / (image_path.stem + ".txt")
            try:
                with Image.open(image_path) as image:
                    image.load()
                    width, height = image.size
                    fingerprint = perceptual_hash(image)
                labels, corrections = normalized_labels(label_path.read_text())
            except (OSError, ValueError) as exc:
                rejected.append({"image": image_path.name, "reason": str(exc)})
                continue
            raw_counts.update(SOURCE_NAMES[int(row[0])] for row in labels)
            mapped = [
                [CLASS_MAP[int(row[0])], *row[1:]] for row in labels if int(row[0]) in CLASS_MAP
            ]
            records.append(
                {
                    "image": str(image_path),
                    "source_split": split,
                    "sha256": hashlib.sha256(image_path.read_bytes()).hexdigest(),
                    "phash": fingerprint,
                    "family": family_name(image_path.name),
                    "width": width,
                    "height": height,
                    "labels": mapped,
                    "rounding_corrections": corrections,
                }
            )
    if len(records) < 3:
        raise ValueError("No usable dataset found at the supplied source path")

    # Connected components keep exact copies, export siblings, and likely near-copies together.
    parent = list(range(len(records)))

    def find(index):
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    suspicious_pairs = []
    for i, first in enumerate(records):
        for j in range(i):
            second = records[j]
            distance = (first["phash"] ^ second["phash"]).bit_count()
            same = first["sha256"] == second["sha256"] or first["family"] == second["family"]
            if same or distance <= 4:
                parent[find(i)] = find(j)
                if first["source_split"] != second["source_split"]:
                    suspicious_pairs.append(
                        {
                            "first": first["image"],
                            "second": second["image"],
                            "phash_distance": distance,
                            "exact": first["sha256"] == second["sha256"],
                        }
                    )
    groups = defaultdict(list)
    for index, record in enumerate(records):
        groups[find(index)].append(record)
    group_keys = sorted(groups)
    random.Random(seed).shuffle(group_keys)
    n_test, n_val = max(1, round(len(groups) * 0.15)), max(1, round(len(groups) * 0.15))
    counts, label_counts, kept, exact_dropped = Counter(), defaultdict(Counter), [], []
    seen = set()
    for group_index, key in enumerate(group_keys):
        split = (
            "test" if group_index < n_test else "val" if group_index < n_test + n_val else "train"
        )
        for record in groups[key]:
            if record["sha256"] in seen:
                exact_dropped.append(record["image"])
                continue
            seen.add(record["sha256"])
            image_path = Path(record["image"])
            images, labels = destination / "images" / split, destination / "labels" / split
            images.mkdir(parents=True, exist_ok=True)
            labels.mkdir(parents=True, exist_ok=True)
            image_name = record["sha256"] + image_path.suffix.lower()
            shutil.copy2(image_path, images / image_name)
            label_text = "\n".join(
                f"{int(row[0])} " + " ".join(f"{v:.8f}" for v in row[1:])
                for row in record["labels"]
            )
            (labels / f"{record['sha256']}.txt").write_text(label_text, encoding="utf-8")
            counts[split] += 1
            label_counts[split].update(NAMES[row[0]] for row in record["labels"])
            kept.append(
                {
                    **record,
                    "split": split,
                    "proxy_group": key,
                    "prepared_image": str(images / image_name),
                }
            )
    if set(counts) != {"train", "val", "test"}:
        raise ValueError("Grouping did not produce three non-empty partitions")
    (destination / "dataset.yaml").write_text(
        yaml.safe_dump(
            {
                "path": str(destination),
                "train": "images/train",
                "val": "images/val",
                "test": "images/test",
                "names": NAMES,
            }
        ),
        encoding="utf-8",
    )
    manifest = {
        "seed": seed,
        "records": kept,
        "grouping": "filename family + SHA256 + pHash<=4",
        "scene_independence_verified": False,
        "head_semantics": "Mapped source no_helmet: explicitly annotated uncovered head",
    }
    (destination / "split_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    report = {
        "source": str(source),
        "seed": seed,
        "source_images": len(records) + len(rejected),
        "source_valid_images_by_split": dict(Counter(r["source_split"] for r in records)),
        "source_annotations_by_class": dict(raw_counts),
        "rejected_images": rejected,
        "cross_source_split_duplicate_pairs": suspicious_pairs,
        "exact_duplicates_removed": exact_dropped,
        "proxy_groups": len(groups),
        "prepared_images_by_split": dict(counts),
        "prepared_labels_by_split": dict(label_counts),
        "box_rounding_corrections": sum(r["rounding_corrections"] for r in records),
        "class_mapping": {SOURCE_NAMES[k]: NAMES[v] for k, v in CLASS_MAP.items()},
        "limitations": [
            "Source camera/site/scene IDs are unavailable.",
            "Proxy grouping reduces duplicate leakage, but does not establish scene independence.",
            "This is an exploratory split; an independently sourced test set is still needed.",
        ],
    }
    report_dir.mkdir(parents=True, exist_ok=True)
    (report_dir / "data_audit.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    # Annotated train-only examples; final-test media is not used for model selection.
    selected = []
    for class_id in range(4):
        selected.extend(
            [
                r
                for r in kept
                if r["split"] == "train" and any(label[0] == class_id for label in r["labels"])
            ][:3]
        )
    sheet = Image.new("RGB", (1200, 4 * 310), "#111827")
    colors = {0: "cyan", 1: "red", 2: "yellow", 3: "lime"}
    for index, record in enumerate(selected):
        with Image.open(record["image"]) as image:
            tile = image.convert("RGB").resize((400, 280))
        draw = ImageDraw.Draw(tile)
        for label, x, y, width, height in record["labels"]:
            box = (
                (x - width / 2) * 400,
                (y - height / 2) * 280,
                (x + width / 2) * 400,
                (y + height / 2) * 280,
            )
            draw.rectangle(box, outline=colors[label], width=2)
            draw.text((box[0], max(0, box[1] - 12)), NAMES[label], fill=colors[label])
        x, y = index % 3 * 400, index // 3 * 310
        sheet.paste(tile, (x, y))
        ImageDraw.Draw(sheet).text((x + 4, y + 282), Path(record["image"]).name[:45], fill="white")
    sheet.save(report_dir / "training-label-review.jpg")
    return report
