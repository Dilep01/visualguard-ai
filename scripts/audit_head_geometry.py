"""Flag head/person annotation geometry for review, without editing source labels."""

import json
from collections import Counter

from PIL import Image, ImageDraw, ImageFont

from visualguard.domain import Box
from visualguard.settings import project_path

if __name__ == "__main__":
    manifest = json.loads(
        project_path("data/processed/construction-ppe-v1/split_manifest.json").read_text()
    )
    counts = {split: Counter() for split in ("train", "val")}
    flags = []
    examples = []
    for row in manifest["records"]:
        if row["split"] not in counts:
            continue
        boxes = {class_id: [] for class_id in range(4)}
        for class_id, x, y, width, height in row["labels"]:
            boxes[class_id].append(
                Box(x - width / 2, y - height / 2, x + width / 2, y + height / 2)
            )
        if not boxes[1]:
            continue
        totals = counts[row["split"]]
        totals["head_images"] += 1
        totals["head_boxes"] += len(boxes[1])
        totals["head_images_without_person_labels"] += not bool(boxes[0])
        flagged = False
        for head in boxes[1]:
            coverage = max(
                (head.intersection(person) / head.area for person in boxes[0]), default=0
            )
            if coverage < 0.8:
                flagged = True
                totals["head_boxes_not_80_percent_inside_any_person"] += 1
                flags.append(
                    {
                        "image": row["prepared_image"],
                        "split": row["split"],
                        "head": vars(head),
                        "max_person_coverage": coverage,
                        "status": "needs_manual_review_not_proven_label_error",
                    }
                )
        if flagged:
            examples.append((row, boxes))
    report = {
        "scope": "Train and validation label geometry only; no labels changed. "
        "Incomplete person labels and close crops can both explain flags.",
        "counts": counts,
        "flags": flags,
    }
    project_path("reports/baseline/head-label-geometry.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    # First six flagged training images in manifest order; validation is not relabeled.
    selected = [(row, boxes) for row, boxes in examples if row["split"] == "train"][:6]
    if selected:
        sheet = Image.new("RGB", (1000, ((len(selected) + 1) // 2) * 360), "#101a2a")
        font = ImageFont.load_default(size=16)
        for index, (row, boxes) in enumerate(selected):
            with Image.open(project_path(row["prepared_image"])) as source:
                panel = source.convert("RGB")
            draw = ImageDraw.Draw(panel)
            for class_id, color in ((0, "#00eeff"), (1, "#ff9e35")):
                for box in boxes[class_id]:
                    draw.rectangle(
                        [
                            box.x1 * panel.width,
                            box.y1 * panel.height,
                            box.x2 * panel.width,
                            box.y2 * panel.height,
                        ],
                        outline=color,
                        width=3,
                    )
            panel.thumbnail((480, 305))
            x, y = (index % 2) * 500 + 10, (index // 2) * 360 + 38
            sheet.paste(panel, (x, y))
            ImageDraw.Draw(sheet).text(
                (x, y - 28), project_path(row["image"]).name, fill="white", font=font
            )
        sheet.save(project_path("reports/baseline/head-label-geometry.jpg"), quality=90)
    print(json.dumps(counts, indent=2))
