"""Record candidate selection and a review sheet without changing the active model."""

import argparse
import json

from PIL import Image, ImageDraw, ImageFont

from visualguard.settings import project_path
from visualguard.training.comparison import compare_candidate


def percentage(value):
    return "undefined" if value is None else f"{value:.1%}"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    baseline_path, candidate_path = project_path(args.baseline), project_path(args.candidate)
    baseline, candidate = (json.loads(path.read_text()) for path in (baseline_path, candidate_path))
    result = compare_candidate(baseline, candidate)
    output = project_path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    result.update(baseline_report=args.baseline, candidate_report=args.candidate)
    (output / "comparison.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    lines = [
        "# Candidate validation comparison",
        "",
        f"{result['images']} validation images, input {result['imgsz']}, "
        f"confidence 0.4 (heads {result['head_confidence']}), IoU 0.5.",
        "No test images were evaluated for this comparison.",
        "",
        "| Class | Baseline precision | Candidate precision | Baseline recall | Candidate recall |",
        "|---|---:|---:|---:|---:|",
    ]
    for name, row in baseline["fixed_operating_point"]["per_class"].items():
        new = candidate["fixed_operating_point"]["per_class"][name]
        values = (row["precision"], new["precision"], row["recall"], new["recall"])
        lines.append(f"| {name} | " + " | ".join(map(percentage, values)) + " |")
    lines += ["", f"Eligible for experimental use: {result['eligible_for_experimental_use']}.", ""]
    for gate in result["gates"]:
        lines.append(
            f"- {gate['criterion']}: {gate['actual']}; minimum {gate['minimum']}; "
            f"{'pass' if gate['passed'] else 'fail'}."
        )
    lines += [
        "",
        result["scope"],
        "",
        "Review sheet: up to four examples each with missed heads, false head detections, "
        "and correct head detections, sorted by filename within each category. "
        "Categories may overlap. Cyan boxes are reference uncovered heads; "
        f"orange boxes are model head predictions at confidence {result['head_confidence']}. "
        "This is an error review, not a random accuracy sample.",
    ]
    (output / "comparison.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    predictions = json.loads((candidate_path.parent / "predictions.json").read_text())
    selected = []
    for category, key in (
        ("Missed heads", "fn"),
        ("False detections", "fp"),
        ("Correct heads", "tp"),
    ):
        rows = sorted(
            (row for row in predictions if row["counts"]["head"][key] > 0),
            key=lambda row: row["image"],
        )[:4]
        selected.extend((category, row) for row in rows)
    manifest = json.loads(
        project_path("data/processed/construction-ppe-v1/split_manifest.json").read_text()
    )
    image_paths = {
        project_path(row["prepared_image"]).name: project_path(row["prepared_image"])
        for row in manifest["records"]
        if row["split"] == "val"
    }
    if selected:
        (output / "review_selection.json").write_text(
            json.dumps(
                [
                    {"category": category, "image": row["image"], "counts": row["counts"]["head"]}
                    for category, row in selected
                ],
                indent=2,
            ),
            encoding="utf-8",
        )
        sheet = Image.new("RGB", (1000, ((len(selected) + 1) // 2) * 360), "#101a2a")
        font = ImageFont.load_default(size=16)
        for index, (category, row) in enumerate(selected):
            with Image.open(image_paths[row["image"]]) as source:
                panel = source.convert("RGB")
            drawing = ImageDraw.Draw(panel)
            for box in row["reference"]:
                if box["class"] == "head":
                    drawing.rectangle(box["box"], outline="#00eeff", width=3)
            for box in row["predictions"]:
                if box["class"] == "head" and box["confidence"] >= result["head_confidence"]:
                    drawing.rectangle(box["box"], outline="#ff9e35", width=3)
                    drawing.text(
                        box["box"][:2], f"{box['confidence']:.2f}", fill="#ff9e35", font=font
                    )
            panel.thumbnail((480, 305))
            x, y = (index % 2) * 500 + 10, (index // 2) * 360 + 38
            sheet.paste(panel, (x, y))
            label = f"{category}: TP/FP/FN " + "/".join(
                str(row["counts"]["head"][key]) for key in ("tp", "fp", "fn")
            )
            ImageDraw.Draw(sheet).text((x, y - 28), label, fill="white", font=font)
        sheet.save(output / "head-review.jpg", quality=90)
    print(json.dumps(result, indent=2))
