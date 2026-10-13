"""Select validation examples by labels before checking candidate predictions."""

import json
import shutil

from visualguard.domain import Box, Detection
from visualguard.settings import project_path
from visualguard.training.dataset import NAMES
from visualguard.vision.matching import associate

if __name__ == "__main__":
    destination = project_path("data/demo/head-validation")
    destination.mkdir(exist_ok=False)
    manifest = json.loads(
        project_path("data/processed/construction-ppe-v1/split_manifest.json").read_text()
    )
    selected = []
    for row in sorted(manifest["records"], key=lambda row: row["sha256"]):
        if row["split"] != "val":
            continue
        width, height = row["width"], row["height"]
        reference = [
            Detection(
                NAMES[label],
                1.0,
                Box(
                    (x - w / 2) * width,
                    (y - h / 2) * height,
                    (x + w / 2) * width,
                    (y + h / 2) * height,
                ),
            )
            for label, x, y, w, h in row["labels"]
        ]
        people = [(i + 1, item) for i, item in enumerate(reference) if item.label == "person"]
        expected = associate(people, reference, width, height)
        if not any(item.helmet == "possible_missing" for item in expected):
            continue
        source = project_path(row["prepared_image"])
        target = destination / f"sample-{len(selected) + 1:02d}{source.suffix}"
        shutil.copy2(source, target)
        selected.append(
            {
                "file": str(target.relative_to(project_path("."))).replace("\\", "/"),
                "source": str(source),
                "sha256": row["sha256"],
                "split": "val",
                "reference_possible_missing": sum(
                    item.helmet == "possible_missing" for item in expected
                ),
            }
        )
        if len(selected) == 3:
            break
    record = {
        "selection": "First three validation images by SHA256 whose reference label geometry "
        "produces a possible_missing observation under the existing association rules. "
        "Selected without consulting candidate predictions.",
        "scope": "Development integration examples, not a representative accuracy sample. "
        "Reference-derived states are not independently reviewed safety labels.",
        "license": "AGPL-3.0 as supplied in the Construction-PPE archive",
        "samples": selected,
    }
    (destination / "provenance.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
    print(json.dumps(record, indent=2))
