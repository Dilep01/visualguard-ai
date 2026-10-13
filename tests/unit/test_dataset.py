import csv

import pytest
from PIL import Image

from visualguard.settings import project_path
from visualguard.training.dataset import prepare, verified_config


def manifest_fixture(tmp_path, scenes=6, duplicate=False):
    rows = []
    for i in range(scenes):
        image = tmp_path / f"{i}.jpg"
        Image.new("RGB", (32, 32), (0 if duplicate else i * 30, 100, 100)).save(image)
        label = tmp_path / f"{i}.txt"
        label.write_text("0 0.5 0.5 0.5 0.5\n")
        rows.append(
            {
                "image": image.name,
                "label": label.name,
                "scene": f"scene{i}",
                "source": "test fixture",
                "license": "generated for test",
            }
        )
    manifest = tmp_path / "manifest.csv"
    with manifest.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    return manifest


def test_scene_split_is_reproducible_and_disjoint(tmp_path):
    manifest = manifest_fixture(tmp_path)
    a = prepare(manifest, tmp_path / "a")
    b = prepare(manifest, tmp_path / "b")
    assert a == b
    assert set(a["scenes"].values()) == {"train", "val", "test"}
    assert verified_config(tmp_path / "a/dataset.yaml").is_file()


def test_duplicate_content_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="Duplicate"):
        prepare(manifest_fixture(tmp_path, duplicate=True), tmp_path / "out")


def test_fewer_than_three_scenes_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="three independent scenes"):
        prepare(manifest_fixture(tmp_path, scenes=2), tmp_path / "out")


def test_invalid_yolo_box_is_rejected(tmp_path):
    manifest = manifest_fixture(tmp_path)
    (tmp_path / "0.txt").write_text("0 0.9 0.5 0.5 0.5")
    with pytest.raises(ValueError, match="outside image"):
        prepare(manifest, tmp_path / "out")


def test_paths_cannot_escape_project():
    with pytest.raises(ValueError, match="inside"):
        project_path("../outside.txt")


def test_corrupt_image_is_rejected(tmp_path):
    manifest = manifest_fixture(tmp_path)
    (tmp_path / "0.jpg").write_bytes(b"not an image")
    with pytest.raises(ValueError, match="Invalid image"):
        prepare(manifest, tmp_path / "out")
