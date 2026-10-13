import zipfile

import pytest
from PIL import Image

from visualguard.training.construction import family_name, normalized_labels, perceptual_hash
from visualguard.training.scoring import score_detections, summarize_counts


def test_source_labels_preserve_explicit_no_helmet_evidence():
    rows, corrections = normalized_labels("7 0.5 0.5 0.2 0.2\n0 0.5 0.5 0.4 0.4")
    assert rows[0][0] == 7
    assert corrections == 0


def test_label_clipping_only_allows_rounding_error():
    rows, corrections = normalized_labels("6 0.5 0.5 1.00001 1.0")
    assert corrections == 1
    assert rows[0][3] == 1
    with pytest.raises(ValueError):
        normalized_labels("6 0.5 0.5 1.1 1.0")


@pytest.mark.parametrize("label", ["99 0.5 0.5 0.2 0.2", "7 nan 0.5 0.2 0.2", "7 0.5 0.5 0 0"])
def test_invalid_labels_are_rejected(label):
    with pytest.raises(ValueError):
        normalized_labels(label)


def test_proxy_hash_and_family_are_stable():
    image = Image.new("RGB", (50, 50), "red")
    assert perceptual_hash(image) == perceptual_hash(image.copy())
    assert family_name("scene.jpg.rf.abc.jpg") == family_name("scene.jpg.rf.xyz.jpg")


def test_detection_evaluation_does_not_double_count_one_reference():
    reference = [{"class": "head", "box": [0, 0, 10, 10]}]
    prediction = {"class": "head", "box": [0, 0, 10, 10], "confidence": 0.8}
    counts = score_detections([prediction, prediction], reference)
    assert counts["head"] == {"tp": 1, "fp": 1, "fn": 0}
    assert summarize_counts(counts)["head"]["precision"] == 0.5


def test_detection_class_mismatch_is_fp_and_fn():
    counts = score_detections(
        [{"class": "helmet", "box": [0, 0, 10, 10], "confidence": 0.9}],
        [{"class": "head", "box": [0, 0, 10, 10]}],
    )
    assert counts["helmet"]["fp"] == 1
    assert counts["head"]["fn"] == 1
    assert summarize_counts(counts)["head"]["recall"] == 0


def test_empty_detection_denominators_are_null():
    assert summarize_counts(score_detections([], []))["head"]["recall"] is None


def test_archive_path_traversal_is_rejected(tmp_path):
    # Import the download script as source, without executing acquisition.
    import importlib.util

    from visualguard.settings import ROOT

    spec = importlib.util.spec_from_file_location("acquire", ROOT / "scripts/acquire_baseline.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    archive = tmp_path / "unsafe.zip"
    with zipfile.ZipFile(archive, "w") as zipped:
        zipped.writestr("../escape.txt", "not allowed")
    with pytest.raises(ValueError, match="Unsafe archive"):
        module.extract_checked(archive, tmp_path / "extracted")
    assert not (tmp_path / "escape.txt").exists()


def test_audit_groups_cross_split_duplicates_before_partitioning(tmp_path):
    import numpy as np
    import yaml

    from visualguard.training.construction import SOURCE_NAMES, audit_prepare

    source = tmp_path / "source"
    source.mkdir()
    (source / "data.yaml").write_text(yaml.safe_dump({"names": SOURCE_NAMES}))
    random = np.random.default_rng(42)
    first = None
    for index in range(6):
        split = ("train", "val", "test")[index // 2]
        folder = source / "images" / split
        folder.mkdir(parents=True, exist_ok=True)
        labels = source / "labels" / split
        labels.mkdir(parents=True, exist_ok=True)
        image = folder / f"{index}.png"
        if index == 5:
            image.write_bytes(first)
        else:
            Image.fromarray(random.integers(0, 255, (32, 32, 3), dtype=np.uint8)).save(image)
            if index == 0:
                first = image.read_bytes()
        (labels / f"{index}.txt").write_text("6 0.5 0.5 0.8 0.8")
    report = audit_prepare(source, tmp_path / "prepared", tmp_path / "reports")
    assert sum(report["prepared_images_by_split"].values()) == 5
    assert len(report["exact_duplicates_removed"]) == 1
    assert len(report["cross_source_split_duplicate_pairs"]) == 1
    assert set(report["prepared_images_by_split"]) == {"train", "val", "test"}


def test_model_report_is_bound_to_checkpoint_hash(tmp_path, monkeypatch):
    import hashlib
    import json
    from pathlib import Path

    import visualguard.model_status as module
    from visualguard.settings import Settings

    weights = tmp_path / "test.pt"
    weights.write_bytes(b"fixture")
    report = tmp_path / "reports/baseline/model_status.json"
    report.parent.mkdir(parents=True)
    report.write_text(json.dumps({"weights_sha256": hashlib.sha256(b"fixture").hexdigest()}))
    monkeypatch.setattr(module, "project_path", lambda path: (tmp_path / Path(path)).resolve())
    settings = Settings(backend="yolo", weights=str(weights), _env_file=None)
    assert module.model_status(settings)["status"] == "experimental"
    assert not module.model_status(settings)["evaluation_matches_configuration"]
    report.write_text(
        json.dumps(
            {
                "weights_sha256": hashlib.sha256(b"fixture").hexdigest(),
                "evaluation": {"imgsz": 320, "fixed_operating_point": {"confidence": 0.4}},
            }
        )
    )
    assert module.model_status(settings)["evaluation_matches_configuration"]
    assert not module.model_status(settings.model_copy(update={"imgsz": 416}))[
        "evaluation_matches_configuration"
    ]
    assert not module.model_status(settings.model_copy(update={"confidence": 0.2}))[
        "evaluation_matches_configuration"
    ]
    profile = tmp_path / "profile.json"
    profile.write_text("original")
    record = json.loads(report.read_text())
    record["evaluation"]["head_profile_sha256"] = hashlib.sha256(profile.read_bytes()).hexdigest()
    record["evaluation"]["fixed_operating_point"]["head_confidence"] = 0.25
    report.write_text(json.dumps(record))
    refined = settings.model_copy(update={"head_profile": str(profile), "head_confidence": 0.25})
    assert module.model_status(refined)["evaluation_matches_configuration"]
    assert not module.model_status(settings)["evaluation_matches_configuration"]
    profile.write_text("modified")
    assert not module.model_status(refined)["evaluation_matches_configuration"]
    report.write_text(json.dumps({"weights_sha256": "different"}))
    assert module.model_status(settings)["status"] == "unverified"
    report.write_text("bad json")
    assert module.model_status(settings)["status"] == "unverified"
