import hashlib
import json
import runpy
import sys
from copy import deepcopy
from types import SimpleNamespace

import pytest

from visualguard.settings import ROOT


def prepare(tmp_path, monkeypatch, passed):
    import visualguard.settings as settings

    script = ROOT / "scripts/register_candidate.py"
    monkeypatch.setattr(settings, "ROOT", tmp_path)
    monkeypatch.setattr(settings, "Settings", lambda: SimpleNamespace(weights="models/base.pt"))
    models = tmp_path / "models"
    models.mkdir()
    (models / "base.pt").write_bytes(b"baseline fixture")
    (models / "candidate.pt").write_bytes(b"candidate fixture")
    baseline = {
        "split": "val",
        "split_manifest_sha256": "split",
        "dataset_config_sha256": "config",
        "images": 10,
        "imgsz": 416,
        "weights_sha256": hashlib.sha256(b"baseline fixture").hexdigest(),
        "fixed_operating_point": {
            "confidence": 0.4,
            "iou": 0.5,
            "per_class": {
                name: {"tp": 8, "fp": 2, "fn": 2, "precision": 0.8, "recall": 0.8}
                for name in ("person", "head", "helmet", "vest")
            },
        },
    }
    candidate = deepcopy(baseline)
    candidate["weights_sha256"] = hashlib.sha256(b"candidate fixture").hexdigest()
    if not passed:
        candidate["fixed_operating_point"]["per_class"]["head"] = {
            "tp": 0,
            "fp": 0,
            "fn": 10,
            "precision": None,
            "recall": 0,
        }
    folder = tmp_path / "experiment"
    folder.mkdir()
    for name, report in (("baseline", baseline), ("candidate", candidate)):
        (folder / f"{name}.json").write_text(json.dumps(report))
    (folder / "comparison.json").write_text(
        json.dumps(
            {
                "baseline_report": "experiment/baseline.json",
                "candidate_report": "experiment/candidate.json",
                "eligible_for_experimental_use": True,  # Must recompute, never trust this flag.
            }
        )
    )
    status = {"weights_sha256": baseline["weights_sha256"], "sample": "data/sample.jpg"}
    status_path = tmp_path / "reports/baseline/model_status.json"
    status_path.parent.mkdir(parents=True)
    status_path.write_text(json.dumps(status))
    env = tmp_path / ".env"
    env.write_text("VG_WEIGHTS=models/base.pt\nVG_IMGSZ=320\nVG_MAX_UPLOAD_MB=20\n")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            str(script),
            "--comparison",
            "experiment/comparison.json",
            "--weights",
            "models/candidate.pt",
            "--model-name",
            "v2-fixture",
        ],
    )
    return script, env, status_path


def test_failed_candidate_keeps_active_model_even_when_saved_flag_says_pass(tmp_path, monkeypatch):
    script, env, status_path = prepare(tmp_path, monkeypatch, passed=False)
    original_env = env.read_text()
    runpy.run_path(str(script), run_name="__main__")
    assert env.read_text() == original_env
    status = json.loads(status_path.read_text())
    assert not status["latest_experiment"]["comparison"]["eligible_for_experimental_use"]
    assert status["weights_sha256"] == hashlib.sha256(b"baseline fixture").hexdigest()


def test_passing_candidate_records_settings_and_preserves_rollback(tmp_path, monkeypatch):
    script, env, status_path = prepare(tmp_path, monkeypatch, passed=True)
    original_env = env.read_text()
    runpy.run_path(str(script), run_name="__main__")
    assert "VG_WEIGHTS=models/v2-fixture.pt" in env.read_text()
    assert "VG_IMGSZ=416" in env.read_text()
    assert "VG_MAX_UPLOAD_MB=20" in env.read_text()
    assert (tmp_path / "experiment/registration-backup/.env").read_text() == original_env
    status = json.loads(status_path.read_text())
    assert status["evaluation"]["split"] == "val"
    assert status["missing_helmet_monitor_ready"] is False


def test_unmatched_checkpoint_cannot_change_active_model(tmp_path, monkeypatch):
    script, env, status_path = prepare(tmp_path, monkeypatch, passed=True)
    before = env.read_bytes(), status_path.read_bytes()
    (tmp_path / "models/candidate.pt").write_bytes(b"different model")
    with pytest.raises(SystemExit):
        runpy.run_path(str(script), run_name="__main__")
    assert (env.read_bytes(), status_path.read_bytes()) == before
    assert not (tmp_path / "experiment/registration-backup").exists()


def test_changed_head_profile_cannot_be_registered(tmp_path, monkeypatch):
    script, env, status_path = prepare(tmp_path, monkeypatch, passed=True)
    candidate_path = tmp_path / "experiment/candidate.json"
    candidate = json.loads(candidate_path.read_text())
    candidate.update(head_profile="profile.json", head_profile_sha256="wrong")
    (tmp_path / "profile.json").write_text("{}")
    candidate_path.write_text(json.dumps(candidate))
    before = env.read_bytes(), status_path.read_bytes()
    with pytest.raises(SystemExit):
        runpy.run_path(str(script), run_name="__main__")
    assert (env.read_bytes(), status_path.read_bytes()) == before
