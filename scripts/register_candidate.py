"""Record a candidate experiment and activate it only when its validation gates pass."""

import argparse
import hashlib
import json
import re
import shutil

from visualguard.settings import Settings, project_path
from visualguard.training.comparison import compare_candidate

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--comparison", required=True)
    parser.add_argument("--weights", required=True)
    parser.add_argument("--model-name", default="construction-ppe-cpu-v2")
    parser.add_argument("--candidate-label", default="Candidate v2")
    args = parser.parse_args()
    if not re.fullmatch(r"[a-z0-9][a-z0-9-]{1,63}", args.model_name):
        parser.error("Use a lowercase model filename containing letters, numbers and hyphens")
    comparison_path = project_path(args.comparison)
    recorded = json.loads(comparison_path.read_text())
    baseline = json.loads(project_path(recorded["baseline_report"]).read_text())
    candidate = json.loads(project_path(recorded["candidate_report"]).read_text())
    comparison = compare_candidate(baseline, candidate)
    profile = candidate.get("head_profile", "")
    if profile and hashlib.sha256(project_path(profile).read_bytes()).hexdigest() != candidate.get(
        "head_profile_sha256"
    ):
        parser.error("Head profile hash does not match the evaluated profile")
    checkpoint = project_path(args.weights)
    digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    if digest != comparison["candidate_sha256"]:
        parser.error("Checkpoint hash does not match the evaluated candidate")
    status_path = project_path("reports/baseline/model_status.json")
    status = json.loads(status_path.read_text())
    settings = Settings()
    active_digest = hashlib.sha256(project_path(settings.weights).read_bytes()).hexdigest()
    if active_digest != comparison["baseline_sha256"] or status["weights_sha256"] != active_digest:
        parser.error("The configured model must match the evaluated baseline before registration")
    target = project_path(f"models/{args.model_name}.pt")
    if target.exists() and hashlib.sha256(target.read_bytes()).hexdigest() != digest:
        parser.error("Refusing to replace an existing different checkpoint")
    backup = comparison_path.parent / "registration-backup"
    backup.mkdir(exist_ok=False)
    shutil.copy2(status_path, backup / "model_status.json")
    env = project_path(".env")
    shutil.copy2(env, backup / ".env")
    shutil.copy2(checkpoint, target)
    heads = candidate["fixed_operating_point"]["per_class"]["head"]
    eligible = comparison["eligible_for_experimental_use"]
    summary = (
        f"{args.candidate_label} detected {heads['tp']} of "
        f"{heads['tp'] + heads['fn']} validation heads "
        f"with {heads['fp']} false head detections "
        f"at head confidence {comparison['head_confidence']}. "
        + (
            "It passed the development checks and is active for experimental use."
            if eligible
            else "It failed the development checks; the previous model remains active."
        )
    )
    if eligible:
        status = {
            "name": args.candidate_label,
            "weights_sha256": digest,
            "weights": str(target.relative_to(project_path("."))).replace("\\", "/"),
            "summary": summary,
            "evaluation": candidate,
            "missing_helmet_monitor_ready": False,
            "sample": status["sample"],
            "limitations": [
                f"Missed {heads['fn']} of {heads['tp'] + heads['fn']} "
                "uncovered heads in validation.",
                "Validation images were used during model development; "
                "independent testing remains pending.",
                "Image detection improvements do not validate tracking or video helmet alerts.",
                "Proxy image grouping does not establish source-scene independence.",
                "Training labels include non-workplace head examples "
                "and inconsistent person boxes.",
                "Vest absence remains unknown; absence of an alert does not establish compliance.",
            ],
        }
        updates = {
            "VG_BACKEND": "yolo",
            "VG_WEIGHTS": status["weights"],
            "VG_IMGSZ": str(candidate["imgsz"]),
            "VG_CONFIDENCE": "0.4",
            "VG_HEAD_CONFIDENCE": str(comparison["head_confidence"]),
            "VG_HEAD_PROFILE": profile,
        }
        lines = []
        for line in env.read_text().splitlines():
            key = line.partition("=")[0].strip()
            lines.append(f"{key}={updates.pop(key)}" if key in updates else line)
        lines.extend(f"{key}={value}" for key, value in updates.items())
        env.write_text("\n".join(lines) + "\n", encoding="utf-8")
    status["latest_experiment"] = {
        "summary": summary,
        "comparison": comparison,
        "candidate_label": args.candidate_label,
    }
    status_path.write_text(json.dumps(status, indent=2), encoding="utf-8")
    (comparison_path.parent / "registration.json").write_text(
        json.dumps(
            {
                "candidate_weights": str(target),
                "activated": eligible,
                "summary": summary,
                "candidate_sha256": digest,
                "previous_sha256": active_digest,
                "restart_required": eligible,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(summary)
