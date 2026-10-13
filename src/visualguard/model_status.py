import hashlib
import json
from functools import lru_cache

from visualguard.settings import Settings, project_path


@lru_cache(maxsize=8)
def _digest(path: str, modified: int, size: int) -> str:
    return hashlib.sha256(project_path(path).read_bytes()).hexdigest()


def model_status(settings: Settings) -> dict:
    weights = project_path(settings.weights)
    report = project_path("reports/baseline/model_status.json")
    status = {
        "backend": settings.backend,
        "weights_present": weights.is_file(),
        "status": "unconfigured",
        "report": None,
        "configuration": {"imgsz": settings.imgsz, "confidence": settings.confidence},
        "evaluation_matches_configuration": False,
    }
    if not weights.is_file() or settings.backend != "yolo":
        return status
    status["status"] = "unverified"
    if report.is_file():
        try:
            recorded = json.loads(report.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return status
        if not isinstance(recorded, dict):
            return status
        stat = weights.stat()
        digest = _digest(str(weights), stat.st_mtime_ns, stat.st_size)
        if recorded.get("weights_sha256") == digest:
            status.update(status="experimental", report=recorded)
            evaluation = recorded.get("evaluation", {})
            point = evaluation.get("fixed_operating_point", {})
            profile_matches = not settings.head_profile and not evaluation.get(
                "head_profile_sha256"
            )
            if settings.head_profile and project_path(settings.head_profile).is_file():
                profile_matches = hashlib.sha256(
                    project_path(settings.head_profile).read_bytes()
                ).hexdigest() == evaluation.get("head_profile_sha256")
            status["evaluation_matches_configuration"] = (
                evaluation.get("imgsz") == settings.imgsz
                and evaluation.get("fixed_operating_point", {}).get("confidence")
                == settings.confidence
                and point.get("head_confidence", point.get("confidence"))
                == settings.detection_threshold("head")
                and profile_matches
            )
    return status
