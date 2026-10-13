import os
from pathlib import Path

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT = Path(os.environ.get("VG_PROJECT_ROOT", Path(__file__).resolve().parents[2])).resolve()


def project_path(value: str | Path) -> Path:
    path = (ROOT / value).resolve()
    if not path.is_relative_to(ROOT):
        raise ValueError("Paths must stay inside the VisualGuard project directory")
    return path


def local_caches() -> None:
    """Constrain caches for optional libraries before importing them."""
    for name, directory in {
        "YOLO_CONFIG_DIR": ".cache/ultralytics",
        "TORCH_HOME": ".cache/torch",
        "HF_HOME": ".cache/huggingface",
        "MPLCONFIGDIR": ".cache/matplotlib",
        "XDG_CACHE_HOME": ".cache",
        "TEMP": ".cache/tmp",
        "TMP": ".cache/tmp",
    }.items():
        path = project_path(directory)
        path.mkdir(parents=True, exist_ok=True)
        os.environ[name] = str(path)
    os.environ["YOLO_AUTOINSTALL"] = "false"
    os.environ["YOLO_OFFLINE"] = "true"


def configure_vision() -> None:
    local_caches()
    from ultralytics import settings

    settings.update(
        {
            "datasets_dir": str(project_path("data/external")),
            "weights_dir": str(project_path("models")),
            "runs_dir": str(project_path("reports/training")),
            "sync": False,
            "tensorboard": False,
        }
    )


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="VG_", env_file=ROOT / ".env", extra="ignore")
    backend: str = "demo"
    weights: str = "models/ppe.pt"
    device: str = "cpu"
    imgsz: int = Field(default=320, ge=64, le=1536)
    confidence: float = Field(default=0.4, gt=0, le=1)
    head_confidence: float | None = Field(default=None, gt=0, le=1)
    head_profile: str = ""
    sample_fps: float = Field(default=4, ge=1, le=30)
    min_evidence_frames: int = Field(default=3, ge=2)
    min_evidence_seconds: float = Field(default=0.5, gt=0)
    max_gap_seconds: float = Field(default=0.75, gt=0)
    max_video_seconds: float = Field(default=300, gt=0, le=3600)
    max_upload_mb: int = Field(default=100, ge=1, le=500)
    max_pixels: int = Field(default=16_000_000, ge=1000)
    database_url: str = ""
    storage: str = "runtime"

    @model_validator(mode="after")
    def valid_configuration(self):
        if self.backend not in {"demo", "yolo"}:
            raise ValueError("VG_BACKEND must be demo or yolo")
        project_path(self.storage)
        project_path(self.weights)
        if self.head_profile:
            project_path(self.head_profile)
        if self.database_url and not self.database_url.startswith("postgresql+psycopg://"):
            raise ValueError("Use PostgreSQL URL or leave blank for project-local SQLite")
        if self.max_gap_seconds < 1 / self.sample_fps:
            raise ValueError("max_gap_seconds must span at least one sampling interval")
        return self

    @property
    def storage_path(self) -> Path:
        path = project_path(self.storage)
        path.mkdir(parents=True, exist_ok=True)
        return path

    def detection_threshold(self, label: str) -> float:
        return (
            self.head_confidence
            if label == "head" and self.head_confidence is not None
            else self.confidence
        )
