import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, HTTPException, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from visualguard.analytics.metrics import summarize
from visualguard.ingestion.demo import create_demo
from visualguard.ingestion.media import read_image, video_info
from visualguard.model_status import model_status
from visualguard.persistence.store import Store
from visualguard.settings import Settings, local_caches, project_path
from visualguard.worker.main import process_job


class ReviewRequest(BaseModel):
    status: Literal["pending", "confirmed", "false_alert", "uncertain"]
    note: str = Field(default="", max_length=2000)


def public_job(job):
    return {
        key: value
        for key, value in job.items()
        if key not in {"source", "token", "lease_until", "output"}
    }


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or Settings()

    @asynccontextmanager
    async def lifespan(app):
        local_caches()
        app.state.store = Store(settings)
        yield
        app.state.store.engine.dispose()

    app = FastAPI(
        title="VisualGuard AI",
        version="0.1.0",
        lifespan=lifespan,
        description="Local PPE review prototype. Demo scenes are synthetic.",
    )

    def get_job(job_id):
        job = app.state.store.get(job_id)
        if job is None:
            raise HTTPException(404, "Job not found")
        return job

    def save_upload(file: UploadFile, kind: str) -> Path:
        if settings.backend != "yolo" or not project_path(settings.weights).is_file():
            raise HTTPException(
                503,
                "Real uploads need VG_BACKEND=yolo and trained PPE weights. "
                "The synthetic demo is available now.",
            )
        suffix = Path(file.filename or "").suffix.lower()
        allowed = {".jpg", ".jpeg", ".png"} if kind == "image" else {".mp4", ".avi", ".mov"}
        if suffix not in allowed:
            raise HTTPException(415, "Unsupported file extension")
        folder = settings.storage_path / "uploads"
        folder.mkdir(exist_ok=True)
        target = folder / f"{uuid.uuid4().hex}{suffix}"
        total = 0
        try:
            with target.open("wb") as output:
                while chunk := file.file.read(1024 * 1024):
                    total += len(chunk)
                    if total > settings.max_upload_mb * 1024 * 1024:
                        raise HTTPException(413, "Upload exceeds configured byte limit")
                    output.write(chunk)
            if total == 0:
                raise ValueError("File is empty")
            read_image(target, settings) if kind == "image" else video_info(target, settings)
            return target
        except Exception as exc:
            target.unlink(missing_ok=True)
            if isinstance(exc, HTTPException):
                raise
            raise HTTPException(422, str(exc)) from exc
        finally:
            file.file.close()

    @app.get("/health")
    def health():
        return {
            "status": "ok",
            "backend": settings.backend,
            "weights_present": project_path(settings.weights).is_file(),
            "model_validated": False,
            "version": "0.1.0",
        }

    @app.post("/api/demo", status_code=202)
    def demo():
        folder = settings.storage_path / "uploads"
        folder.mkdir(exist_ok=True)
        source = folder / f"demo-{uuid.uuid4().hex}.mp4"
        create_demo(source)
        job_id = app.state.store.create("video", source, synthetic=True)
        return public_job(get_job(job_id))

    @app.get("/api/model")
    def model_report():
        return model_status(settings)

    @app.post("/api/images")
    def image(file: UploadFile):
        source = save_upload(file, "image")
        job_id = app.state.store.create("image", source)
        job = app.state.store.claim(job_id)
        process_job(app.state.store, settings, job)
        result = get_job(job_id)
        if result["status"] == "failed":
            raise HTTPException(422, result["error"])
        return public_job(result)

    @app.post("/api/videos", status_code=202)
    def video(file: UploadFile):
        source = save_upload(file, "video")
        job_id = app.state.store.create("video", source)
        return public_job(get_job(job_id))

    @app.get("/api/jobs")
    def list_jobs():
        return app.state.store.list_jobs()

    @app.get("/api/jobs/{job_id}")
    def job(job_id: str):
        return public_job(get_job(job_id))

    @app.get("/api/jobs/{job_id}/media/{name}")
    def media(job_id: str, name: str):
        if name not in {"preview.jpg", "annotated.jpg", "annotated.mp4"}:
            raise HTTPException(404, "Media not found")
        job = get_job(job_id)
        if job["status"] != "completed":
            raise HTTPException(409, "Job is not completed")
        path = project_path(job["output"]) / name
        if not path.is_file():
            raise HTTPException(404, "Media not found")
        return FileResponse(path, media_type="video/mp4" if name.endswith("mp4") else "image/jpeg")

    @app.get("/api/events")
    def list_events(job_id: str | None = None, review: str | None = None):
        return app.state.store.list_events(job_id, review)

    @app.patch("/api/events/{event_id}/review")
    def review_event(event_id: str, body: ReviewRequest):
        if not app.state.store.review(event_id, body.status, body.note):
            raise HTTPException(404, "Event not found")
        return {"id": event_id, "review": body.status, "note": body.note}

    @app.get("/api/analytics")
    def analytics(synthetic: bool = False):
        return summarize(app.state.store, synthetic)

    return app


app = create_app()
