import time
from concurrent.futures import ThreadPoolExecutor

import cv2
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import update

from visualguard.api.main import create_app
from visualguard.ingestion.demo import demo_frame
from visualguard.persistence.store import Store, jobs
from visualguard.settings import Settings
from visualguard.vision.detector import DemoDetector
from visualguard.worker.main import process_job


def test_pipeline_preserves_lower_head_threshold(settings, tmp_path, monkeypatch):
    from visualguard.domain import Box, Detection
    from visualguard.pipeline import analyze

    class Detector:
        name = "threshold-test"
        metadata = {}

        def predict(self, frame):
            return [
                Detection("head", 0.3, Box(20, 20, 40, 40)),
                Detection("person", 0.3, Box(10, 10, 100, 200)),
            ]

    monkeypatch.setattr("visualguard.pipeline.build_detector", lambda *a, **kw: Detector())
    source = tmp_path / "source.jpg"
    cv2.imwrite(str(source), demo_frame(1))
    configured = settings.model_copy(update={"confidence": 0.4, "head_confidence": 0.25})
    result = analyze(source, tmp_path / "output", "image", configured)
    assert [d["label"] for d in result["timeline"][0]["detections"]] == ["head"]
    assert result["events"] == []


def test_demo_queue_worker_artifact_review_and_analytics(settings):
    app = create_app(settings)
    with TestClient(app) as client:
        assert client.get("/health").status_code == 200
        response = client.post("/api/demo")
        assert response.status_code == 202
        job_id = response.json()["id"]
        assert response.json()["status"] == "queued"
        assert "source" not in response.json()
        store = app.state.store
        job = store.claim()
        assert store.claim() is None
        process_job(store, settings, job)
        done = client.get(f"/api/jobs/{job_id}").json()
        assert done["status"] == "completed", done.get("error")
        result = done["result"]
        assert result["synthetic"] is True
        assert result["sampled_frames"] == 32
        assert len(result["events"]) == 1
        assert result["events"][0]["end_seconds"] == pytest.approx(3.75)
        assert result["observed_person_seconds"] == pytest.approx(24)
        assert (
            client.get(f"/api/jobs/{job_id}/media/annotated.mp4").headers["content-type"]
            == "video/mp4"
        )
        assert client.get(f"/api/jobs/{job_id}/media/secret.txt").status_code == 404
        events = client.get("/api/events").json()
        event_id = events[0]["id"]
        assert events[0]["synthetic"] == "yes"
        assert (
            client.patch(
                f"/api/events/{event_id}/review",
                json={"status": "false_alert", "note": "Review persistence test"},
            ).status_code
            == 200
        )
        assert (
            client.get("/api/events?review=false_alert").json()[0]["note"]
            == "Review persistence test"
        )
        analytics = client.get("/api/analytics?synthetic=true").json()
        assert analytics["candidate_events_per_person_minute"] == pytest.approx(2.5)
        assert analytics["review_counts"] == {"false_alert": 1}
        assert client.get("/api/analytics").json()["candidate_events"] == 0
        # Persistence survives opening another connection/store.
        second_store = Store(settings)
        assert second_store.get(job_id)["status"] == "completed"
        second_store.engine.dispose()


def test_unconfigured_real_upload_is_not_sent_to_demo_detector(settings):
    with TestClient(create_app(settings)) as client:
        response = client.post("/api/images", files={"file": ("test.jpg", b"fake", "image/jpeg")})
        assert response.status_code == 503
        assert client.get("/api/jobs").json() == []


def test_image_analysis_and_invalid_media(settings, tmp_path, monkeypatch):
    weights = tmp_path / "fixture.pt"
    weights.write_bytes(b"test only")
    real_settings = Settings(
        storage=settings.storage,
        backend="yolo",
        weights=str(weights),
        max_upload_mb=1,
        _env_file=None,
    )
    monkeypatch.setattr("visualguard.pipeline.build_detector", lambda *args, **kw: DemoDetector())
    _, image = cv2.imencode(".jpg", demo_frame(1))
    with TestClient(create_app(real_settings)) as client:
        response = client.post("/api/images", files={"file": ("image.jpg", image.tobytes())})
        assert response.status_code == 200
        result = response.json()["result"]
        assert result["kind"] == "image"
        assert result["events"] == []
        assert result["observed_person_seconds"] == 0
        assert len(result["timeline"][0]["observations"]) == 3
        assert client.post("/api/images", files={"file": ("fake.jpg", b"bad")}).status_code == 422
        assert client.post("/api/images", files={"file": ("fake.exe", b"bad")}).status_code == 415
        assert (
            client.post("/api/images", files={"file": ("big.jpg", b"0" * 1_100_000)}).status_code
            == 413
        )
        assert client.post("/api/videos", files={"file": ("bad.mp4", b"bad")}).status_code == 422


def test_worker_failure_is_persisted(settings):
    store = Store(settings)
    job_id = store.create("video", settings.storage_path / "missing.mp4", synthetic=True)
    process_job(store, settings, store.claim())
    assert store.get(job_id)["status"] == "failed"
    assert store.get(job_id)["error"]
    store.engine.dispose()


def test_expired_job_can_be_reclaimed_and_old_worker_cannot_commit(settings):
    store = Store(settings)
    job_id = store.create("video", "unused")
    first = store.claim()
    with store.engine.begin() as connection:
        connection.execute(
            update(jobs).where(jobs.c.id == job_id).values(lease_until=time.time() - 1)
        )
    second = store.claim()
    assert first["token"] != second["token"]
    with pytest.raises(RuntimeError, match="lease was lost"):
        store.complete(first, {"events": []}, "unused")
    store.engine.dispose()


def test_concurrent_claim_has_only_one_owner(settings):
    store = Store(settings)
    store.create("video", "unused")
    with ThreadPoolExecutor(max_workers=4) as pool:
        claims = list(pool.map(lambda _: store.claim(), range(4)))
    assert sum(job is not None for job in claims) == 1
    store.engine.dispose()


def test_missing_resources_and_invalid_review(settings):
    with TestClient(create_app(settings)) as client:
        assert client.get("/api/jobs/missing").status_code == 404
        assert (
            client.patch("/api/events/missing/review", json={"status": "confirmed"}).status_code
            == 404
        )
        assert (
            client.patch("/api/events/missing/review", json={"status": "invented"}).status_code
            == 422
        )
