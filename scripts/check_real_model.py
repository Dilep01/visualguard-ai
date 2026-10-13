"""Exercise actual trained-checkpoint inference through the image API, without a stub."""

import json
import time

from fastapi.testclient import TestClient

from visualguard.api.main import create_app
from visualguard.settings import Settings, project_path

if __name__ == "__main__":
    settings = Settings(storage="runtime/real-model-check")
    app = create_app(settings)
    sample = project_path("data/demo/real-ppe-sample.jpg")
    with TestClient(app) as client:
        model = client.get("/api/model").json()
        assert model["status"] == "experimental"
        started = time.perf_counter()
        response = client.post("/api/images", files={"file": (sample.name, sample.read_bytes())})
        assert response.status_code == 200, response.text
        job = response.json()
        result = job["result"]
        assert not result["synthetic"]
        assert result["model_metadata"]["checkpoint_sha256"] == model["report"]["weights_sha256"]
        assert result["timeline"][0]["detections"]
        artifact = client.get(f"/api/jobs/{job['id']}/media/annotated.jpg")
        assert artifact.status_code == 200
        project_path("reports/baseline/real-image-annotated.jpg").write_bytes(artifact.content)
        record = {
            "status": "passed",
            "job_id": job["id"],
            "api_seconds": time.perf_counter() - started,
            "synthetic": False,
            "detector": result["model"],
            "detections": result["timeline"][0]["detections"],
            "observations": result["timeline"][0]["observations"],
            "scope": "Actual checkpoint/API plumbing check on a validation image, "
            "not new accuracy evidence",
        }
        project_path("reports/baseline/real-inference-check.json").write_text(
            json.dumps(record, indent=2)
        )
        print(
            json.dumps(
                {
                    key: value
                    for key, value in record.items()
                    if key not in {"detections", "observations"}
                },
                indent=2,
            )
        )
