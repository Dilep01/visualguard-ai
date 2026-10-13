"""Exercise preselected validation images through the running local API."""

import json

import httpx

from visualguard.settings import project_path

if __name__ == "__main__":
    results = []
    with httpx.Client(base_url="http://127.0.0.1:8000", timeout=120) as client:
        status = client.get("/api/model").raise_for_status().json()
        assert status["evaluation_matches_configuration"]
        for path in sorted(project_path("data/demo/head-validation").glob("*.jpg")):
            with path.open("rb") as file:
                response = client.post(
                    "/api/images", files={"file": (path.name, file, "image/jpeg")}
                )
            response.raise_for_status()
            results.append({"image": path.name, "result": response.json()})
    output = project_path("reports/baseline/head-upload-check.json")
    output.write_text(
        json.dumps(
            {
                "scope": "Preselected validation image integration checks; "
                "not independent or video accuracy",
                "results": results,
            },
            indent=2,
        )
    )
    print(f"Completed {len(results)} image uploads; saved {output}")
