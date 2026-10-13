# Verification — 7 October 2026

The local synthetic workflow is runnable. Real PPE model accuracy is not established.

## Executed checks

- Python 3.11.9, project-local virtual environment; installed dependency versions captured in `requirements-lock.txt`.
- `pytest --cov=visualguard --cov-report=term --cov-report=xml:reports/coverage.xml --junitxml=reports/tests.xml`: **27 passed**, 8.45 seconds; 648 of 721 application statements covered (**89.88%, rounded to 90%**).
- Ruff lint: passed. Ruff formatting: all 46 Python files already formatted.
- `pip check`: no broken requirements.
- CLI demo: completed; one possible-missing-helmet event, 32 sampled frames, 24 observed person-seconds. End of supporting evidence at 3.75 seconds. Output: `runtime/cli-demo/result.json` and `runtime/cli-demo/annotated.mp4`.
- Live API, separate video worker, and dashboard: started successfully on localhost. Project stop/start was exercised, including Windows venv child-process cleanup.
- Headless Chrome browser check: all four dashboard pages visited, generated video metadata loaded, review note persisted and read back through the API, zero browser page errors. Details: `browser-check.json`.
- Visual inspection: annotated preview and dashboard screenshots. Evidence: `dashboard-analysis.png`, `dashboard-review.png`, `dashboard-analytics.png`.
- Synthetic workflow timing: three eight-second-clip runs took 0.9138, 0.7811, and 0.7922 seconds; median 0.7922 seconds. Includes fixture processing and encoding, **not ML inference**. Platform details are in `synthetic_benchmark.json`.

Integration tests cover temporal continuity and unknown resets, equipment ambiguity and border crops, track expiry and unique assignment, API file rejection, image observations without temporal events, queue ownership, expired-lease recovery, stale-owner commit rejection, failed-job persistence, generated media delivery, review persistence, and synthetic/real analytics separation. Dataset checks cover scene-level splitting, exact duplicates, corrupt images, invalid boxes, and project path confinement.

## Limits and observations

- API image tests use a stub detector. The actual YOLO adapter, training, evaluation, and model checkpoint loading are not exercised because no dataset or weights were supplied.
- The Windows sandbox blocked the asyncio localhost socket pair used by TestClient. Integration tests passed when run with permitted localhost access. Unit tests also passed inside the sandbox.
- Starlette emitted one deprecation warning for its current `httpx` TestClient transport. It did not cause a failure. The pinned tested dependency set is retained; a future dependency update should review the recommended transport migration.
- The browser test initially targeted the wrong event after multiple synthetic runs; it now scopes actions to the exact job and waits for persisted results. Navigation uses the visible Streamlit labels.
- PostgreSQL advisory-lock initialization and Docker Compose are supplied but untested. Docker was not run because engine-managed files would be outside the requested project directory.
- Optional camera zones, Grad-CAM, anomaly modeling, and VLM assistance remain future work.
- The initial system Python shim reported a Python Install Manager self-update to 26.3 before it was stopped. Subsequent application tooling, dependencies, temporary files, caches, browser profile, and outputs were placed inside this project.

The app remains running at `http://127.0.0.1:8501`, with API docs on `http://127.0.0.1:8000/docs`. Stop it with `scripts/start.ps1 -Stop` from the project directory. Synthetic jobs used during browser verification remain available in its history.
