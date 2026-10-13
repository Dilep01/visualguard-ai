# Baseline-stage verification — 8 October 2026

- **39 automated tests passed**, 8.08 seconds, with 826/914 application statements covered (90.37%, displayed as 90%). Tests include proxy duplicate grouping across source splits, normalized-label validation, checkpoint/report hash binding, corrupted-report fallback, safe archive extraction, and fixed-threshold matching without duplicate true positives.
- Ruff lint and formatting checks passed. `pip check` found no broken requirements. The full installed training dependency snapshot is `requirements-vision-lock.txt`.
- Training completed eight epochs and wrote a best checkpoint. The fixed nine-minute training budget resulted in 572.41 seconds total script wall time, including setup and final validation. The `--time-hours` option controls the schedule and can override the requested epoch count.
- The corrected test evaluation processed all 211 images, including `.jpeg`. The canonical report is `reports/evaluation/construction-ppe-v1-complete/detection_metrics.json`; the earlier fixed-threshold report omitted `.jpeg` and is superseded.
- All 1,303 proxy groups were checked: none spans multiple prepared partitions. This verifies the grouping implementation, not true source-scene independence.
- Actual checkpoint inference through FastAPI passed on the traceable validation sample. The returned SHA-256 matched the evaluation report; annotations and evidence were generated. First request took 11.65 seconds including cold imports/model load. This is a single integration timing, not a latency benchmark.
- Browser check passed across all four dashboard pages, synthetic video metadata, saved review notes, the model report, and an actual image upload. No browser page errors were recorded. The app remains running on localhost port 8501.
- Annotated real sample and model/dashboard screenshots were visually inspected. The sample displays detected person, helmet, and vest boxes. It was selected from validation annotations deterministically, not selected based on prediction success.

**Model outcome:** exploratory mAP50 is 66.39%, but uncovered-head recall is **0/58 at confidence 0.4**. This checkpoint is not a working missing-helmet monitor. The dashboard presents it as experimental; an absence of alerts cannot establish compliance. No real video-event accuracy claim is made.

One existing Starlette TestClient deprecation warning remains. PostgreSQL/Docker, source-scene independence, real-video tracking/event metrics, and the optional extensions are not verified by these checks.

All new data, checkpoints, dependency installs, caches, logs, reports, and browser test artifacts for this stage were placed within the project directory. No cloud training, publication, or external image upload was performed.
