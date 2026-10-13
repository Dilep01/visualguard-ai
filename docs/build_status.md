# Build status

Current model: v3 with head refinement is active for experimental use. On 216 validation images it detected 21 of 68 uncovered heads, with 7 false detections (75% precision, 30.9% recall). It still missed 47 heads. Real-video accuracy remains unverified. See [head refinement results](head_refinement.md). Earlier v1/v2 results below are historical.


## Implemented

Local package/environment, generated demo, media validation, YOLO inference, tracking, conservative PPE matching, temporal rules, annotated media, database queue/worker, API, dashboard, reviews, analytics, dataset auditing, training/evaluation, and tests. The real-data stage added an acquired dataset, a trained CPU checkpoint, 211-image exploratory evaluation, a checkpoint-bound model report, and actual image inference. See [baseline_report.md](baseline_report.md).

The v2 development experiment is complete: 12 additional epochs and 216-image validation, followed by rejection because uncovered-head recall remained 0/68 at confidence 0.4. V1 remained configured until the subsequent v3 refinement. The app now shows validation/test scope, evaluation-setting mismatches, and the candidate comparison. Candidate registration checks hashes and selection criteria and preserves rollback files. See [model_development.md](model_development.md).

## Not established

Missing-helmet monitoring is not ready: current v3 missed 47/68 validation heads. Independent-scene generalization and real video-event accuracy remain unverified. PostgreSQL/Docker has not been run. Zone polygons and optional explainability/anomaly/VLM extensions are not implemented.

## Delivery verification

See `reports/verification.md` for the original application checks and `reports/baseline/verification-v3.md` for the latest development verification. Machine-readable test and coverage reports are stored in `reports/`.

## Environment note

During the initial Python version probe, the system `python` shim invoked Python Install Manager and reported that it updated itself to 26.3. The invocation was stopped. Subsequent Python commands used the direct existing Python 3.11 executable and the project-local `.venv`. No global dependency installations or Docker operations were performed for the application.
