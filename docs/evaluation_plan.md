# Evaluation plan

## Detection

Establish label definitions before downloading or relabeling data. Split by original scene/video and inspect cross-split near-duplicates. Tune confidence thresholds only on validation data. Freeze the test split and record hashes. Report per-class precision, recall, AP50, AP50–95, sample counts, input resolution, detector version, and hardware. A second architecture is justified only by a measured accuracy/speed tradeoff.

The general `evaluate.py` entry point exports detector metrics and hashes. The newer `evaluate_baseline.py` additionally records fixed-confidence precision/recall with one-to-one matching and supports the complete baseline audit. The first trained checkpoint and its 211-image exploratory results are recorded in [baseline_report.md](baseline_report.md). These results do not validate temporal alerts or independent-scene generalization.

Development now supports `evaluate_baseline.py --split val --sweep`; confidence sweeps are rejected for test data. Candidate selection uses matching validation reports and predeclared checks. See [model_development.md](model_development.md) for the uncovered-head experiment and label review. The dashboard distinguishes validation from test results and warns if recorded evaluation settings differ from active inference settings.

## Video events

Create a manually reviewed clip set with person-local IDs, visible/unknown intervals, helmet and vest annotations, event intervals, and reasons for uncertainty. Keep clips from the same site/session together. Report event precision and recall with a predeclared temporal overlap threshold, false alerts per video hour, missed visible-head incidents, duplicate events, identity switches, and unknown fractions. Match predicted and reference events one-to-one; report ambiguous cases separately.

A scene with no reference events is still necessary for measuring false alerts. Track performance separately for partial occlusion, small objects, overlapping people, poor lighting, camera motion, and cuts. Do not infer a false-positive rate from model confidence or synthetic fixtures. Human review feedback is operational metadata until independently adjudicated.

## System behavior and timing

Test decode errors, empty and oversized uploads, unsupported file types, one-frame false detections, unknown gaps, reappearance after track expiry, exclusive queue claims, stale-worker completion, media delivery, and review persistence. Record image API latency and video processing speed separately, including CPU/GPU, input size, sampled FPS, codec, and whether model load/encoding are included.

`benchmark.py` measures the synthetic workflow including encoding; it does not benchmark YOLO inference. API image integration tests use a stub detector. The test suite must not be cited as evidence of trained-model performance.
