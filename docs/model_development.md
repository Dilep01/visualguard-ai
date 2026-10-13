# Uncovered-head development experiment

The v1 detector produced no uncovered-head detections at confidence 0.4 on the 216-image validation partition. This experiment keeps the original train/validation/test membership and labels unchanged. The original test result remains in [baseline_report.md](baseline_report.md); it is not used to choose the candidate.

## Outcome — 11 October 2026

The 12-epoch v2 run is complete. **V2 was rejected and v1 remains active.** At confidence 0.40, v2 detected none of the 68 annotated uncovered heads. The other classes improved in recall, but the missing-helmet use case did not improve. No new test-set evaluation was run.

At input 416 and confidence 0.40, using the same validation images:

| Class | V1 precision | V2 precision | V1 recall | V2 recall |
|---|---:|---:|---:|---:|
| Person | 82.7% | 81.8% | 80.7% | 88.2% |
| Uncovered head | Undefined: no predictions | Undefined: no predictions | 0.0% | 0.0% |
| Helmet | 91.4% | 92.0% | 70.4% | 76.2% |
| Vest | 90.5% | 87.1% | 65.2% | 78.1% |

V2 validation mAP50 is 68.42% and mAP50–95 is 39.37%; these aggregate values do not resolve the head-class failure. At confidence 0.10, it detected 31 heads with 130 false detections (19.3% precision, 45.6% recall). At 0.20, it detected four with 11 false detections. None of the recorded head thresholds met both the 70% precision and 30% recall development targets.

Candidate weights remain at `models/construction-ppe-cpu-v2.pt`, SHA-256 `5baebc034be5c764cd62414e39b7d337bf21afd990692358d3c3c97eb589e545`. The best checkpoint was selected using aggregate validation fitness; the class-specific deployment checks then rejected it. The resumed six-epoch segment took 1,080.57 seconds including setup and final validation. Historical elapsed time includes a long interruption and is not an active-compute measurement.

Detailed evidence: `reports/evaluation/construction-ppe-v2-validation/`, `reports/experiments/construction-ppe-v2/comparison.json`, and `reports/experiments/construction-ppe-v2/head-review.jpg`. The dashboard's **Latest training experiment** section reports this decision. The prior active model report and `.env` were backed up alongside registration evidence; the active checkpoint/configuration was not switched.

The next model experiment should address the head class directly, with reviewed annotation definitions and workplace examples, then test an unfrozen or class-weighted training setup against validation data. This run does not isolate the cause of the failure, so it cannot establish that any one change will fix it. Independent workplace clips remain necessary for event evaluation.

## Diagnosis

At 320-pixel input, lowering v1 confidence to 0.10 recovered 34 of 68 validation heads, with 175 false head detections (16.3% precision). At 0.05 it recovered 46 heads with 456 false detections. At 0.40 it detected none. Increasing input size to 416 without training still detected none at 0.40. This does not support fixing the problem by lowering the threshold alone.

`scripts/audit_head_geometry.py` checks training and validation label geometry without relabeling images. It found 359 head boxes across 198 training images and 68 across 44 validation images. Of these, 22 training and five validation head boxes have less than 80% of their area inside any annotated person box. One image in each split has head annotations but no person annotations. These are review flags, not proven errors: partial people and close crops can explain some cases.

The generated training review sheet includes person boxes that exclude part of the head, crowd scenes, and non-workplace examples. The app's association rules depend on person/head geometry, so detector metrics alone cannot establish alert performance. No source annotations were changed to make the evaluation easier. Review files are `reports/baseline/head-label-geometry.json` and `head-label-geometry.jpg`.

## Candidate and selection

Candidate v2 starts from v1 weights, with 12 additional epochs, input 416, batch eight, four CPU threads, first ten layers frozen, AdamW at initial learning rate 0.001, and seed 42. It is a new fine-tuning run, not an initial continuation of v1's optimizer state. Within v2, an interruption after six saved epochs required resuming its optimizer/checkpoint state. The unfinished seventh epoch was discarded. The original v2 weights, arguments and CSV were backed up before resuming. Resuming does not guarantee identical random-number replay.

The experiment plan was saved before candidate evaluation in `reports/baseline/candidate-v2-plan.md`. Both checkpoints are compared at input 416 and confidence 0.40, using one-to-one class matching at box IoU 0.5. The development gates require uncovered-head precision at least 70%, recall at least 30%, and no more than a five-percentage-point recall loss for each other class relative to v1 at 416. These are experimental model-selection criteria, not safety acceptance criteria. Failure keeps v1 active.

`scripts/compare_candidate.py` rejects test reports, different splits/configurations, different input sizes, and changed operating thresholds. It saves machine-readable gates, a class comparison, and selected head errors for visual review. Examples are selected by filename within missed-head, false-detection and correct-detection groups; the sheet is not a random sample.

## Reproduce development checks

Use new output directories so saved results remain intact. All paths below are relative to the project root.

```powershell
.\.venv\Scripts\python.exe scripts/audit_head_geometry.py
.\.venv\Scripts\python.exe scripts/evaluate_baseline.py --weights models/construction-ppe-cpu-v1.pt --split val --sweep --imgsz 416 --output reports/evaluation/new-v1-validation
.\.venv\Scripts\python.exe scripts/train.py --data data/processed/construction-ppe-v1/dataset.yaml --model models/construction-ppe-cpu-v1.pt --epochs 12 --imgsz 416 --batch 8 --freeze 10 --threads 4 --name new-candidate
.\.venv\Scripts\python.exe scripts/evaluate_baseline.py --weights reports/training/new-candidate/weights/best.pt --split val --sweep --imgsz 416 --output reports/evaluation/new-candidate-validation
.\.venv\Scripts\python.exe scripts/compare_candidate.py --baseline reports/evaluation/new-v1-validation/detection_metrics.json --candidate reports/evaluation/new-candidate-validation/detection_metrics.json --output reports/experiments/new-candidate
```

For an interrupted run only, first verify its training process has stopped, then use `scripts/resume_training.py --run reports/training/new-candidate`. This preserves a pre-resume backup and continues to the run's original epoch target. Historical CSV time may include sleep/interruption; the resume metadata records the new segment's wall time separately.

`scripts/register_candidate.py --comparison reports/experiments/new-candidate/comparison.json --weights reports/training/new-candidate/weights/best.pt --model-name construction-ppe-cpu-v2` recomputes the gates, verifies the checkpoint hash, preserves the old report/configuration, and records the candidate. It updates `.env` only if the candidate passes. Restart services after activation. This helper currently labels the development experiment as v2; it is not a general model registry.

Independent workplace image/video evaluation is still required after development. A validation improvement must not be presented as a new independent test result or as measured event precision/recall.
