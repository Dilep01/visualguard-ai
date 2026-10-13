# Real-data baseline — 8 October 2026

VisualGuard now has a locally trained detector and actual image inference. It does **not** yet have a dependable missing-helmet monitor.

## Data and provenance

The acquired [Construction-PPE dataset](https://docs.ultralytics.com/datasets/detect/construction-ppe/) contains 1,416 images. The downloaded archive includes an AGPL-3.0 license and a class configuration. Archive SHA-256: `bef8dcb599aa4e9d9f5e602cb6fa7143d3c84d7f6a0ff40463d7f2a4c2632ccc`. Acquisition details, original archive, source YAML, and license are retained in `data/external/construction-ppe-source/`.

The four-class projection retains `Person → person`, `no_helmet → head`, `helmet → helmet`, and `vest → vest`. Here, `head` means the source's explicitly annotated uncovered-head class, not all visible heads. The other seven source classes are dropped. Visual inspection confirmed head-sized `no_helmet` boxes, but also found non-workplace head examples and inconsistent person boxes. These are material domain and annotation limitations.

The audit found 35 cross-split pairs within the configured similarity/grouping rule. Exact hashes, original filename families, and perceptual hashes with Hamming distance at most four were grouped before splitting. This produced 1,303 proxy groups and **989 train / 216 validation / 211 test images**. No exact byte duplicates were removed. There were 113 small box-boundary rounding corrections, each within a 0.001 normalized-coordinate tolerance; no image was rejected.

These groups are not verified source scenes. Camera/site/video metadata is unavailable. They reduce obvious leakage but cannot establish independent-scene generalization. The split and image hashes are recorded in `data/processed/construction-ppe-v1/split_manifest.json`. This is an exploratory test partition; an independent workplace set is still needed.

## Training

- Official [YOLO11n base checkpoint](https://github.com/ultralytics/assets/releases/tag/v8.3.0), SHA-256 `0ebbc80d4a7680d14987a577cd21342b65ecfd94632bd9a8da63ae6417644ee1`.
- Ultralytics 8.4.174; PyTorch 2.14.1+cpu; Windows; Ryzen 5 5600U, 16 GB system RAM.
- 320-pixel input, batch eight, four CPU threads, first ten layers frozen, AdamW, initial learning rate 0.001, seed 42.
- Time budget 0.15 hours. Ultralytics' time-based training overrode the requested epoch schedule and completed **eight epochs**, with 572.41 seconds total script wall time including final validation.
- Best checkpoint selected using validation results only. Installed checkpoint SHA-256: `bd931ce4357b3649f6e3d4efc590b4f7f997c07d9b6b67454367d892936393ea`.

The checkpoint is `models/construction-ppe-cpu-v1.pt`. Training logs, CSV history, arguments, and run metadata are in `reports/training/construction-ppe-cpu-v1/`.

## Test findings

The final evaluation uses all 211 `.jpg` and `.jpeg` test images, 320-pixel input, and the saved checkpoint. Overall **mAP50 = 0.6639**, **mAP50–95 = 0.3633**. These are detection metrics, not percentages of safe workers or correct alerts.

At confidence 0.4 and one-to-one class-aware matching with box IoU at least 0.5:

| Class | True positives | False positives | Missed objects | Precision | Recall |
|---|---:|---:|---:|---:|---:|
| Person | 264 | 48 | 56 | 84.62% | 82.50% |
| Uncovered head | 0 | 0 | 58 | Undefined: no predictions | 0.00% |
| Helmet | 202 | 18 | 70 | 91.82% | 74.26% |
| Vest | 175 | 28 | 83 | 86.21% | 67.83% |

The uncovered-head result prevents any claim that this version detects missing helmets reliably. Absence of an alert is not evidence of compliance. Lowering confidence was not used to improve the reported test result. Future threshold tuning must use validation data, with an independent final evaluation after model development.

Canonical metrics and per-image predictions: `reports/evaluation/construction-ppe-v1-complete/`. The earlier `construction-ppe-v1/` evaluation is superseded: its fixed-threshold loop omitted `.jpeg` images, although the upstream mAP evaluation included them. The corrected run uses all formats, with unchanged model and thresholds.

## What is usable now

Local experimental image inspection, detection boxes/confidences, recorded evidence, synthetic workflow testing, and transparent model/data reporting. The supplied real sample is selected deterministically from validation labels, not chosen for successful predictions. Image analyses create no temporal events or person-time.

Real video-event accuracy, tracking quality, and false alerts per video hour have not been measured. Do not use the current model for workplace safety decisions. The next model-development work should audit and improve uncovered-head annotations/domain coverage, train against the existing validation partition, and acquire a separately reviewed workplace image/video test set.

Reproduction commands:

```powershell
.\.venv\Scripts\python.exe scripts/acquire_baseline.py
.\.venv\Scripts\python.exe scripts/prepare_construction_ppe.py --output data/processed/construction-ppe-v2
.\.venv\Scripts\python.exe scripts/train.py --data data/processed/construction-ppe-v1/dataset.yaml --model models/yolo11n.pt --epochs 5 --time-hours 0.15 --name new-run
.\.venv\Scripts\python.exe scripts/evaluate_baseline.py --weights reports/training/new-run/weights/best.pt --output reports/evaluation/new-run
```

Use new output directories to preserve existing evidence. Acquisition uses Windows `curl.exe` because Python's HTTP transfer stalled in this environment. No cloud training, publishing, or external uploads were performed.
