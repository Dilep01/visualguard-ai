# Model card

Current model: v3 with head refinement is active for experimental use. On 216 validation images it detected 21 of 68 uncovered heads, with 7 false detections (75% precision, 30.9% recall). It still missed 47 heads. Real-video accuracy remains unverified. See [head refinement results](head_refinement.md). Earlier v1/v2 results below are historical.


Weights are excluded from Git. The earlier v1 checkpoint detected 0/58 uncovered heads on its exploratory test partition at confidence 0.4; see [baseline_report.md](baseline_report.md) for that historical evaluation. Current v3 results above are validation-only. Tracking quality and real video-event accuracy remain unmeasured.

The v2 candidate completed 12 additional epochs but was rejected by validation checks: uncovered-head recall remained 0/68 at confidence 0.4. It is saved separately and is not the active model. V1's original test scores remain unchanged. See [model_development.md](model_development.md) for the comparison and reviewed failure examples.

The runnable fixture detector thresholds known BGR colors in the synthetic clip. Its fixed confidence of 1.0 indicates a deterministic fixture match; it is not a learned probability. The UI labels the resulting analyses as synthetic.

The Ultralytics adapter loads a local checkpoint and checks its class names. The installed local checkpoint, software versions, parameters, hardware, and dataset hashes are recorded in the baseline report and `reports/baseline/model_status.json`. The API only attaches that report when its checkpoint hash matches the configured weights. Calibration, independent workplace testing, and video error analysis remain pending.

Intended use: a local demonstration of a human-reviewed PPE observation pipeline. Not established for safety certification, enforcement, disciplinary decisions, live emergency response, or identifying individual workers.

Current failure modes include false head detections, missed helmets, small objects, poor lighting, occlusion, ambiguous equipment ownership, tracking ID switches, and incomplete detections that undercount person-time. There is no trained visibility classifier. Vest absence is always unknown. The present prototype has no segmentation, anomaly model, Grad-CAM implementation, or language-model assistant.
