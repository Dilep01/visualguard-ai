# V3 head detection development

The active experimental model is `models/construction-ppe-cpu-v3-profile.pt` with the
profile at `reports/training/head-adjustment-v3/inference_profile.json`. It uses
320-pixel inputs, head confidence 0.25, and confidence 0.4 for other classes.

On 216 validation images, the actual application detector produced:

| Class | Correct detections | False detections | Missed objects | Precision | Recall |
|---|---:|---:|---:|---:|---:|
| Uncovered head | 21 | 7 | 47 | 75.0% | 30.9% |
| Person | 256 | 74 | 40 | 77.6% | 86.5% |
| Helmet | 185 | 13 | 55 | 93.4% | 77.1% |
| Vest | 179 | 22 | 54 | 89.1% | 76.8% |

At the same head threshold, cached v1 validation predictions detected 1 of 68
heads with no false head detections. V1 at its original 0.4 threshold detected
none. The refined v3 passes the development gates of at least 70% head precision,
30% head recall, and no more than a five-point recall loss in other classes.
These gates are development criteria, not a safety standard. Person precision
fell from 85.6% to 77.6%; person false detections increased from 41 to 74.

## Training and selection

V3 continued from v2 for 20 CPU epochs with the backbone unfrozen, class-frequency
weighting, AdamW learning rate 0.0005, and no mosaic augmentation. Epoch 12 was
selected by validation head AP50. The checkpoint SHA256 is
`ae41881cf271f5d9a335b95fbb4b050eb19c51faaff3f2375dd2c6a8ad00074f`.

Raw v3 at confidence 0.4 found only 4 of 68 validation heads. Increasing input
size to 640 or adding upper-person crop inference did not solve this. Reviewing
errors revealed inconsistent source head box geometry and overlapping duplicate
predictions. A four-parameter box adjustment was fitted only on 336 matched
predictions from 198 training images. Overlapping head boxes are then fused at
IoU 0.5 using confidence-weighted coordinates, retaining the highest original
confidence. Scores are never increased. Simple duplicate suppression reached
20/68 recall and failed; fusion reached 21/68. These choices used validation
feedback and must be treated as model development.

The final report runs the actual app detector one image at a time with fixed
square preprocessing. Baseline comparison rescored cached original validation
predictions; preprocessing differs between those pipelines. No new test-set
evaluation was used to select v3. AP was not recomputed for the refined boxes;
the dashboard displays it as not measured. Raw model AP is separately labeled.

## Limits and evidence

47 of 68 uncovered heads were missed. These results do not establish reliable
missing-helmet monitoring. Validation images influenced model and profile
selection; scene independence is not established. The user has no workplace
videos available, so real tracking and video alert accuracy remain unverified.
An image detection is not a temporal alert. Unknown/ambiguous ownership remains
unknown, and vest absence never implies a violation.

- Final app-path results: `reports/evaluation/construction-ppe-v3-profile-validation/`
- Comparison and error review: `reports/experiments/construction-ppe-v3-profile/`
- Training history: `reports/training/construction-ppe-cpu-v3/`
- Training-only geometry fit: `reports/training/head-adjustment-v3/`
- Previous configuration: `reports/experiments/construction-ppe-v3-profile/registration-backup/`

Registration checks checkpoint and profile hashes. The dashboard's evaluation
match also checks image size, per-class thresholds, and profile hash. Previous
v1 exploratory test results and rejected v2 evidence remain historical records.
