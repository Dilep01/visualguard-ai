# VisualGuard AI: project and skills interview questions

These are practice answers grounded in the implemented code and recorded results. Read the [project understanding guide](PROJECT_UNDERSTANDING.md) first. Answer in your own words; do not memorize claims you cannot explain or demonstrate. Numbers below refer to the **current v3 validation-image evaluation** unless a different version/split is named.

## A. Explaining the project

**1. What is VisualGuard AI?**

A local prototype for reviewing PPE observations in images and recorded video. It detects people, uncovered heads, helmets, and vests; associates objects with tracked people; creates candidate missing-helmet events after repeated video evidence; and shows evidence for human review.

**2. What problem were you trying to solve?**

A raw helmet detection does not say who owns the helmet, and a missing detection does not prove a person lacks one. I wanted a review workflow that links detections to people, preserves uncertainty, and records why an event was raised.

**3. Give a 30-second architecture explanation.**

Streamlit calls a FastAPI service. The API checks uploads and stores a job. Images are processed during the request; videos are claimed by a separate worker from a SQL queue. Each sampled frame goes through detection, person tracking, PPE association, and temporal rules. Results, annotated media, and review decisions are stored locally and shown in the dashboard.

**4. What are the main user flows?**

Run a clearly labeled synthetic demo; upload an image for an immediate observation; upload a video and poll its job ID; inspect boxes/timeline/media; review candidate events; and inspect analytics and model limitations.

**5. What makes this more than a YOLO notebook?**

The detector is one component. The repository also implements validated uploads, a durable queue and worker, person tracking, equipment association, temporal rules, stored review decisions, annotated media, analytics, a dashboard, tests, and recorded model comparisons.

**6. Is the project production ready?**

No. The current model missed 47 of 68 annotated uncovered heads in development validation. Real workplace videos and event accuracy have not been measured. The app also lacks authentication, retention automation, migrations, and deployment checks.

**7. What can a fresh GitHub clone demonstrate?**

After local setup it can run the synthetic end-to-end workflow and tests. The repository does not distribute the dataset or trained model weights, so real YOLO inference requires a compatible local checkpoint and configuration.

**8. What would you show first to an interviewer?**

Run the synthetic demo to show the queue, annotated video, event timeline, review, and analytics. Then show the model status page and explain the actual validation counts and limits. If my local checkpoint is installed, I can show an image inference while making clear it is not independent accuracy evidence.

## B. Data and label design

**9. Which dataset did you use?**

The Construction-PPE asset distributed by Ultralytics. The local archive had 1,416 images and an AGPL-3.0 license. The repository records the acquisition and audit, but does not publish the images.

**10. What are your four classes?**

`person`, `head`, `helmet`, and `vest`. Crucially, `head` maps the source `no_helmet` class: an annotated uncovered head, not every visible head.

**11. Why is that `head` definition important?**

The application treats an associated uncovered-head detection without an associated helmet as possible missing-helmet evidence. If `head` meant any visible head, the same rule would be invalid.

**12. How was the data split?**

Proxy grouping was applied before the split, yielding 989 training, 216 validation, and 211 test images. Original file family, exact SHA-256, and perceptual hash similarity were used to reduce obvious duplicate leakage.

**13. Does the split prove scene independence?**

No. Source camera, site, and scene identifiers were unavailable. Proxy groups help, but images from the same real scene could still cross partitions. I describe the test as exploratory.

**14. What is data leakage here?**

Near-duplicate views of the same scene appearing in both training and evaluation can make performance look better than it is. Using validation images repeatedly to choose thresholds and refinements also means those images are development data, not an independent final test.

**15. What quality issues did you find in labels?**

Head boxes varied in extent, some annotated heads did not sit mostly within a person box, person boxes sometimes omitted parts of the head, and non-workplace examples appeared. These are review flags and domain limitations; I did not relabel the source to improve the score.

**16. Why keep the original test split out of v2/v3 selection?**

If I tune on the test split, the final result is no longer an honest estimate of unseen data. V2 and v3 choices used validation data. The original v1 test result remains historical and should not be presented as current-v3 independent evidence.

**17. What would a better dataset look like?**

Separately sourced workplace images and video with camera/site identifiers, consistent annotation rules, difficult lighting/crowds/crops, and human-reviewed helmet states. Split by site or scene before choosing model settings, and keep a final holdout untouched.

**18. How did you validate the source files?**

The preparation code checks image decoding, class IDs, normalized box ranges, invalid geometry, and duplicates. It records source and prepared hashes, class mapping, split membership, and audit issues. The generic manifest preparer additionally requires explicit scene/source/license fields.

## C. Training and model selection

**19. Why use YOLO11n?**

It provides a compact, local object detector suitable for a CPU-based first experiment. The key requirement was a four-class detector that could be measured within the available compute budget; I did not benchmark other model families.

**20. What is transfer learning in this project?**

Training started from an official YOLO11n checkpoint and adapted it to the four PPE-related classes. Later runs continued from earlier project checkpoints.

**21. What happened in v1?**

It trained for eight time-limited CPU epochs at 320-pixel input, batch eight, with the first ten layers frozen. Its exploratory 211-image test showed 66.39% mAP50, but at confidence 0.4 it found 0 of 58 uncovered heads. The aggregate mAP hid the important class failure.

**22. What happened in v2?**

V2 added 12 epochs at 416 pixels with ten layers still frozen. Other classes improved in recall, but it still found 0 of 68 uncovered validation heads at confidence 0.4. It failed the head-specific gate and was not activated.

**23. What changed for v3 training?**

It started from v2, unfroze the backbone, used class-frequency weighting, ran 20 CPU epochs at 320 pixels, used AdamW with initial learning rate 0.0005, and disabled mosaic. Epoch 12 was retained because it had the best validation head AP50.

**24. Why select by head AP rather than overall AP?**

The project failed specifically on uncovered heads. A checkpoint with better average performance can still have unusable head detection. Head AP guided checkpoint selection, while class-level fixed-threshold gates checked whether the deployed configuration met the development target.

**25. Why not just lower the confidence threshold?**

On v1, lowering head confidence to 0.10 recovered 34 of 68 validation heads but produced 175 false head detections. That was about 16% precision. Lowering the threshold alone did not meet the precision/recall target.

**26. What is the difference between confidence and probability of a safety violation?**

The detector's confidence is a model score for an object box/class. It is not calibrated as the probability that a worker violated a rule. An event combines boxes, association, tracking, and temporal logic, and still requires review.

**27. What is the v3 head refinement?**

A four-parameter adjustment changes predicted head box center and size, fitted from training-image residuals. At inference, sufficiently overlapping head boxes are fused with confidence-weighted coordinates while retaining the highest original confidence. Head threshold is 0.25; other class thresholds are 0.4.

**28. Was the adjustment fitted on validation images?**

Its four numerical coefficients were fitted on 336 matched head predictions across 198 **training** images. But the choice to try adjustment, choose the 0.25 threshold, and choose fusion was guided by validation feedback. The final 216-image result is therefore a development result, not an independent test.

**29. Did larger input or extra crops solve the head problem?**

No. A 640-pixel validation pass and upper-person crop inference were tried and did not meet the target. The unsuccessful probes were recorded rather than silently dropped.

**30. Why merge boxes instead of deleting duplicates?**

Simple duplicate suppression reduced false boxes but left only 20 of 68 heads matched, below the 30% recall gate. Confidence-weighted box fusion improved the geometry enough to match 21 of 68 while leaving 7 false detections. It does not increase prediction confidence.

**31. How do you avoid claiming that a checkpoint and report match when they do not?**

The app checks the checkpoint SHA-256, profile SHA-256, image size, global confidence, and head confidence against the saved evaluation. Registration recomputes gates and verifies hashes before activation. A changed configuration makes the dashboard mark the evaluation as mismatched.

**32. What is the next model improvement?**

Review and standardize uncovered-head annotation semantics, add relevant workplace head examples, train with scene-aware splits, then evaluate on independently sourced workplace images and videos. I would track head recall and false alerts, not just aggregate mAP.

## D. Metrics and experimental reasoning

**33. Define precision and recall.**

Precision is `TP/(TP+FP)`: the fraction of predicted heads that matched a reference. Recall is `TP/(TP+FN)`: the fraction of reference heads found. For v3 heads, precision is `21/28 = 75%`; recall is `21/68 ≈ 30.9%`.

**34. What are the current head TP, FP, and FN?**

21 true positives, 7 false positives, and 47 false negatives on 216 development-validation images. These count object boxes, not people or video events.

**35. What does a false negative mean in this use case?**

An annotated uncovered head was not matched by a prediction under the evaluation rule. In a workflow, missed head evidence may leave the helmet state unknown or prevent a candidate alert. It does not imply the person was safe.

**36. Explain IoU.**

Intersection over union is the overlap area of two boxes divided by their union area. The project uses it in different places: at least 0.5 for evaluation matches, at least 0.25 to match person tracks, and 0.5 to group duplicate head boxes for fusion.

**37. Why use one-to-one matching in evaluation?**

Without it, two predictions could both claim the same reference head and inflate true positives. Here, the highest-confidence matching prediction consumes the reference; any extra prediction is a false positive.

**38. What is mAP50?**

Mean average precision across classes at box IoU 0.5, computed from precision–recall behavior across confidence levels. It is a detector measure. V1's exploratory test mAP50 was 66.39%; it does not mean 66.39% of workers were classified as safe.

**39. What is mAP50–95?**

Average AP across several IoU thresholds from 0.50 to 0.95. Higher IoU thresholds demand more precise localization. V1's exploratory test mAP50–95 was 36.33%.

**40. What is v3's mAP after refinement?**

It was **not recomputed** for the postprocessed application boxes. Raw checkpoint AP is saved separately, but I do not report that as the refined pipeline's AP. The dashboard says “Not measured.”

**41. What changed besides head detection?**

On validation, person recall rose from about 82.4% to 86.5%, helmet recall from 71.3% to 77.1%, and vest recall from 70.0% to 76.8%. Person precision fell from 85.6% to 77.6%, with person false positives increasing from 41 to 74. The comparisons have a preprocessing caveat.

**42. What were the v3 activation gates?**

Head precision at least 70%, head recall at least 30%, and no more than a five-percentage-point recall decrease for each of person, helmet, and vest relative to the comparison baseline. These are internal development criteria, not industry safety thresholds.

**43. Can you round 20/68 up to pass the recall gate?**

No. `20/68 ≈ 29.41%`, below 30%. The simple suppression variant failed; the fused variant's `21/68 ≈ 30.88%` passed. Code compares actual values, not rounded display percentages.

**44. What is the biggest limitation of the measured result?**

It misses 47 of 68 annotated uncovered heads, and the validation set helped select the model and refinement. We have no independently reviewed workplace videos, so there is no real alert accuracy claim.

**45. How would you evaluate the full video system?**

Collect reviewed clips from separate sites/cameras. Label people, PPE, temporal ground-truth events, and track identity changes. Measure detector precision/recall, event precision/recall, false alerts per video hour, detection delay, ID switches, and throughput. Compare by lighting, crowding, camera angle, and distance.

## E. Computer vision rules and failure modes

**46. How do you associate a helmet with a person?**

At least 80% of the helmet box must fall in one person's upper 40% region. If multiple people qualify, the evidence is ambiguous and cannot establish that person's helmet state. This avoids assigning one helmet to two people.

**47. What happens if both a helmet and uncovered head are matched?**

The helmet observation takes precedence and the state is present, provided the person is not cropped, crowded, or ambiguous. The result is still a model observation, not proof of compliance.

**48. Why is no helmet detection not enough to call a violation?**

The helmet may be occluded, too small, outside the crop, or missed by the detector. The rule requires positive uncovered-head evidence associated with the person; otherwise it returns unknown.

**49. Why does a missing vest detection become unknown?**

The model has a vest-present class but no reliable class proving a visible, unvested torso. Absence of a vest box can be caused by occlusion or detector error.

**50. What is an “unknown” state?**

An explicit decision that the image lacks enough trustworthy evidence. It appears for missing head evidence, border crops, crowding, ambiguous equipment ownership, and similar cases. It prevents false certainty.

**51. How are people tracked?**

A greedy IoU tracker matches person boxes between sampled frames, one detection per existing track, and expires stale tracks. It does not use appearance embeddings, face recognition, or cross-camera identity.

**52. What can cause an ID switch?**

People crossing, overlapping boxes, abrupt motion, camera movement, or missed frames can make a different person have the highest IoU with an old track. The prototype has no measured ID-switch rate.

**53. Why require three samples spanning 0.5 seconds?**

It reduces alerts from one-frame flicker. The two conditions also prevent three near-simultaneous samples from satisfying a duration requirement. The thresholds are design defaults, not empirically validated safety limits.

**54. What happens after a gap or an unknown observation?**

The current run closes. If it had enough evidence it becomes one candidate event; otherwise it is discarded. Later missing evidence starts a new run.

**55. Why does an image not create an event?**

One image has no temporal sequence. The app records boxes and a person observation, with zero video duration and no person-time or temporal candidate.

**56. Why might detector improvements fail to improve alerts?**

The detector could find more heads but fail to find their person boxes; boxes could be assigned to the wrong person; tracking could switch IDs; or temporal sampling could break a run. Detection metrics alone cannot measure those downstream errors.

## F. API, jobs, and database

**57. Why process videos in a separate worker?**

Video decoding and inference can take much longer than an HTTP request should remain open. The API creates a durable job and returns its ID; a worker can process it while the user polls progress and later retrieves artifacts.

**58. Why are images synchronous?**

An image is one frame and is generally short enough for the current local prototype to process within the request. The code still stores an image job and result for consistency. Large images are limited by upload and decoded-pixel checks.

**59. What is the job lifecycle?**

`queued → running → completed` or `failed`. A worker claims an eligible row, receives a token and 15-minute lease, heartbeats progress, and writes a result. An expired running job can be reclaimed.

**60. How do you stop two workers completing the same job?**

Claiming uses a conditional database update. Completion, heartbeat, and failure changes require the current random token; a stale worker cannot commit after another worker reclaims the job. Output folders are separated by attempt token.

**61. Why use SQLite WAL?**

It supports this local multi-process API/worker prototype with better read/write coexistence than the default journal mode. The database queue is simple to inspect and keeps job state after process restarts. It is not evidence of high-scale throughput.

**62. Did you run PostgreSQL or Docker?**

No. Compose and PostgreSQL connection support are configured, but they were not exercised under this project's local-only constraints. I would not claim a deployed container or validated PostgreSQL behavior.

**63. What happens if a worker crashes?**

Its heartbeat stops. After the 15-minute lease expires, a new worker may claim the job. The old token then cannot complete it. Each attempt writes to a separate artifact folder to avoid mixing partial outputs.

**64. How are event reviews stored?**

The events table stores a JSON event payload, review status, and note. The API accepts pending, confirmed, false alert, or uncertain with a bounded note; the dashboard updates and retrieves these values. Review decisions do not automatically retrain the detector.

**65. What input protections does the API have?**

It restricts extensions, upload bytes, decoded pixel count, and video duration; checks decoding; assigns generated filenames; constrains project paths; and serves only named artifacts. It is intended for trusted localhost use and has no authentication.

**66. What does `/health` mean?**

It says the service is running and reports whether a configured weights file is present. Its `model_validated` field is false; health is not a claim that model predictions are accurate. `/api/model` provides the checkpoint-bound evaluation status.

**67. What would you change for multi-user deployment?**

Add authentication and authorization, a retention policy, migration tooling, more robust queue/load handling, observability, privacy controls for identifiable footage, and tested PostgreSQL/container deployment. Those are future requirements, not features already demonstrated.

## G. Dashboard and analytics

**68. What does the dashboard show?**

Analyze/review, Event log, Analytics, and Model & data pages. Users can run the synthetic demo, upload media, inspect annotated results and events, record review decisions, and view class-level validation results and limitations.

**69. How is synthetic data kept separate from real uploads?**

Jobs carry a synthetic flag. The detector metadata names the fixture as not ML, the UI labels synthetic evidence, and the analytics API filters synthetic and non-synthetic completed videos separately.

**70. What is observed person-time?**

For each sampled video interval, count detected people and multiply by the interval length. Sum those seconds and divide by 60 for person-minutes. It counts detections over time, not unique workers; missed people can make it too low.

**71. What does “events per person-minute” mean here?**

Candidate event count divided by observed detected-person minutes for completed videos in the selected synthetic/real category. Repeated uploads count again, and the metric is not a compliance or risk score.

**72. Why show the model's limitations in the UI?**

Users otherwise might interpret a neat annotated video as reliable safety monitoring. The dashboard states the validation scope, missed heads, settings mismatch, and unmeasured postprocessing AP near the results.

## H. Testing, reproducibility, and release

**73. What tests were run?**

The original final run had 59 passing automated tests and 88.74% line coverage, plus Ruff checks. Tests cover dataset validation, tracker and event rules, queue ownership/recovery, API workflows, review persistence, candidate gates, hash binding, and head fusion. A browser smoke check exercised all four pages, synthetic video, review updates, and a real image upload.

**74. Does 88.74% coverage prove model quality?**

No. Coverage shows how much application code the tests execute. It does not measure detection, tracking, or event accuracy. Those need labeled evaluation data.

**75. Why have a synthetic demo if a real detector exists?**

It gives a deterministic way to test integration: uploads, worker, tracking, events, annotations, review, and analytics. Its color fixture cannot support ML accuracy claims, which is why it is labeled separately.

**76. How do you reproduce a model result?**

Keep the exact dataset/split manifest, labels, training settings, software versions, checkpoint SHA-256, inference settings, and scoring rule. The reports record these. Because the public repo omits weights and dataset media, an external reader can inspect the evidence but cannot rerun the same model result from GitHub alone.

**77. Why save SHA-256 hashes?**

A hash identifies the exact bytes of a checkpoint, profile, source archive, or split manifest. It helps prevent accidentally attaching old metrics to a changed model or dataset. It does not prove the data labels or model are correct.

**78. How did you check the public release?**

Before the first push, a clean project-local copy of the staged files passed all 59 tests. The staged list and ignore rules were checked to exclude `.env`, runtime database/uploads, dataset images, and weights. GitHub's `main` commit, README, and dashboard image were then verified.

**79. Is a Docker Compose file the same as a tested deployment?**

No. It describes a possible local configuration. Docker and PostgreSQL were not run and should not be listed as deployed or operationally verified skills based on this project.

**80. What is the license situation?**

The acquired data archive carried AGPL-3.0, and Ultralytics has its own licensing terms. The repository intentionally does not distribute the data, weights, or bundled Ultralytics code, and no license was chosen for this project's source. Any reuse or deployment needs a separate license review; I should not imply that posting source grants commercial rights.

## I. Personal ownership and tradeoffs

**81. What was the hardest technical problem?**

Uncovered-head detection. V1 and v2 both failed at the original operating threshold. I traced the issue through threshold sweeps and label/box inspection, tried targeted training and two unsuccessful inference probes, and retained a narrowly improved v3 configuration while documenting its low recall.

**82. Which tradeoff did you make deliberately?**

I chose an explicit unknown state and reviewable candidate events over automatic violation labels. That loses apparent coverage, but the underlying detector misses many heads and cannot establish absence from a missing helmet box.

**83. What would you do differently if starting again?**

Obtain a small, carefully annotated real workplace image/video set and consistent label rules before extensive model tuning. I would evaluate event-level errors early, because optimizing still-image boxes alone can miss tracking and association failures.

**84. Did you use AI assistance while building the project?**

Yes. I used Codex during planning and implementation. I should describe the decisions and code I can actually explain, identify assistance honestly if asked, and avoid claiming that I personally wrote or validated work I did not inspect.

**85. What result are you most confident reporting?**

The exact application detector's 216-image validation counts: 21 true, 7 false, and 47 missed uncovered-head boxes at the stated thresholds, plus the tested application workflow. I am equally explicit that these are development images and real video alert accuracy is unknown.

## Practice follow-ups that require a live explanation

An interviewer may ask you to draw one uploaded video's path through `api/main.py`, `store.py`, `worker/main.py`, and `pipeline.py`; calculate `21/(21+7)` and `21/(21+47)` without notes; explain why one head box cannot be assigned to two people; trace an unknown observation through `EventRules`; or show why a fresh clone runs the synthetic demo but not real inference. Practice those using the source files instead of memorizing this document.
