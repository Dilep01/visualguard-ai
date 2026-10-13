# VisualGuard AI: complete project understanding

This guide describes the code and recorded results in this repository. It is written for someone who needs to demonstrate the application and explain the design in an interview. The separate [interview guide](PROJECT_INTERVIEW_QA.md) turns the same evidence into questions and answers.

## 1. What the project does

VisualGuard AI is a **local, experimental PPE review application**. It accepts a photograph or recorded video, detects people and selected PPE-related objects, links detections to tracked people, records cautious helmet observations, and presents evidence for human review. It has a FastAPI service, a separate video worker, a SQL database, and a Streamlit dashboard. The synthetic demo exercises the workflow without real people or an ML model.

The application is **not a reliable workplace safety monitor**. Its current trained detector found 21 of 68 annotated uncovered heads on 216 development-validation images. No real workplace videos were supplied, so tracking and video alert accuracy have not been measured. The database stores *candidate* events and review decisions, not proven violations.

### One-minute explanation

> I built a local PPE review prototype around a four-class YOLO detector. Images are analyzed immediately; videos enter a durable database queue and a separate worker processes sampled frames. The pipeline tracks people, associates helmets and uncovered heads with each person, and requires repeated video evidence before producing a possible missing-helmet event. Ambiguous observations stay unknown. I evaluated several model versions and documented the failure cases: the current detector found 21 of 68 validation heads at 75% precision, but missed 47, so it is not ready for operational safety decisions.

## 2. Skills and technologies actually used

| Area | Technology | What it does here |
|---|---|---|
| Programming | Python 3.11 | Application, model workflow, data audit, evaluation, tests |
| Deep learning | PyTorch, Ultralytics YOLO11n | Transfer learning and four-class object detection; CPU training in this run |
| Vision and media | OpenCV, NumPy, Pillow, imageio-ffmpeg | Decode/check images and video, sample frames, track box geometry, draw annotations, encode H.264 MP4 |
| API | FastAPI, Uvicorn, Pydantic | Uploads, job status, evidence, review updates, analytics, input validation |
| Data persistence | SQLAlchemy, SQLite WAL | Durable job queue, result records, review decisions; PostgreSQL is configured but untested |
| UI | Streamlit, pandas, requests | Upload/review workflow, job/media display, analytics, model status |
| Evaluation | Precision, recall, IoU, AP/mAP, SHA-256 | Compare detections with labels, check model candidates, bind reports to artifacts |
| Quality | pytest, pytest-cov, Ruff, Playwright browser check | Unit/integration checks, coverage, style, four-page UI smoke test |
| Local operations | PowerShell, Git, GitHub | Start/stop services, manage local environment, publish source and selected evidence |

These are **project skills**, not a claim of professional workplace deployment. Docker Compose and PostgreSQL configuration exist, but were **not run**. There is no Redis, Celery, cloud deployment, production authentication, facial recognition, VLM assistant, Grad-CAM, SHAP, or anomaly model in the working application. The tracker is a simple IoU tracker, not DeepSORT or re-identification.

## 3. How the system works

```text
Browser / Streamlit
       |
       v
FastAPI validates upload and creates a job
       | image: claim and process in request
       | video: return job ID; separate worker claims it
       v
Decode or sample frames -> detector -> person tracker -> PPE association
       -> temporal helmet rule (video only) -> annotated image/MP4 + JSON result
       -> SQL jobs/events -> API -> dashboard, review, analytics
```

An **image** request finishes synchronously. It records a single observation and annotated image. It cannot generate a temporal event, duration, or person-time. A **video** request responds with HTTP 202 and a job ID. The worker polls the database, claims a job, heartbeats during processing, and saves a result. The dashboard polls job state and displays annotated media, a timeline, events, and review controls.

The API is local to `127.0.0.1` in the tested launcher. `/health` is a service check and explicitly does **not** claim model validation. `/api/model` reports whether the installed checkpoint, image size, thresholds, and head profile match the saved evaluation. The main endpoints are `/api/demo`, `/api/images`, `/api/videos`, `/api/jobs`, `/api/jobs/{id}`, `/api/jobs/{id}/media/{name}`, `/api/events`, `/api/events/{id}/review`, and `/api/analytics`. FastAPI documents them at `http://127.0.0.1:8000/docs` when running.

### Four labels and what they mean

`person` is a detected person box; `helmet` and `vest` are detected equipment boxes. `head` maps the source dataset's **`no_helmet`** label, meaning an explicitly annotated uncovered head. It is **not** a generic “any visible head” class. This mapping is central to the missing-helmet logic.

The detector creates boxes and confidences. The application then makes **observations**, which are different from raw boxes:

- Helmet **present**: a helmet box is uniquely associated with an eligible person.
- Helmet **possible_missing**: an uncovered-head box is uniquely associated with an eligible person and no helmet is associated; this still needs human review.
- Helmet **unknown**: the necessary evidence is missing or the person is cropped, crowded, or ownership is ambiguous.
- Vest **present** can be observed; missing vest detection is always **unknown** because this model has no reliable torso-visibility/absence class.

The association rule requires at least 80% of an equipment box to lie within a candidate person's region. Head/helmet use the upper 40% of the person box; vest uses 15%–80% of its height. The equipment box must belong to exactly one person. If a person is at the image border, overlaps another person substantially, or has ambiguously owned equipment, helmet state is unknown. A matched helmet takes precedence over a matched uncovered head. These are conservative geometry rules, not a trained occlusion classifier.

### Tracking and temporal events

`IoUTracker` greedily matches each new person box to at most one recent track at box IoU at least 0.25. Tracks expire after a 0.75-second gap by default. IDs exist only within one clip; crossings, occlusion, camera movement, and missed detections can switch or fragment IDs.

For a video event, `EventRules` needs at least **three consecutive sampled `possible_missing` observations** spanning at least **0.5 seconds**. An unknown/present observation or a long gap closes the run. A qualifying run becomes one `possible_missing_helmet` candidate. The event stores the first and last supporting timestamps, number of supporting frames, track ID, reason, and box. Its confidence is the **minimum supporting detection confidence**, not an event probability. Image observations never create such events.

## 4. The work done, step by step

1. **Defined a narrow use case.** The original idea contained several possible computer-vision features. The implemented scope became person/helmet/vest/uncovered-head detection, local tracking, reviewable helmet rules, an API, and dashboard. More speculative components were deferred.
2. **Built the application workflow with a synthetic fixture.** A generated clip contains colored silhouettes. A deterministic color-based detector lets the queue, tracker, rule, media, review, and analytics behavior be tested without claiming ML accuracy.
3. **Added upload checks and local persistence.** The API validates file extensions, size, decoded pixels, duration, and frame metadata. Uploaded names are replaced by UUIDs. SQLite stores job states and events; a video worker claims jobs with a random ownership token and a renewable 15-minute lease. Stale workers cannot complete a reclaimed job.
4. **Acquired and audited the Construction-PPE dataset.** The original local archive contained 1,416 images and an AGPL-3.0 license. Source labels were projected onto four classes. Exact hashes, filename families, and perceptual hashes were used as proxy grouping before a 989 train / 216 validation / 211 test split. There were 1,303 proxy groups. Source camera/scene IDs were unavailable, so the split is not proven scene independent. Source images and weights are excluded from GitHub.
5. **Trained and evaluated v1.** An official YOLO11n starting checkpoint was fine-tuned on CPU at 320-pixel input, batch 8. The time-limited run completed eight epochs. On 211 exploratory test images, mAP50 was 66.39% and mAP50–95 was 36.33%. At confidence 0.4 it found **0/58** uncovered heads. This is a historical v1 test result; it is not a current-v3 score.
6. **Investigated the head-class failure.** Lowering v1 head confidence recovered heads but produced many false detections. Label review found non-workplace head examples, inconsistent head/person box geometry, and some head boxes not contained in a person box. The labels were not silently changed. These observations motivated a new training attempt; they do not prove a single root cause.
7. **Trained and rejected v2.** V2 added 12 epochs at 416-pixel input while keeping the first ten layers frozen. It still found **0/68** validation heads at confidence 0.4 and failed the class-specific acceptance rules, despite improving other classes. The active model stayed at v1 at that stage.
8. **Trained v3 for the head class.** V3 continued from v2 for 20 CPU epochs at 320 pixels, with the backbone unfrozen, class-frequency weighting, AdamW at learning rate 0.0005, and mosaic disabled. The checkpoint from epoch 12 was selected by validation head AP50 rather than aggregate AP. Raw v3 at confidence 0.4 found **4/68** validation heads, which was still insufficient.
9. **Tested and recorded failed refinements.** A 640-pixel pass and a second upper-person crop pass did not resolve the head problem. At a lower head threshold, some predicted head boxes overlapped or differed from source label geometry. The discarded approaches remain in local experiment records.
10. **Fitted a small geometry adjustment and merged duplicate heads.** Four box residual parameters were estimated from **336 matched predictions in 198 training images only**. A head-specific 0.25 confidence threshold was used; other classes stayed at 0.4. Overlapping head boxes at IoU 0.5 were merged by confidence-weighted geometry while retaining the strongest original confidence. Simple suppression failed the recall gate. Fusion reached **21 true head detections, 7 false detections, and 47 misses** on validation.
11. **Ran the exact application detector on all validation images.** That check confirmed **75.0% head precision and 30.9% head recall** on 216 validation images. Candidate registration recomputed the comparison, verified checkpoint/profile hashes, saved rollback files, and activated v3 for **experimental** local use. AP was not recomputed after postprocessing and appears as “Not measured” in the dashboard.
12. **Checked the integrated app and published the source.** The final original run recorded 59 passing tests and 88.74% line coverage. A browser check exercised four pages, synthetic video playback, review persistence, and a real image upload. Three selected validation image uploads produced two unknown observations and one possible-missing observation; these are integration examples, not accuracy estimates. The public GitHub repository contains source, setup guidance, and selected evaluation evidence, but no model checkpoint, dataset, uploads, database, or local `.env`.

### Current measured detection results

These are **validation object detections**, with confidence 0.25 for heads, 0.4 for other classes, and one-to-one box matching at IoU 0.5. They are not event or safety metrics.

| Class | True positives | False positives | False negatives | Precision | Recall |
|---|---:|---:|---:|---:|---:|
| Uncovered head | 21 | 7 | 47 | 75.0% | 30.9% |
| Person | 256 | 74 | 40 | 77.6% | 86.5% |
| Helmet | 185 | 13 | 55 | 93.4% | 77.1% |
| Vest | 179 | 22 | 54 | 89.1% | 76.8% |

At the same head threshold, cached v1 validation predictions found **1/68** heads. That comparison uses the same image membership and confidence thresholds, but the cached v1 and final v3 pipelines use different preprocessing, so do not present the difference as a perfectly controlled architecture comparison. The v3 person false detections increased from 41 to 74 relative to v1. The gates used for experimental selection were at least 70% head precision, at least 30% head recall, and no more than five percentage points of recall loss for person, helmet, or vest. Passing these gates is **not a safety certification**. Validation images influenced model choices, so an independent final evaluation is still needed.

## 5. Understanding the metrics

- **True positive (TP):** one predicted object of the correct class matches one reference box at IoU at least 0.5.
- **False positive (FP):** a predicted object has no unused matching reference of the same class.
- **False negative (FN):** a reference object was not matched.
- **Precision = TP / (TP + FP).** For the current head class: `21 / (21 + 7) = 75%`.
- **Recall = TP / (TP + FN).** For the current head class: `21 / (21 + 47) ≈ 30.9%`.
- **IoU = intersection area / union area** for two boxes; it is used for matching, tracking, and head-duplicate fusion, with different thresholds for each purpose.
- **AP/mAP:** area under a precision–recall curve for detector boxes; it is not the fraction of safe workers or correct events. The v1 exploratory test has mAP values. The refined v3 pipeline has **no recomputed postprocessing AP**, so do not attribute raw checkpoint AP to the deployed detector.
- **Observed person-time:** sum of detected people multiplied by sampled time intervals in videos. It is not a count of unique workers or independently measured exposure time. Repeated uploads count again.

## 6. Important implementation choices

**Durable queue.** A SQL row persists when the API process restarts. A conditional claim and random token prevent two workers from both committing a result. Heartbeats extend the lease; each attempt writes to a separate output folder. Completion and event insertion share one database transaction. Local SQLite uses WAL. PostgreSQL appears in Compose, but it has not been verified in a running container.

**Model/report integrity.** The detector checks that class names match the four labels and that a head-refinement profile belongs to the checkpoint and image size. The model status also checks evaluation settings and hashes. If configuration differs, the recorded metrics must not be presented as describing the running settings. The synthetic detector is labeled separately from a trained model.

**Input and path boundaries.** Project paths are resolved under the project root. The API limits upload bytes, decoded pixels, and video duration, validates media, and only serves known artifact names. This reduces accidental exposure and misuse, but the application has no user authentication or production retention policy.

**Review rather than automatic judgment.** “Possible missing” means evidence worth checking. No alert means neither helmet compliance nor safety. “Unknown” is a meaningful output when evidence is insufficient.

## 7. Where to find each part

| File or folder | Purpose |
|---|---|
| `src/visualguard/api/main.py` | HTTP endpoints, upload checks, API responses |
| `src/visualguard/pipeline.py` | Frame-by-frame analysis orchestration |
| `src/visualguard/vision/` | Real/synthetic detectors, tracker, association, box refinement |
| `src/visualguard/events/rules.py` | Temporal event conditions |
| `src/visualguard/persistence/store.py` | Jobs, leases, events, reviews in SQL |
| `src/visualguard/worker/main.py` | Video worker loop |
| `src/visualguard/analytics/metrics.py` | Summary counts and person-time denominators |
| `dashboard/app.py` | Four-page Streamlit UI |
| `scripts/` | Acquisition, preparation, training, evaluation, registration, launch |
| `tests/` | Unit and integrated workflow checks |
| `docs/evidence/` | Publicly shareable validation metrics and dashboard screenshot |

The local `models/`, `data/processed/`, `runtime/`, and `.env` contain generated or private assets and are excluded from Git. A fresh GitHub clone can run the synthetic workflow after setup; real inference additionally needs a compatible local checkpoint and configuration.

## 8. How to demonstrate it

1. In the original configured checkout, run `./scripts/start.ps1` from PowerShell. A fresh clone needs Python 3.11+, local installation, and `.env.example` copied to `.env` as described in the root README.
2. Open `http://127.0.0.1:8501`. Run **synthetic demo**, refresh job status, and show the annotated clip and candidate timeline. State that the color fixture proves application plumbing, not detector accuracy.
3. Open **Event log** and change a review decision. Open **Analytics** and explain that synthetic results are separated from real uploads and the denominator is observed person-time.
4. Open **Model & data**. Explain the current head TP/FP/FN counts, the validation-only scope, why refined AP says “Not measured,” and the 47 missed heads.
5. If the local v3 weights are installed, upload a permitted image. Explain the detected boxes, the per-person observation, and why an image produces no temporal event. Do not describe a validation image as independent workplace testing.
6. Stop the owned local services with `./scripts/start.ps1 -Stop`.

The public repository is [Dilep01/visualguard-ai](https://github.com/Dilep01/visualguard-ai). The root README gives setup and dependency instructions.

## 9. What remains unverified

There is no measured real-video event precision/recall, false alerts per video hour, tracking ID-switch rate, or site-independent workplace test. The model misses most annotated uncovered heads and makes more person false detections than v1. Dataset grouping cannot prove different camera scenes. Face identity, worker identity, and camera-to-camera tracking are not implemented. The app is for trusted localhost use, with no login, automated deletion, or production operations. Explain these limits directly when presenting the project.

The highest-value next evaluation is a separately sourced, human-reviewed workplace video set with camera/site splits and frame-level as well as event-level labels. Only after that can the detector, tracking, event rules, and alert rate be judged together.
