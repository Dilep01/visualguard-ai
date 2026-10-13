# V3 delivery verification

- Full automated suite: 59 passed; coverage 88.74%. One existing Starlette/httpx deprecation warning.
- Ruff source, scripts, tests, and dashboard checks passed.
- Actual application detector evaluated on all 216 validation images: head TP21/FP7/FN47, precision 75%, recall 30.88%.
- Registration recomputed quality gates, verified checkpoint/profile hashes, and preserved the previous checkpoint and configuration.
- Live API confirmed evaluation settings match the active model and profile.
- Three preselected validation images uploaded successfully: sample-01 unknown (insufficient head evidence), sample-02 unknown (cropped/ambiguous person), sample-03 possible_missing. All are image observations with no temporal events.
- Browser check passed on all four pages: synthetic queue/worker, annotated MP4 metadata, persistent review note, analytics, model comparison, and real image upload. No browser errors. An initial check encountered duplicate summary text; its selector was fixed before the successful rerun.
- Model screenshot inspected: counts, per-class thresholds, unmeasured refined AP, validation scope, and missed-head limitations are visible.

Evidence: `reports/tests.xml`, `reports/coverage.xml`, `reports/browser-check.json`,
`reports/dashboard-model.png`, `reports/baseline/head-upload-check.json`, and
`reports/evaluation/construction-ppe-v3-profile-validation/detection_metrics.json`.

These checks establish application behavior and validation-image results only.
They do not validate real workplace video alerts. No real videos were supplied.
All changes and generated outputs for this continuation were inside the project.
