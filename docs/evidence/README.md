# Published evaluation evidence

These files preserve selected results from the original local development run:

- `v3-validation.json`: metrics from the actual application detector on 216 validation images.
- `v1-v3-comparison.json`: comparison at matching per-class thresholds and development gates.
- `inference-profile.json`: the checkpoint-bound head geometry adjustment and fusion configuration.
- `dashboard-model.png`: the local dashboard displaying the recorded evaluation.

These are recorded results, not a new independent test. They do not make a fresh
clone report a verified model: the application checks the installed checkpoint,
profile, and inference settings. Model weights, images, full per-image predictions,
runtime databases, and local configuration are not distributed here. Paths inside
the reports refer to the original project layout and may not exist in a fresh clone.

The original local checks passed 59 tests, with 88.74% line coverage. All four
dashboard pages, synthetic video playback, review persistence, and real image
uploads were checked. See `reports/baseline/verification-v3.md` for the scope.

The model missed 47 of 68 uncovered heads. No real workplace video accuracy was
measured. Full development details are in [head_refinement.md](../head_refinement.md).
