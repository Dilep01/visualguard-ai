# Data provenance and preparation

Construction-PPE has now been acquired and audited. See [the baseline report](../docs/baseline_report.md) for provenance, license, class projection, split limitations, and evaluation. `data/demo/` contains generated fixtures and a separately labeled real validation sample with a provenance sidecar. Do not use either demonstration as independent accuracy evidence.

The proposal identified [Hard Hat Universe](https://universe.roboflow.com/universe-datasets/hard-hat-universe-0dy7t) as a candidate. Its exact version, license, class definitions, source scenes, and redistribution permissions remain unverified. A listing-level license statement is insufficient for recording a version as approved.

Download permitted data manually into `data/external/`, then record the exact version and terms in `docs/data_card.md`. Normalize class IDs to `0 person`, `1 head`, `2 helmet`, `3 vest`. Review whether the source's `head` class includes helmeted heads or only bare heads; do not silently combine incompatible definitions.

Create a CSV manifest inside the project with these columns:

```csv
image,label,scene,source,license
```

Paths are relative to the manifest's directory. One row represents one image and its YOLO text-label file. Empty label files represent negative images. `scene` must identify the original scene or video, not the individual frame. Include source/version references and license evidence in the last two fields.

```powershell
.\.venv\Scripts\python.exe scripts/prepare_dataset.py --manifest data/external/manifest.csv
```

The preparer rejects malformed boxes, unknown class IDs, exact duplicate images, missing provenance fields, fewer than three independent scenes, and paths outside the project. It assigns approximately 70/15/15 percent of **scenes**, not necessarily images, and guarantees at least one scene per split. It writes a hash-bearing split manifest and YOLO dataset YAML. Near-duplicate images across different scenes still require manual or perceptual-deduplication review.

Training and held-out detection evaluation:

```powershell
.\.venv\Scripts\python.exe scripts/train.py --model models/trusted-base.pt --device cpu
.\.venv\Scripts\python.exe scripts/evaluate.py --weights models/ppe.pt --device cpu
```

These commands require the optional vision dependencies. Supply local model weights yourself; the scripts do not select a model release, download a dataset, or fabricate evaluation numbers. Keep the test split untouched during threshold tuning. Record calibration and temporal-rule settings before reviewing final test clips.
