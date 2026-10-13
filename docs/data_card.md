# Data card

## Included fixture

- Source: `visualguard.ingestion.demo`, generated locally from NumPy arrays and OpenCV shapes.
- Content: three geometric person silhouettes; one consistent helmet candidate resolves at four seconds; another head is unknown.
- Duration: eight seconds, 12 FPS, 800 × 480 pixels.
- Purpose: ingestion, queue, temporal-rule, artifact, review, and analytics testing.
- Real people, workplace recordings, and personal identities: none.
- Suitability for ML training or accuracy claims: none.

## External training data

Construction-PPE was acquired on 7 October 2026 from the official Ultralytics asset release. The archive includes its class YAML and AGPL-3.0 license. Exact asset URLs and hashes are in `data/external/construction-ppe-source/acquisition.json`. It contains 1,416 images; the four-class projection and proxy-grouped split are detailed in [baseline_report.md](baseline_report.md). Source scenes remain unverified, so this is exploratory data rather than a scene-independent benchmark. Hard Hat Universe was not downloaded.

Potential gaps to examine: head labeling semantics, vest visibility, small/distant workers, overlapping people, camera angle, lighting, helmet colors, partial bodies, and geographic/site diversity. Reserve separate reviewed video clips because image labels cannot establish tracking or event accuracy.
