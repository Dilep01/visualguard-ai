# Candidate v3 plan

Recorded before v3 training/evaluation. V2 did not resolve head detection. Preserve its rejection and all original test evidence.

Start from v2's local checkpoint. Train 20 epochs at 320 pixels, batch 8, four CPU threads, all backbone layers unfrozen, AdamW initial learning rate 0.0005, class-weight power 1.0, mosaic disabled, warmup one epoch with bias warmup learning rate zero, seed 42. This is a combined intervention, not an ablation identifying one cause. Dataset labels and partition membership remain unchanged.

Save a separate best-head checkpoint by validation head AP50 each epoch, in addition to Ultralytics' aggregate-best checkpoint. Evaluate the head-selected checkpoint on validation at 320 pixels and confidence 0.4. Compare with the existing v1 validation report at the same settings. Retain the existing development gates: head precision >=70%, head recall >=30%, each other class recall loses no more than five percentage points. These gates do not establish safety readiness. No test sweep or test-based selection.

After selection, exercise actual detector/association behavior and image/video application paths. Real video-event accuracy still requires independently reviewed workplace footage; synthetic fixtures or repeated still frames cannot provide that evidence.
