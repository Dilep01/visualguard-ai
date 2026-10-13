# Head box adjustment experiment

Raw v3 at 320 improves head AP50 but has TP4/FP0/FN64 at confidence 0.4. At 0.25 its counts are TP20/FP12/FN48. Visual error review found predictions around heads with excess vertical extent relative to source annotations. Crop refinement and 640-pixel inference did not resolve this.

Fit four robust median box residuals using v3 predictions on head-positive TRAINING images only: center dx/dy divided by predicted width/height, and log width/height ratios. Match one-to-one by confidence, requiring IoU at least 0.1. Require at least 30 pairs. Preserve all source labels and splits. Freeze the resulting correction before validation.

Evaluate with head confidence 0.25, other classes 0.4, box IoU 0.5. Compare v1 at the same class-specific thresholds. Keep precision >=70% and recall >=30% for head, with at most five points of recall loss in other classes. Record the failed raw-v3 experiment separately. This is a development experiment motivated by validation errors, not independent testing or a claim of safety readiness. No test images are used.

Execution note: training-fitted adjustment alone did not pass the gates. Subsequent validation-guided duplicate suppression reached 20/68 recall and failed. Confidence-weighted geometry fusion at IoU 0.5 reached 21/68 with 7 false detections. The exact application path confirmed those counts with square preprocessing. This extra selection step was exploratory, not part of the initial plan; its results remain development evidence. See docs/head_refinement.md.
