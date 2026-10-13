# Candidate v2 experiment plan

Written before candidate training/evaluation. Baseline validation: 216 images, 68 uncovered-head boxes. At confidence 0.4: head TP 0, FP 0, FN 68. At 0.1: TP 34, FP 175, FN 34. Lowering confidence alone is not an acceptable fix.

Candidate: continue from v1 best weights for 12 epochs, input 416, batch 8, first 10 layers frozen, AdamW lr 0.001, four CPU threads, seed 42. Original train/val/test membership and labels unchanged. No test evaluation during this experiment.

Comparison: evaluate both v1 and candidate at 416 and confidence 0.4 on validation. Also retain v1's 320 validation result to expose the effect of input size. Selection for experimental local use requires head precision >= 0.70, head recall >= 0.30, and no more than 0.05 absolute loss in each other class's recall versus v1 at 416. These are development gates, not a workplace-safety standard. Failure leaves v1 active.

Regardless of result, scene independence and real video event accuracy remain unverified. Original test results are already consumed and cannot act as a new independent final test after model development.
