from copy import deepcopy

import pytest

from visualguard.training.comparison import compare_candidate


def reports():
    baseline = {
        "split": "val",
        "split_manifest_sha256": "split",
        "dataset_config_sha256": "config",
        "images": 216,
        "imgsz": 416,
        "weights_sha256": "baseline",
        "fixed_operating_point": {
            "confidence": 0.4,
            "iou": 0.5,
            "per_class": {
                name: {"precision": 0.8, "recall": 0.8}
                for name in ("person", "head", "helmet", "vest")
            },
        },
    }
    baseline["fixed_operating_point"]["per_class"]["head"] = {"precision": None, "recall": 0}
    candidate = deepcopy(baseline)
    candidate["weights_sha256"] = "candidate"
    candidate["fixed_operating_point"]["per_class"]["head"] = {"precision": 0.75, "recall": 0.4}
    return baseline, candidate


def test_candidate_with_better_head_detection_is_still_not_safety_validated():
    result = compare_candidate(*reports())
    assert result["eligible_for_experimental_use"]
    assert result["safety_validated"] is False


@pytest.mark.parametrize(
    "field,value",
    [
        ("split", "test"),
        ("imgsz", 320),
        ("images", 215),
        ("split_manifest_sha256", "other"),
        ("dataset_config_sha256", "other"),
    ],
)
def test_incomparable_or_test_results_cannot_select_candidate(field, value):
    baseline, candidate = reports()
    candidate[field] = value
    with pytest.raises(ValueError):
        compare_candidate(baseline, candidate)


def test_better_head_recall_does_not_override_other_class_regression():
    baseline, candidate = reports()
    candidate["fixed_operating_point"]["per_class"]["helmet"]["recall"] = 0.7
    assert not compare_candidate(baseline, candidate)["eligible_for_experimental_use"]


def test_no_predictions_cannot_pass_precision_gate():
    baseline, candidate = reports()
    candidate["fixed_operating_point"]["per_class"]["head"]["precision"] = None
    assert not compare_candidate(baseline, candidate)["eligible_for_experimental_use"]


def test_exact_five_point_recall_drop_is_within_the_declared_limit():
    baseline, candidate = reports()
    candidate["fixed_operating_point"]["per_class"]["helmet"]["recall"] = 0.75
    assert compare_candidate(baseline, candidate)["eligible_for_experimental_use"]


def test_threshold_changes_cannot_pass_as_like_for_like_comparison():
    baseline, candidate = reports()
    candidate["fixed_operating_point"]["confidence"] = 0.1
    with pytest.raises(ValueError, match="confidence"):
        compare_candidate(baseline, candidate)


def test_head_thresholds_must_match_and_recall_is_not_rounded_up():
    baseline, candidate = reports()
    candidate["fixed_operating_point"]["head_confidence"] = 0.25
    with pytest.raises(ValueError, match="head_confidence"):
        compare_candidate(baseline, candidate)
    baseline["fixed_operating_point"]["head_confidence"] = 0.25
    candidate["fixed_operating_point"]["per_class"]["head"]["recall"] = 20 / 68
    assert not compare_candidate(baseline, candidate)["eligible_for_experimental_use"]
