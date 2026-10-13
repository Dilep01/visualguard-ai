import math

import pytest

from visualguard.domain import Box
from visualguard.settings import Settings
from visualguard.vision.head_adjustment import adjust_head, merge_head_duplicates


def test_fusion_preserves_confidence_other_classes_and_inputs():
    rows = [
        {"class": "head", "confidence": 0.6, "box": [0, 0, 10, 10]},
        {"class": "head", "confidence": 0.3, "box": [1, 0, 11, 10]},
        {"class": "helmet", "confidence": 0.8, "box": [0, 0, 10, 10]},
        {"class": "head", "confidence": 0.4, "box": [30, 0, 40, 10]},
    ]
    result = merge_head_duplicates(rows)
    assert len(result) == 3
    assert result[0] == rows[2]
    assert result[1]["confidence"] == 0.6
    assert result[1]["box"][0] == pytest.approx(1 / 3)
    assert rows[0]["box"] == [0, 0, 10, 10]


def test_adjustment_clips_bounds_and_rejects_invalid_geometry():
    coefficients = dict(dx=0, dy=0, log_w=math.log(2), log_h=0)
    box = adjust_head(Box(0, 0, 10, 10), coefficients, 10, 10)
    assert box == Box(0, 0, 10, 10)
    for value in (float("nan"), float("inf"), 2):
        with pytest.raises(ValueError):
            adjust_head(Box(0, 0, 10, 10), {**coefficients, "dx": value}, 10, 10)


def test_head_threshold_does_not_lower_other_classes():
    settings = Settings(_env_file=None, confidence=0.4, head_confidence=0.25)
    assert settings.detection_threshold("head") == 0.25
    assert settings.detection_threshold("person") == 0.4
