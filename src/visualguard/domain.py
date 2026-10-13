from dataclasses import dataclass
from typing import Literal

Label = Literal["person", "head", "helmet", "vest"]
Status = Literal["present", "possible_missing", "unknown"]


@dataclass(frozen=True)
class Box:
    x1: float
    y1: float
    x2: float
    y2: float

    @property
    def area(self) -> float:
        return max(0, self.x2 - self.x1) * max(0, self.y2 - self.y1)

    def intersection(self, other: "Box") -> float:
        return max(0, min(self.x2, other.x2) - max(self.x1, other.x1)) * max(
            0, min(self.y2, other.y2) - max(self.y1, other.y1)
        )

    def iou(self, other: "Box") -> float:
        intersection = self.intersection(other)
        return intersection / max(self.area + other.area - intersection, 1e-9)

    def region(self, top: float, bottom: float) -> "Box":
        height = self.y2 - self.y1
        return Box(self.x1, self.y1 + top * height, self.x2, self.y1 + bottom * height)


@dataclass(frozen=True)
class Detection:
    label: Label
    confidence: float
    box: Box


@dataclass
class Observation:
    track_id: int
    box: Box
    helmet: Status
    vest: Status
    confidence: float
    reason: str
