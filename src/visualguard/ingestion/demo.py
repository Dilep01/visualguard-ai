from pathlib import Path

import cv2
import numpy as np

from visualguard.ingestion.media import video_writer
from visualguard.vision.detector import DEMO_COLORS


def demo_frame(timestamp: float) -> np.ndarray:
    frame = np.full((480, 800, 3), (28, 23, 19), dtype=np.uint8)
    cv2.putText(
        frame,
        "SYNTHETIC WORKFLOW DEMO - NOT MODEL RESULTS",
        (24, 38),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (220, 220, 220),
        1,
        cv2.LINE_AA,
    )
    for index, base in enumerate((100, 340, 580)):
        x = base + int(12 * np.sin(timestamp * 0.5))
        cv2.rectangle(frame, (x, 100), (x + 100, 405), DEMO_COLORS["person"], -1)
        if index != 2:  # Third person's head is obscured -> unknown.
            cv2.rectangle(frame, (x + 26, 113), (x + 74, 154), DEMO_COLORS["head"], -1)
        if index == 0 or (index == 1 and timestamp >= 4):
            cv2.rectangle(frame, (x + 18, 104), (x + 82, 123), DEMO_COLORS["helmet"], -1)
        if index != 1:
            cv2.rectangle(frame, (x + 15, 175), (x + 85, 290), DEMO_COLORS["vest"], -1)
        caption = ("Equipment present", "Review then resolved", "Head unknown")[index]
        cv2.putText(
            frame,
            caption,
            (x - 24, 440),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            (220, 220, 220),
            1,
            cv2.LINE_AA,
        )
    return frame


def create_demo(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = video_writer(path, 800, 480, 12)
    try:
        for index in range(96):
            writer.send(demo_frame(index / 12).tobytes())
    finally:
        writer.close()
