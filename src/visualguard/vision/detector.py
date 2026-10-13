import hashlib
import importlib.metadata
import json
from pathlib import Path

import cv2
import numpy as np

from visualguard.domain import Box, Detection
from visualguard.model_status import model_status
from visualguard.settings import Settings, configure_vision, project_path
from visualguard.vision.head_adjustment import adjust_head, merge_head_duplicates

# BGR colors are encoded into the generated demo only; this is NOT an ML detector.
DEMO_COLORS = {
    "person": (190, 80, 40),
    "head": (160, 180, 210),
    "helmet": (20, 220, 245),
    "vest": (60, 220, 70),
}


class DemoDetector:
    name = "synthetic-color-fixture-v1 (not machine learning)"
    metadata = {"type": "synthetic_fixture", "trained_model": False}

    def predict(self, frame: np.ndarray) -> list[Detection]:
        detections = []
        for label, color in DEMO_COLORS.items():
            base = np.array(color, dtype=np.int16)
            mask = cv2.inRange(
                frame,
                np.clip(base - 35, 0, 255).astype(np.uint8),
                np.clip(base + 35, 0, 255).astype(np.uint8),
            )
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for contour in contours:
                x, y, width, height = cv2.boundingRect(contour)
                if width * height >= 100:
                    detections.append(Detection(label, 1.0, Box(x, y, x + width, y + height)))
        return detections


class YoloDetector:
    def __init__(self, settings: Settings):
        weights = project_path(settings.weights)
        if not weights.is_file():
            raise ValueError("PPE weights are missing. Place verified weights at VG_WEIGHTS.")
        try:
            configure_vision()
            from ultralytics import YOLO
        except ImportError as exc:
            raise ValueError("Install the optional vision dependencies first.") from exc
        self.model = YOLO(str(weights))
        names = self.model.names
        self.names = {int(key): str(value).lower() for key, value in names.items()}
        if set(self.names.values()) != {"person", "head", "helmet", "vest"}:
            raise ValueError("Model labels must be exactly person, head, helmet, vest.")
        self.settings = settings
        self.profile = None
        profile_digest = None
        if settings.head_profile:
            profile_bytes = project_path(settings.head_profile).read_bytes()
            self.profile = json.loads(profile_bytes)
            profile_digest = hashlib.sha256(profile_bytes).hexdigest()
            if self.profile["weights_sha256"] != hashlib.sha256(weights.read_bytes()).hexdigest():
                raise ValueError("Head profile does not match the checkpoint")
            if self.profile["imgsz"] != settings.imgsz or self.profile[
                "head_confidence"
            ] != settings.detection_threshold("head"):
                raise ValueError("Head profile does not match the inference settings")
        self.name = f"yolo:{Path(weights).name}"
        self.metadata = {
            "type": "yolo",
            "checkpoint_sha256": hashlib.sha256(weights.read_bytes()).hexdigest(),
            "ultralytics_version": importlib.metadata.version("ultralytics"),
            "class_names": self.names,
            "validation_status": model_status(settings)["status"],
            "head_profile_sha256": profile_digest,
            "head_confidence": settings.detection_threshold("head"),
        }

    def predict(self, frame: np.ndarray) -> list[Detection]:
        result = self.model.predict(
            frame,
            conf=min(self.settings.confidence, self.settings.detection_threshold("head")),
            device=self.settings.device,
            imgsz=self.settings.imgsz,
            verbose=False,
            rect=self.profile["rect"] if self.profile else True,
        )[0]
        predictions = []
        for detected in result.boxes:
            label = self.names[int(detected.cls.item())]
            confidence = float(detected.conf.item())
            if confidence < self.settings.detection_threshold(label):
                continue
            box = Box(*map(float, detected.xyxy[0].tolist()))
            if label == "head" and self.profile:
                box = adjust_head(box, self.profile["coefficients"], frame.shape[1], frame.shape[0])
            predictions.append(
                {"class": label, "confidence": confidence, "box": [box.x1, box.y1, box.x2, box.y2]}
            )
        if self.profile:
            predictions = merge_head_duplicates(predictions, self.profile["merge_iou"])
        return [Detection(p["class"], p["confidence"], Box(*p["box"])) for p in predictions]


def build_detector(settings: Settings, *, demo: bool = False):
    if demo:
        return DemoDetector()
    if settings.backend != "yolo":
        raise ValueError("Real uploads require VG_BACKEND=yolo and trained PPE weights. Use demo.")
    return YoloDetector(settings)
