import time
from dataclasses import asdict
from pathlib import Path

import cv2

from visualguard.events.rules import EventRules
from visualguard.explain.annotate import annotate
from visualguard.ingestion.media import read_image, sampled_frames, video_info, video_writer
from visualguard.settings import Settings
from visualguard.vision.detector import build_detector
from visualguard.vision.matching import associate
from visualguard.vision.tracker import IoUTracker


def analyze(
    source: Path, output: Path, kind: str, settings: Settings, *, demo=False, progress=None
) -> dict:
    started = time.perf_counter()
    detector = build_detector(settings, demo=demo)
    tracker = IoUTracker(max_gap=settings.max_gap_seconds)
    rules = EventRules(
        settings.min_evidence_frames, settings.min_evidence_seconds, settings.max_gap_seconds
    )
    output.mkdir(parents=True, exist_ok=True)
    timeline = []
    observed_seconds = 0.0
    unknown_seconds = 0.0
    writer = None
    last_timestamp = -1.0
    if kind == "image":
        frames = [(0.0, read_image(source, settings), 0.0)]
        duration = 0.0
    else:
        info = video_info(source, settings)
        duration = info["duration_seconds"]
        frames = sampled_frames(source, settings)
        stride = max(1, round(info["fps"] / settings.sample_fps))
        writer = video_writer(
            output / "annotated.mp4", info["width"], info["height"], info["fps"] / stride
        )
    try:
        for timestamp, frame, interval in frames:
            if timestamp <= last_timestamp:
                raise ValueError("Non-increasing frame timestamps")
            last_timestamp = timestamp
            detections = [
                item
                for item in detector.predict(frame)
                if item.confidence >= settings.detection_threshold(item.label)
            ]
            tracked = tracker.update(
                [item for item in detections if item.label == "person"], timestamp
            )
            observations = associate(tracked, detections, frame.shape[1], frame.shape[0])
            if kind == "video":
                rules.update(observations, timestamp)
                observed_seconds += len(observations) * interval
                unknown_seconds += sum(item.helmet == "unknown" for item in observations) * interval
            annotated = annotate(frame, detections, observations, timestamp)
            if not timeline:
                cv2.imwrite(str(output / "preview.jpg"), annotated)
            if writer:
                writer.send(annotated.tobytes())
            else:
                cv2.imwrite(str(output / "annotated.jpg"), annotated)
            timeline.append(
                {
                    "timestamp": timestamp,
                    "people": len(observations),
                    "observations": [asdict(item) for item in observations],
                    "detections": [asdict(item) for item in detections],
                }
            )
            if progress:
                progress(min(0.99, (timestamp + interval) / max(duration, 1)))
    finally:
        if writer:
            writer.close()
    if not timeline:
        raise ValueError("No decodable frames")
    elapsed = time.perf_counter() - started
    return {
        "schema_version": 1,
        "synthetic": demo,
        "model": detector.name,
        "model_metadata": detector.metadata,
        "kind": kind,
        "duration_seconds": duration,
        "sampled_frames": len(timeline),
        "processing_seconds": round(elapsed, 4),
        "sampled_frames_per_second": len(timeline) / elapsed,
        "observed_person_seconds": observed_seconds,
        "unknown_helmet_person_seconds": unknown_seconds,
        "events": rules.finish(),
        "timeline": timeline,
        "artifact": "annotated.mp4" if kind == "video" else "annotated.jpg",
        "config": {
            "confidence": settings.confidence,
            "head_confidence": settings.detection_threshold("head"),
            "head_profile": settings.head_profile,
            "imgsz": settings.imgsz,
            "sample_fps": settings.sample_fps,
            "min_evidence_frames": settings.min_evidence_frames,
            "min_evidence_seconds": settings.min_evidence_seconds,
            "max_gap_seconds": settings.max_gap_seconds,
        },
        "limitations": [
            "Possible missing PPE requires human review.",
            "Vest absence is unknown with the current four-class model.",
            "IoU tracking can switch identities during crossings or occlusion.",
            "Image observations do not create temporal events or person-time.",
        ],
    }
