import cv2

from visualguard.domain import Detection, Observation


def annotate(frame, detections: list[Detection], observations: list[Observation], timestamp):
    canvas = frame.copy()
    for item in detections:
        if item.label == "person":
            continue
        box = item.box
        cv2.rectangle(
            canvas, (int(box.x1), int(box.y1)), (int(box.x2), int(box.y2)), (230, 185, 90), 1
        )
        cv2.putText(
            canvas,
            f"{item.label} {item.confidence:.2f}",
            (int(box.x1), max(12, int(box.y1) - 4)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.35,
            (240, 220, 170),
            1,
        )
    for item in observations:
        color = {
            "present": (100, 210, 70),
            "possible_missing": (70, 150, 255),
            "unknown": (180, 180, 180),
        }[item.helmet]
        box = item.box
        cv2.rectangle(canvas, (int(box.x1), int(box.y1)), (int(box.x2), int(box.y2)), color, 2)
        cv2.putText(
            canvas,
            f"#{item.track_id} helmet:{item.helmet}",
            (int(box.x1), max(14, int(box.y1) - 20)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.40,
            color,
            1,
        )
    cv2.putText(
        canvas,
        f"{timestamp:.2f}s | Review observations; not a safety certification",
        (12, canvas.shape[0] - 12),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.42,
        (225, 225, 225),
        1,
    )
    return canvas
