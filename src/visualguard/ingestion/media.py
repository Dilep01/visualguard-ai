import math
from io import BytesIO
from pathlib import Path

import cv2
import imageio_ffmpeg
import numpy as np
from PIL import Image, UnidentifiedImageError

from visualguard.settings import Settings


def read_image(path: Path, settings: Settings) -> np.ndarray:
    try:
        with Image.open(BytesIO(path.read_bytes())) as image:
            if image.width * image.height > settings.max_pixels:
                raise ValueError("Image exceeds the decoded pixel limit")
            image.verify()
        frame = cv2.imread(str(path))
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError) as exc:
        raise ValueError("Invalid or unsupported image") from exc
    if frame is None:
        raise ValueError("Image could not be decoded")
    return frame


def video_info(path: Path, settings: Settings) -> dict:
    capture = cv2.VideoCapture(str(path))
    try:
        fps = capture.get(cv2.CAP_PROP_FPS)
        count = capture.get(cv2.CAP_PROP_FRAME_COUNT)
        width = capture.get(cv2.CAP_PROP_FRAME_WIDTH)
        height = capture.get(cv2.CAP_PROP_FRAME_HEIGHT)
        if not capture.isOpened() or not all(
            math.isfinite(v) and v > 0 for v in (fps, count, width, height)
        ):
            raise ValueError("Video has invalid duration, dimensions, or frame rate")
        if width * height > settings.max_pixels:
            raise ValueError("Video exceeds the decoded pixel limit")
        if count / fps > settings.max_video_seconds:
            raise ValueError(f"Video exceeds {settings.max_video_seconds:g} seconds")
        return {
            "fps": fps,
            "frames": int(count),
            "width": int(width),
            "height": int(height),
            "duration_seconds": count / fps,
        }
    finally:
        capture.release()


def sampled_frames(path: Path, settings: Settings):
    info = video_info(path, settings)
    stride = max(1, round(info["fps"] / settings.sample_fps))
    capture = cv2.VideoCapture(str(path))
    index = 0
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            if index / info["fps"] >= settings.max_video_seconds:
                raise ValueError("Decoded video exceeded duration limit")
            if frame.shape[0] * frame.shape[1] > settings.max_pixels:
                raise ValueError("Decoded frame exceeded pixel limit")
            if index % stride == 0:
                yield (
                    index / info["fps"],
                    frame,
                    min(stride / info["fps"], info["duration_seconds"] - index / info["fps"]),
                )
            index += 1
        if index < info["frames"] - 1:
            raise ValueError("Video ended before its declared frame count; file may be truncated")
    finally:
        capture.release()


def video_writer(path: Path, width: int, height: int, fps: float):
    writer = imageio_ffmpeg.write_frames(
        str(path),
        (width, height),
        fps=fps,
        codec="libx264",
        pix_fmt_in="bgr24",
        pix_fmt_out="yuv420p",
        macro_block_size=2,
        ffmpeg_log_level="error",
        output_params=["-movflags", "+faststart"],
    )
    writer.send(None)
    return writer
