"""Video validation and metadata extraction."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from core.exceptions import VideoError

log = logging.getLogger(__name__)
SUPPORTED_SUFFIXES = {".mp4", ".mov", ".avi", ".mkv", ".m4v"}


@dataclass(frozen=True)
class VideoMetadata:
    path: Path
    width: int
    height: int
    fps: float
    frame_count: int
    duration_seconds: float


def read_metadata(path: str | Path) -> VideoMetadata:
    """Open a video, read metadata, and verify the first frame decodes."""
    p = Path(path)
    if not p.is_file():
        raise VideoError(f"Video file not found: {p}")
    if p.stat().st_size == 0:
        raise VideoError(f"Video file is empty: {p}")
    if p.suffix.lower() not in SUPPORTED_SUFFIXES:
        raise VideoError(f"Unsupported extension '{p.suffix}'. Supported: {sorted(SUPPORTED_SUFFIXES)}")
    cap = cv2.VideoCapture(str(p))
    try:
        if not cap.isOpened():
            raise VideoError(
                f"OpenCV could not open {p}. The file may be corrupted or use an unsupported codec."
            )
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        ok, _ = cap.read()
        if not ok:
            raise VideoError(f"No decodable frames in {p}.")
        if width <= 0 or height <= 0:
            raise VideoError(f"Invalid frame size {width}x{height} in {p}.")
        if fps <= 0 or np.isnan(fps):
            raise VideoError(f"Invalid frame rate ({fps}) in {p}; timestamps cannot be computed.")
        if frames <= 0:
            raise VideoError(f"Frame count unavailable for {p}.")
        return VideoMetadata(p, width, height, fps, frames, frames / fps)
    finally:
        cap.release()
