"""Spatial filter that drops detections whose foot point lies outside the pitch polygon.

The polygon is stored in *normalised* coordinates (0..1 of frame width/height) so it is
independent of the video resolution and of any --max-width downscaling.
"""

from __future__ import annotations

import cv2
import numpy as np

from core.exceptions import ConfigError
from detection.detector import TrackedBox

Polygon = tuple[tuple[float, float], ...]


def parse_polygon(text: str) -> Polygon:
    """Parse 'x,y;x,y;x,y;...' (normalised 0..1) into a polygon. Raises ConfigError if malformed."""
    try:
        pts = tuple(
            (float(x), float(y)) for x, y in (pair.split(",") for pair in text.split(";") if pair.strip())
        )
    except ValueError as exc:
        raise ConfigError(f"Bad pitch polygon {text!r}: expected 'x,y;x,y;x,y' with numbers.") from exc
    validate_polygon(pts)
    return pts


def validate_polygon(poly: Polygon) -> None:
    if len(poly) < 3:
        raise ConfigError(f"Pitch polygon needs at least 3 points, got {len(poly)}.")
    if any(not (0.0 <= x <= 1.0 and 0.0 <= y <= 1.0) for x, y in poly):
        raise ConfigError("Pitch polygon points must be normalised to the range 0..1.")


class PitchFilter:
    def __init__(self, polygon: Polygon) -> None:
        validate_polygon(polygon)
        self.polygon = polygon

    def _pixels(self, size: tuple[int, int]) -> np.ndarray:
        w, h = size
        return np.array([(x * w, y * h) for x, y in self.polygon], dtype=np.float32)

    def keep(self, box: TrackedBox, size: tuple[int, int]) -> bool:
        """True if the box's foot point (bottom-centre) is inside or on the polygon edge."""
        foot = (float((box.x1 + box.x2) / 2), float(box.y2))
        return cv2.pointPolygonTest(self._pixels(size).reshape(-1, 1, 2), foot, False) >= 0

    def filter(self, boxes: list[TrackedBox], size: tuple[int, int]) -> list[TrackedBox]:
        return [b for b in boxes if self.keep(b, size)]

    def draw(self, frame: np.ndarray, size: tuple[int, int]) -> None:
        """Outline the pitch area in place (for visually checking the polygon)."""
        cv2.polylines(frame, [self._pixels(size).astype(np.int32)], True, (0, 255, 255), 1, cv2.LINE_AA)
