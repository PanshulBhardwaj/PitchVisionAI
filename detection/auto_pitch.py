"""Automatic pitch region from the grass colour: follows the real pitch outline as the camera moves.

Per (cheap, low-res) update: green mask -> temporal smoothing -> close gaps left by players/lines ->
largest blob -> convex hull. Players whose foot point is outside the hull (crowd, staff, ad boards)
are dropped; a small tolerance keeps players standing on the touchline.
"""

from __future__ import annotations

import cv2
import numpy as np

from detection.detector import TrackedBox

WORK_WIDTH = 320
MIN_PITCH_FRACTION = 0.08  # hull smaller than this share of the frame = detection failed


class AutoPitch:
    def __init__(self, update_every: int = 3, tolerance: float = 0.06) -> None:
        self.update_every = update_every
        self.tolerance = tolerance  # fraction of box height allowed outside the hull
        self.hull: np.ndarray | None = None  # normalised (0..1), shape (N, 1, 2) float32
        self._acc: np.ndarray | None = None
        self._calls = 0

    def update(self, frame: np.ndarray) -> None:
        """Call on every inferred frame; recomputes the outline only every `update_every` calls."""
        self._calls += 1
        if self.hull is not None and (self._calls - 1) % self.update_every:
            return
        h, w = frame.shape[:2]
        sw = WORK_WIDTH
        sh = max(int(h * sw / w), 8)
        small = cv2.resize(frame, (sw, sh), interpolation=cv2.INTER_AREA)
        hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
        green = ((hsv[..., 0] >= 28) & (hsv[..., 0] <= 95) & (hsv[..., 1] >= 35) & (hsv[..., 2] >= 30))
        green = green.astype(np.float32)
        self._acc = green if self._acc is None or self._acc.shape != green.shape else 0.6 * self._acc + 0.4 * green
        mask = (self._acc >= 0.5).astype(np.uint8) * 255
        k = max(int(sw * 0.05), 3)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return
        hull = cv2.convexHull(max(contours, key=cv2.contourArea))
        if cv2.contourArea(hull) < MIN_PITCH_FRACTION * sw * sh:
            return  # keep the previous outline
        hull = cv2.approxPolyDP(hull, 0.01 * cv2.arcLength(hull, True), True)
        norm = hull.astype(np.float32).reshape(-1, 1, 2)
        norm[..., 0] /= sw
        norm[..., 1] /= sh
        self.hull = norm

    def _pixels(self, size: tuple[int, int]) -> np.ndarray:
        w, h = size
        return (self.hull * np.array([w, h], dtype=np.float32)).astype(np.float32)

    def filter(self, boxes: list[TrackedBox], size: tuple[int, int]) -> list[TrackedBox]:
        if self.hull is None:
            return boxes
        poly = self._pixels(size)
        out = []
        for b in boxes:
            foot = (float((b.x1 + b.x2) / 2), float(b.y2))
            dist = cv2.pointPolygonTest(poly, foot, True)  # >0 inside, <0 outside (pixels)
            if dist >= -self.tolerance * (b.y2 - b.y1):
                out.append(b)
        return out

    def draw(self, frame: np.ndarray, size: tuple[int, int]) -> None:
        if self.hull is not None:
            cv2.polylines(frame, [self._pixels(size).astype(np.int32)], True, (0, 255, 255), 1, cv2.LINE_AA)
