"""Detector/tracker interface and the Ultralytics YOLO + ByteTrack backend."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Protocol

import numpy as np

from core.config import ProcessingConfig
from core.exceptions import DetectorError

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class TrackedBox:
    track_id: int
    confidence: float
    x1: float
    y1: float
    x2: float
    y2: float


class PersonTracker(Protocol):
    """Anything that turns a BGR frame into tracked person boxes."""

    def reset(self) -> None: ...
    def update(self, frame: np.ndarray) -> list[TrackedBox]: ...


class YoloByteTrack:
    """Ultralytics YOLO detection with its built-in ByteTrack tracker.

    Ultralytics is imported lazily so the rest of the package (and the unit
    tests) work without torch installed.
    """

    def __init__(self, cfg: ProcessingConfig) -> None:
        cfg.validate()
        self.cfg = cfg
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise DetectorError("ultralytics is not installed. Run: pip install -r requirements.txt") from exc
        try:
            self.model = YOLO(cfg.model_weights)
        except Exception as exc:  # download, corrupt weights, etc.
            raise DetectorError(
                f"Could not load weights '{cfg.model_weights}': {exc}. "
                "If this is a download failure, check your connection or place the .pt file "
                "in the project folder and set model_weights to its path."
            ) from exc
        self.device = self._resolve_device(cfg.device)
        log.info("Loaded %s on device=%s", cfg.model_weights, self.device)

    @staticmethod
    def _resolve_device(requested: str) -> str:
        if requested != "auto":
            return requested
        try:
            import torch

            return "0" if torch.cuda.is_available() else "cpu"
        except ImportError:
            return "cpu"

    def reset(self) -> None:
        """Clear tracker state; call at the start of every video."""
        predictor = getattr(self.model, "predictor", None)
        for t in getattr(predictor, "trackers", []) or []:
            t.reset()

    def update(self, frame: np.ndarray) -> list[TrackedBox]:
        try:
            results = self.model.track(
                frame,
                persist=True,  # must ALWAYS be True: Ultralytics binds this value on the first call
                tracker=self.cfg.tracker,
                conf=self.cfg.confidence,
                iou=self.cfg.iou,
                imgsz=self.cfg.image_size,
                classes=[self.cfg.person_class_id],
                device=self.device,
                half=self.device.startswith("cuda") or self.device.isdigit(),  # FP16 on GPU: faster
                verbose=False,
            )
        except Exception as exc:
            raise DetectorError(f"Inference failed: {exc}") from exc
        return parse_results(results[0] if results else None)


def parse_results(result) -> list[TrackedBox]:
    """Convert an Ultralytics result to TrackedBox list.

    Handles zero detections and detections without a track id (dropped, since
    ID-less boxes must not enter the analytics dataset).
    """
    if result is None or result.boxes is None or len(result.boxes) == 0:
        return []
    boxes = result.boxes
    if boxes.id is None:
        return []
    ids = boxes.id.int().cpu().tolist()
    conf = boxes.conf.cpu().tolist()
    xyxy = boxes.xyxy.cpu().numpy()
    return [TrackedBox(int(i), float(c), *map(float, b)) for i, c, b in zip(ids, conf, xyxy, strict=True)]
