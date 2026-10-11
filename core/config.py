"""Configuration objects. No magic numbers elsewhere in the codebase."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

from core.exceptions import ConfigError

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SAFE_NAME = re.compile(r"[^A-Za-z0-9._-]+")


def safe_stem(name: str) -> str:
    """Return a filesystem-safe stem derived from an arbitrary file name."""
    stem = SAFE_NAME.sub("_", Path(name).stem).strip("._")
    return stem[:80] or "video"


@dataclass(frozen=True)
class ProcessingConfig:
    model_weights: str = "yolov8n.pt"  # auto-downloaded by Ultralytics on first use
    confidence: float = 0.25
    iou: float = 0.5
    image_size: int = 640  # inference size (letterboxed by the model)
    device: str = "auto"  # "auto" -> GPU if available else CPU
    tracker: str = "bytetrack.yaml"
    frame_stride: int = 1  # run inference on every Nth frame
    max_frames: int | None = None  # useful for smoke tests
    person_class_id: int = 0  # COCO "person"
    max_width: int | None = None  # downscale before inference/annotation
    style: str = "boxes"  # "boxes" (debug) or "tactical" (ellipses at the feet, Wyscout-like)
    detect_teams: bool = False  # cluster shirt colours into teams A/B/other
    reid: bool = False  # give lost players their old ID back when the tracker re-finds them
    reid_max_gap_seconds: float = 6.0
    auto_pitch: bool = False  # detect the pitch from grass colour (ignored if pitch_polygon is set)
    pitch_polygon: tuple[tuple[float, float], ...] | None = None  # normalised (0..1) pitch area; None = off

    def validate(self) -> None:
        if not 0.0 < self.confidence < 1.0:
            raise ConfigError(f"confidence must be in (0, 1), got {self.confidence}")
        if not 0.0 < self.iou <= 1.0:
            raise ConfigError(f"iou must be in (0, 1], got {self.iou}")
        if self.image_size < 32:
            raise ConfigError(f"image_size must be >= 32, got {self.image_size}")
        if self.frame_stride < 1:
            raise ConfigError(f"frame_stride must be >= 1, got {self.frame_stride}")
        if self.max_frames is not None and self.max_frames < 1:
            raise ConfigError("max_frames must be >= 1 when set")
        if self.max_width is not None and self.max_width < 64:
            raise ConfigError("max_width must be >= 64 when set")
        if self.reid_max_gap_seconds <= 0:
            raise ConfigError("reid_max_gap_seconds must be > 0")
        if self.style not in ("boxes", "tactical"):
            raise ConfigError(f"style must be 'boxes' or 'tactical', got {self.style!r}")
        if self.pitch_polygon is not None:
            if len(self.pitch_polygon) < 3:
                raise ConfigError("pitch_polygon needs at least 3 points")
            if any(not (0.0 <= x <= 1.0 and 0.0 <= y <= 1.0) for x, y in self.pitch_polygon):
                raise ConfigError("pitch_polygon points must be normalised to 0..1")
        t = Path(self.tracker)
        # Bare names ("bytetrack.yaml") resolve to Ultralytics built-ins; anything with a directory must exist.
        if t.suffix in (".yaml", ".yml") and t.name != self.tracker and not t.is_file():
            raise ConfigError(f"Tracker config not found: {self.tracker}")


@dataclass(frozen=True)
class PathsConfig:
    data_dir: Path = field(default_factory=lambda: PROJECT_ROOT / "data")

    @property
    def uploads(self) -> Path:
        return self.data_dir / "uploads"

    @property
    def outputs(self) -> Path:
        return self.data_dir / "outputs"
