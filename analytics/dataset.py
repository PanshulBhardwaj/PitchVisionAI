"""Tracking dataset construction, validation and export.

Coordinate convention: image coordinates have their origin at the top-left,
x to the right, y downward, in pixels of the *original* video frame.
pitch_x_meters / pitch_y_meters stay null until a valid calibration exists.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from core.exceptions import DatasetValidationError
from detection.detector import TrackedBox

log = logging.getLogger(__name__)

COLUMNS = [
    "video_id",
    "frame_number",
    "timestamp_seconds",
    "track_id",
    "detection_confidence",
    "bounding_box_x1",
    "bounding_box_y1",
    "bounding_box_x2",
    "bounding_box_y2",
    "center_x",
    "center_y",
    "foot_point_x",
    "foot_point_y",
    "team_label",
    "pitch_x_meters",
    "pitch_y_meters",
]


def box_to_row(video_id: str, frame_number: int, fps: float, b: TrackedBox) -> dict:
    return {
        "video_id": video_id,
        "frame_number": frame_number,
        "timestamp_seconds": frame_number / fps,
        "track_id": b.track_id,
        "detection_confidence": b.confidence,
        "bounding_box_x1": b.x1,
        "bounding_box_y1": b.y1,
        "bounding_box_x2": b.x2,
        "bounding_box_y2": b.y2,
        "center_x": (b.x1 + b.x2) / 2,
        "center_y": (b.y1 + b.y2) / 2,
        "foot_point_x": (b.x1 + b.x2) / 2,  # bottom-center approximates ground contact
        "foot_point_y": b.y2,
        "team_label": None,
        "pitch_x_meters": np.nan,
        "pitch_y_meters": np.nan,
    }


def build_dataframe(rows: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(rows, columns=COLUMNS)
    df["team_label"] = df["team_label"].astype("object")
    return df


def validate(df: pd.DataFrame, frame_w: int | None = None, frame_h: int | None = None) -> None:
    """Raise DatasetValidationError on any schema or sanity violation."""
    missing = [c for c in COLUMNS if c not in df.columns]
    if missing:
        raise DatasetValidationError(f"Missing columns: {missing}")
    if df.empty:
        return
    core = [
        "video_id",
        "frame_number",
        "timestamp_seconds",
        "track_id",
        "detection_confidence",
        "bounding_box_x1",
        "bounding_box_y1",
        "bounding_box_x2",
        "bounding_box_y2",
    ]
    if df[core].isna().any().any():
        raise DatasetValidationError("Null values in required columns (e.g. missing track_id).")
    if (df["track_id"] < 0).any():
        raise DatasetValidationError("Negative track_id present.")
    if (df["timestamp_seconds"] < 0).any() or (df["frame_number"] < 0).any():
        raise DatasetValidationError("Negative frame number or timestamp.")
    if not df["detection_confidence"].between(0, 1).all():
        raise DatasetValidationError("detection_confidence outside [0, 1].")
    if (
        (df["bounding_box_x2"] <= df["bounding_box_x1"]) | (df["bounding_box_y2"] <= df["bounding_box_y1"])
    ).any():
        raise DatasetValidationError("Degenerate bounding box (x2<=x1 or y2<=y1).")
    if frame_w is not None and frame_h is not None:
        tol = 2.0
        if (
            (df["bounding_box_x1"] < -tol)
            | (df["bounding_box_y1"] < -tol)
            | (df["bounding_box_x2"] > frame_w + tol)
            | (df["bounding_box_y2"] > frame_h + tol)
        ).any():
            raise DatasetValidationError("Bounding box outside frame bounds.")
    if df.duplicated(["video_id", "frame_number", "track_id"]).any():
        raise DatasetValidationError("Duplicate (video_id, frame_number, track_id) records.")
    # timestamps must be non-decreasing in frame order
    ordered = df.sort_values("frame_number")["timestamp_seconds"]
    if not ordered.is_monotonic_increasing:
        raise DatasetValidationError("Timestamps do not increase with frame number.")


def export_csv(df: pd.DataFrame, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".part")
    df.to_csv(tmp, index=False)
    tmp.replace(path)  # atomic: never leave a half-written CSV under the final name
    return path


def export_parquet_if_available(df: pd.DataFrame, path: Path) -> Path | None:
    try:
        df.to_parquet(path, index=False)
        return path
    except ImportError:
        log.info("pyarrow/fastparquet not installed; skipping Parquet export.")
        return None
