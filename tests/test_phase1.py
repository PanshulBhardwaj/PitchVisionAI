import numpy as np
import pandas as pd
import pytest

from analytics.dataset import COLUMNS, box_to_row, build_dataframe, validate
from core.config import ProcessingConfig, safe_stem
from core.exceptions import ConfigError, DatasetValidationError, VideoError
from detection.detector import TrackedBox, parse_results
from scripts.make_synthetic_video import make
from video.ingestion import read_metadata
from video.processing import process_video


class StubTracker:
    """Deterministic fake: one box moving right with a stable ID, empty on frame 5."""

    def __init__(self):
        self.n = 0
        self.resets = 0

    def reset(self):
        self.resets += 1
        self.n = 0

    def update(self, frame):
        i, self.n = self.n, self.n + 1
        if i == 5:
            return []
        return [TrackedBox(1, 0.9, 10 + i, 20, 50 + i, 100)]


def test_metadata_and_errors(tmp_path):
    v = make(tmp_path / "a.mp4", frames=10, fps=25)
    m = read_metadata(v)
    assert (m.width, m.height, m.fps, m.frame_count) == (640, 360, 25.0, 10)
    assert m.duration_seconds == pytest.approx(0.4)
    with pytest.raises(VideoError):
        read_metadata(tmp_path / "missing.mp4")
    empty = tmp_path / "e.mp4"
    empty.write_bytes(b"")
    with pytest.raises(VideoError):
        read_metadata(empty)
    bad = tmp_path / "bad.mp4"
    bad.write_bytes(b"not a video")
    with pytest.raises(VideoError):
        read_metadata(bad)
    txt = tmp_path / "x.txt"
    txt.write_text("hi")
    with pytest.raises(VideoError):
        read_metadata(txt)


def test_pipeline_with_stub(tmp_path):
    v = make(tmp_path / "clip.mp4", frames=12, fps=25)
    t = StubTracker()
    out = tmp_path / "out"
    res = process_video(v, t, ProcessingConfig(), out)
    assert t.resets == 1
    assert res.frames_read == 12 and res.detections == 11  # frame 5 empty
    assert res.annotated_video.exists() and res.tracking_csv.exists()
    assert not list(out.glob("*.part*"))
    df = pd.read_csv(res.tracking_csv)
    assert list(df.columns) == COLUMNS
    assert df["timestamp_seconds"].iloc[1] == pytest.approx(1 / 25)
    assert df["pitch_x_meters"].isna().all()  # no calibration -> null
    # annotated video decodes
    assert read_metadata(res.annotated_video).frame_count == 12
    assert v.exists()  # input untouched


def test_stride_and_cancel(tmp_path):
    v = make(tmp_path / "c.mp4", frames=10)
    res = process_video(v, StubTracker(), ProcessingConfig(frame_stride=2), tmp_path / "o")
    assert res.frames_inferred == 5
    with pytest.raises(InterruptedError):
        process_video(v, StubTracker(), ProcessingConfig(), tmp_path / "o2", should_cancel=lambda: True)
    assert not list((tmp_path / "o2").glob("*_annotated.mp4"))  # incomplete never looks finished


def test_validation():
    good = build_dataframe([box_to_row("v", 0, 25, TrackedBox(1, 0.5, 0, 0, 10, 20))])
    validate(good, 100, 100)
    dup = pd.concat([good, good], ignore_index=True)
    with pytest.raises(DatasetValidationError):
        validate(dup)
    bad = good.copy()
    bad["bounding_box_x2"] = bad["bounding_box_x1"]
    with pytest.raises(DatasetValidationError):
        validate(bad)
    nan_id = good.copy().astype({"track_id": float})
    nan_id["track_id"] = np.nan
    with pytest.raises(DatasetValidationError):
        validate(nan_id)
    with pytest.raises(DatasetValidationError):
        validate(good, 5, 5)  # box outside frame


def test_parse_results_empty_and_idless():
    assert parse_results(None) == []

    class R:
        boxes = None

    assert parse_results(R()) == []


def test_config_and_names():
    with pytest.raises(ConfigError):
        ProcessingConfig(confidence=1.5).validate()
    with pytest.raises(ConfigError):
        ProcessingConfig(frame_stride=0).validate()
    assert safe_stem("../../evil name!.mp4") == "evil_name"


def test_tracker_always_persists(monkeypatch):
    """Regression: persist=False on the first call made Ultralytics rebuild the tracker every frame."""
    import sys
    import types

    calls = []

    class FakeYOLO:
        def __init__(self, w):
            self.predictor = None

        def track(self, frame, **kw):
            calls.append(kw["persist"])
            return []

    monkeypatch.setitem(sys.modules, "ultralytics", types.SimpleNamespace(YOLO=FakeYOLO))
    from detection.detector import YoloByteTrack

    t = YoloByteTrack(ProcessingConfig(device="cpu"))
    t.reset()
    for _ in range(3):
        t.update(np.zeros((64, 64, 3), np.uint8))
    assert calls == [True, True, True]
