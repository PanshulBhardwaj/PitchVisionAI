import pytest

from core.config import ProcessingConfig
from core.exceptions import ConfigError
from detection.detector import TrackedBox
from detection.pitch_filter import PitchFilter, parse_polygon
from scripts.make_synthetic_video import make
from video.processing import process_video

SIZE = (640, 360)
POLY = parse_polygon("0.1,0.3;0.9,0.3;0.95,0.95;0.05,0.95")


def box(cx, foot_y, track_id=1):
    return TrackedBox(track_id, 0.9, cx - 10, foot_y - 40, cx + 10, foot_y)


def test_parse_polygon_ok_and_errors():
    assert len(POLY) == 4
    for bad in ("0.1,0.1;0.5,0.5", "a,b;c,d;e,f", "0.1,0.1;2,0.5;0.5,0.9"):
        with pytest.raises(ConfigError):
            parse_polygon(bad)


def test_foot_point_decides():
    f = PitchFilter(POLY)
    assert f.keep(box(320, 250), SIZE)  # on the pitch
    assert not f.keep(box(320, 60), SIZE)  # crowd/stand above the touchline
    # head is above the line but feet are inside: kept (foot point, not box centre)
    assert f.keep(TrackedBox(2, 0.9, 300, 90, 340, 200), SIZE)


def test_scales_with_frame_size():
    f = PitchFilter(POLY)
    assert f.keep(box(640, 500), (1280, 720))
    assert not f.keep(box(640, 100), (1280, 720))


def test_pipeline_drops_and_reports(tmp_path):
    class Stub:
        def reset(self):
            pass

        def update(self, frame):  # one player on the pitch, one in the stands
            return [box(320, 250, 1), box(320, 60, 2)]

    v = make(tmp_path / "c.mp4", frames=6)
    cfg = ProcessingConfig(pitch_polygon=POLY)
    res = process_video(v, Stub(), cfg, tmp_path / "o")
    assert res.unique_tracks == 1 and res.detections == 6
    assert res.dropped_outside_pitch == 6
    assert res.median_people_per_frame == 1.0


def test_tracker_path_validation(tmp_path):
    ProcessingConfig(tracker="bytetrack.yaml").validate()  # built-in name
    with pytest.raises(ConfigError):
        ProcessingConfig(tracker=str(tmp_path / "missing.yaml")).validate()


def test_presets_and_overrides():
    from run_baseline import resolve_preset

    w, sz, tr = resolve_preset("accurate", None, None, None)
    assert w == "yolov8m.pt" and sz == 960 and tr.endswith("botsort_pitch.yaml")
    ProcessingConfig(model_weights=w, image_size=sz, tracker=tr).validate()  # shipped config exists
    assert resolve_preset("fast", None, None, None) == ("yolov8n.pt", 640, "bytetrack.yaml")
    assert resolve_preset("accurate", "x.pt", 1280, "t.yaml") == ("x.pt", 1280, "t.yaml")
