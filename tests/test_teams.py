import cv2
import numpy as np
import pandas as pd

from analytics.teams import TeamClassifier, torso_feature
from core.config import ProcessingConfig
from detection.detector import TrackedBox
from scripts.make_synthetic_video import make
from video.processing import annotate_tactical, process_video


def _scene():
    frame = np.full((300, 400, 3), (60, 140, 50), np.uint8)
    boxes = []
    kits = [(235, 235, 235)] * 5 + [(40, 210, 230)] * 5 + [(120, 120, 30)]
    for i, kit in enumerate(kits):
        x, y = 20 + 33 * i, 200
        cv2.rectangle(frame, (x, y - 56), (x + 22, y - 20), kit, -1)
        boxes.append(TrackedBox(i + 1, 0.9, x, y - 56, x + 22, y))
    return frame, boxes


def test_two_teams_and_other():
    frame, boxes = _scene()
    clf = TeamClassifier(min_samples=11)
    for b in boxes:
        clf.observe(b.track_id, torso_feature(frame, b))
    lab = clf.final_labels()
    assert len({lab[i] for i in range(1, 6)}) == 1 and len({lab[i] for i in range(6, 11)}) == 1
    assert lab[1] != lab[6] and lab[11] == "other"


def test_tactical_draws_and_empty_ok():
    frame, boxes = _scene()
    before = frame.copy()
    annotate_tactical(frame, boxes)
    assert (frame != before).any()
    annotate_tactical(frame, [])  # no boxes: no crash


def test_pipeline_tactical_writes_team_labels(tmp_path):
    class Stub:
        def reset(self):
            pass

        def update(self, frame):
            return [TrackedBox(i, 0.9, 20 + 25 * i, 100, 42 + 25 * i, 156) for i in range(1, 6)]

    v = make(tmp_path / "t.mp4", frames=30)
    res = process_video(v, Stub(), ProcessingConfig(style="tactical", detect_teams=True), tmp_path / "o")
    df = pd.read_csv(res.tracking_csv)
    assert "team_label" in df.columns and res.annotated_video.exists()
