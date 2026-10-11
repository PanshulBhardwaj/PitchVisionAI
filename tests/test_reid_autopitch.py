import cv2
import numpy as np
import pandas as pd

from core.config import ProcessingConfig
from detection.auto_pitch import AutoPitch
from detection.detector import TrackedBox
from detection.identity import IdentityManager
from scripts.make_synthetic_video import make
from video.processing import process_video

WHITE = np.array([235, 235, 235], np.float32)
YELLOW = np.array([40, 210, 230], np.float32)


def box(raw, cx, foot, h=60):
    return TrackedBox(raw, 0.9, cx - 12, foot - h, cx + 12, foot)


def run(im, steps):
    """steps: list of (frame_idx, [(box, feat)])"""
    out = {}
    for f, items in steps:
        bs, ft = im.assign(f, [b for b, _ in items], [x for _, x in items])
        out[f] = [b.track_id for b in bs]
    return out


def test_player_returns_with_new_raw_id_gets_old_id():
    im = IdentityManager(fps=25)
    seq = [(f, [(box(1, 100 + 2 * f, 300), WHITE)]) for f in range(0, 20)]
    seq += [(f, [(box(7, 100 + 2 * f, 300), WHITE)]) for f in range(60, 70)]  # gone 40 frames, new raw id
    ids = run(im, seq)
    assert ids[5] == [1] and ids[65] == [1] and im.merges == 1


def test_different_kit_or_far_away_is_new_identity():
    im = IdentityManager(fps=25)
    seq = [(f, [(box(1, 100, 300), WHITE)]) for f in range(0, 10)]
    seq += [(30, [(box(2, 100, 300), YELLOW)])]  # same place, other team
    seq += [(31, [(box(3, 1500, 300), WHITE)])]  # same kit, impossibly far
    ids = run(im, seq)
    assert ids[30] != [1] and ids[31] not in ([1], ids[30]) and im.merges == 0


def test_two_lost_two_returning_matched_by_position():
    im = IdentityManager(fps=25)
    seq = [(f, [(box(1, 100, 300), WHITE), (box(2, 600, 300), WHITE)]) for f in range(10)]
    seq += [(40, [(box(9, 610, 300), WHITE), (box(8, 105, 300), WHITE)])]  # returned in swapped order
    ids = run(im, seq)
    assert ids[40] == [2, 1]


def test_gone_too_long_is_new():
    im = IdentityManager(fps=25, max_gap_seconds=2)
    seq = [(f, [(box(1, 100, 300), WHITE)]) for f in range(10)] + [(200, [(box(5, 100, 300), WHITE)])]
    assert run(im, seq)[200] == [2]


def _pitch_frame():
    f = np.full((360, 640, 3), (90, 90, 90), np.uint8)  # grey stands
    cv2.rectangle(f, (0, 120), (639, 359), (50, 150, 60), -1)  # pitch from y=120 down
    for x in range(40, 600, 90):
        cv2.rectangle(f, (x, 200), (x + 12, 250), (240, 240, 240), -1)  # white players on grass
    return f


def test_auto_pitch_keeps_pitch_drops_stands():
    ap = AutoPitch()
    ap.update(_pitch_frame())
    assert ap.hull is not None
    on, stand = box(1, 320, 300), box(2, 320, 80)
    kept = ap.filter([on, stand], (640, 360))
    assert [b.track_id for b in kept] == [1]
    assert len(ap.filter([box(1, 640, 600)], (1280, 720))) == 1  # hull is resolution-independent


def test_auto_pitch_no_grass_keeps_everything():
    ap = AutoPitch()
    ap.update(np.full((360, 640, 3), 90, np.uint8))
    assert ap.hull is None and len(ap.filter([box(1, 320, 300)], (640, 360))) == 1


def test_pipeline_reid_merges_raw_ids(tmp_path):
    class Stub:
        n = 0

        def reset(self):
            self.n = 0

        def update(self, frame):
            i, self.n = self.n, self.n + 1
            if 10 <= i < 20:
                return []  # lost
            raw = 1 if i < 10 else 2  # tracker issues a new raw id on return
            return [TrackedBox(raw, 0.9, 100 + i, 100, 124 + i, 160)]

    v = make(tmp_path / "r.mp4", frames=30)
    off = process_video(v, Stub(), ProcessingConfig(), tmp_path / "a")
    on = process_video(v, Stub(), ProcessingConfig(reid=True), tmp_path / "b")
    assert off.unique_tracks == 2 and on.unique_tracks == 1 and on.identity_merges == 1
    assert pd.read_csv(on.tracking_csv)["track_id"].nunique() == 1
