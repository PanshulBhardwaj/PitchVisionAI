"""Re-identification: give a player their old ID back when the tracker lost them and re-finds them.

The tracker (ByteTrack/BoT-SORT) hands out a brand-new raw ID whenever it loses a track for longer than
its buffer. This layer sits on top and maps raw IDs to *stable* IDs. When a new raw ID appears, it is
compared with identities that disappeared recently, using:
  * where the player should be now (last position + velocity, with a gate that grows with the gap),
  * box height (perspective-consistent size),
  * shirt colour (mean torso colour), so a yellow player cannot become a white one or the referee.
Matching is greedy lowest-cost-first and each lost identity can be claimed once per frame.

Limit: two team-mates who vanish together and return together near the same spot can be swapped.
Telling them apart reliably needs jersey-number reading or a learned ReID model.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from detection.detector import TrackedBox

MAX_COLOUR_DIST = 75.0  # BGR distance beyond which two torsos are "different kits"
HEIGHT_RATIO = (0.55, 1.8)
MAX_EXTRAPOLATE_FRAMES = 15
SPEED_BOX_HEIGHTS_PER_S = 3.0  # extra slack on top of the predicted position


@dataclass
class _Identity:
    sid: int
    last_frame: int
    foot: np.ndarray
    vel: np.ndarray  # pixels per video frame
    height: float
    feat: np.ndarray | None


class IdentityManager:
    def __init__(self, fps: float, max_gap_seconds: float = 6.0) -> None:
        self.fps = fps
        self.max_gap_frames = max(1, int(max_gap_seconds * fps))
        self.merges = 0  # how many times a lost identity was given back
        self.reset()

    def reset(self) -> None:
        self._raw_to_sid: dict[int, int] = {}
        self._ids: dict[int, _Identity] = {}
        self._next = 1

    def assign(
        self, frame_idx: int, boxes: list[TrackedBox], feats: list[np.ndarray | None]
    ) -> tuple[list[TrackedBox], list[np.ndarray | None]]:
        """Return (boxes with stable track IDs, their features). `feats[i]` is boxes[i]'s torso colour."""
        sid_of: dict[int, int] = {}
        seen: set[int] = set()
        new: list[int] = []
        for i, b in enumerate(boxes):
            sid = self._raw_to_sid.get(b.track_id)
            if sid is not None and sid not in seen:
                sid_of[i] = sid
                seen.add(sid)
            else:
                new.append(i)

        lost = [
            ident
            for ident in self._ids.values()
            if ident.sid not in seen and 0 < frame_idx - ident.last_frame <= self.max_gap_frames
        ]
        pairs = []
        for i in new:
            for ident in lost:
                cost = self._cost(frame_idx, boxes[i], feats[i], ident)
                if cost is not None:
                    pairs.append((cost, i, ident.sid))
        taken: set[int] = set()
        for _, i, sid in sorted(pairs):
            if i in sid_of or sid in taken:
                continue
            sid_of[i] = sid
            taken.add(sid)
            self._raw_to_sid[boxes[i].track_id] = sid
            self.merges += 1
        for i in new:
            if i not in sid_of:
                sid = self._next
                self._next += 1
                sid_of[i] = sid
                self._raw_to_sid[boxes[i].track_id] = sid

        out: list[TrackedBox] = []
        out_feats: list[np.ndarray | None] = []
        used: set[int] = set()
        for i, b in enumerate(boxes):
            sid = sid_of[i]
            if sid in used:  # same identity twice in one frame (tracker duplicate): keep the first
                continue
            used.add(sid)
            self._update(sid, frame_idx, b, feats[i])
            out.append(replace(b, track_id=sid))
            out_feats.append(feats[i])

        horizon = frame_idx - 2 * self.max_gap_frames  # forget identities nobody will match any more
        for sid in [s for s, v in self._ids.items() if v.last_frame < horizon]:
            del self._ids[sid]
        return out, out_feats

    def _cost(self, frame_idx, b: TrackedBox, feat, ident: _Identity) -> float | None:
        gap = frame_idx - ident.last_frame
        h = max(b.y2 - b.y1, 1.0)
        ratio = h / max(ident.height, 1.0)
        if not HEIGHT_RATIO[0] <= ratio <= HEIGHT_RATIO[1]:
            return None
        foot = np.array([(b.x1 + b.x2) / 2, b.y2])
        pred = ident.foot + ident.vel * min(gap, MAX_EXTRAPOLATE_FRAMES)
        radius = h * (0.8 + SPEED_BOX_HEIGHTS_PER_S * gap / self.fps)
        spatial = float(np.linalg.norm(foot - pred)) / radius
        if spatial > 1.0:
            return None
        app = 0.0
        if feat is not None and ident.feat is not None:
            d = float(np.linalg.norm(feat - ident.feat))
            if d > MAX_COLOUR_DIST:
                return None
            app = d / MAX_COLOUR_DIST
        return spatial + 0.5 * app

    def _update(self, sid: int, frame_idx: int, b: TrackedBox, feat) -> None:
        foot = np.array([(b.x1 + b.x2) / 2, b.y2], dtype=float)
        h = b.y2 - b.y1
        ident = self._ids.get(sid)
        if ident is None:
            self._ids[sid] = _Identity(sid, frame_idx, foot, np.zeros(2), h, feat)
            return
        gap = max(frame_idx - ident.last_frame, 1)
        if gap <= 5:  # velocity only from (nearly) consecutive observations
            ident.vel = 0.7 * ident.vel + 0.3 * (foot - ident.foot) / gap
        ident.foot, ident.last_frame = foot, frame_idx
        ident.height = 0.8 * ident.height + 0.2 * h
        if feat is not None:
            ident.feat = feat if ident.feat is None else 0.9 * ident.feat + 0.1 * feat
