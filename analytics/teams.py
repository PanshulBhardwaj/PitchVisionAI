"""Unsupervised team assignment from shirt colour (no training data needed).

Torso pixels (grass removed) are averaged per box. After a warm-up the samples are clustered
with k-means (k=3): the two largest clusters become teams "A" and "B", the smallest becomes
"other" (referee / goalkeepers). Labels are smoothed per track by majority vote.
"""

from __future__ import annotations

from collections import Counter, deque

import cv2
import numpy as np

from detection.detector import TrackedBox

TEAM_LABELS = ("A", "B", "other")


def torso_feature(frame: np.ndarray, box: TrackedBox) -> np.ndarray | None:
    """Mean BGR colour of the upper-body region, ignoring green pitch pixels. None if unusable."""
    h, w = frame.shape[:2]
    bw, bh = box.x2 - box.x1, box.y2 - box.y1
    x1, x2 = int(max(0, box.x1 + 0.25 * bw)), int(min(w, box.x1 + 0.75 * bw))
    y1, y2 = int(max(0, box.y1 + 0.15 * bh)), int(min(h, box.y1 + 0.55 * bh))
    if x2 - x1 < 2 or y2 - y1 < 2:
        return None
    crop = frame[y1:y2, x1:x2]
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    grass = (hsv[..., 0] > 30) & (hsv[..., 0] < 90) & (hsv[..., 1] > 50)
    keep = ~grass
    if keep.sum() < 8:
        return None
    return crop[keep].mean(axis=0).astype(np.float32)


class TeamClassifier:
    def __init__(self, min_samples: int = 60, history: int = 30) -> None:
        self.min_samples = min_samples
        self.history = history
        self.colors: dict[str, tuple[int, int, int]] = {}  # label -> cluster-mean BGR
        self._centers: np.ndarray | None = None
        self._order: dict[int, str] = {}
        self._samples: list[np.ndarray] = []
        self._pending: dict[int, list[np.ndarray]] = {}
        self._recent: dict[int, deque[str]] = {}
        self._totals: dict[int, Counter] = {}

    @property
    def ready(self) -> bool:
        return self._centers is not None

    def observe(self, track_id: int, feat: np.ndarray | None) -> None:
        if feat is None:
            return
        if self._centers is None:
            self._samples.append(feat)
            self._pending.setdefault(track_id, []).append(feat)
            if len(self._samples) >= self.min_samples:
                self._fit()
        else:
            self._vote(track_id, self._predict(feat))

    def label(self, track_id: int) -> str | None:
        votes = self._recent.get(track_id)
        return Counter(votes).most_common(1)[0][0] if votes else None

    def final_labels(self) -> dict[int, str]:
        return {t: c.most_common(1)[0][0] for t, c in self._totals.items() if c}

    def _fit(self) -> None:
        data = np.asarray(self._samples, dtype=np.float32)
        crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 50, 0.5)
        _, labels, centers = cv2.kmeans(data, 3, None, crit, 5, cv2.KMEANS_PP_CENTERS)
        sizes = np.bincount(labels.ravel(), minlength=3)
        ranked = np.argsort(-sizes)  # biggest first -> A, B, other
        self._order = {int(c): TEAM_LABELS[i] for i, c in enumerate(ranked)}
        self._centers = centers
        self.colors = {self._order[i]: tuple(int(v) for v in centers[i]) for i in range(3)}
        for tid, feats in self._pending.items():  # back-fill tracks seen during warm-up
            for f in feats:
                self._vote(tid, self._predict(f))
        self._pending.clear()
        self._samples.clear()

    def _predict(self, feat: np.ndarray) -> str:
        d = np.linalg.norm(self._centers - feat, axis=1)
        return self._order[int(d.argmin())]

    def _vote(self, track_id: int, lab: str) -> None:
        self._recent.setdefault(track_id, deque(maxlen=self.history)).append(lab)
        self._totals.setdefault(track_id, Counter())[lab] += 1
