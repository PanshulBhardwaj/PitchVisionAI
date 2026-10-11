"""Create a synthetic pipeline-test video (moving rectangles on green).

NOT football footage and NOT a substitute for real evaluation. It only
verifies decoding, writing and I/O when no real sample is available.
"""

import sys
from pathlib import Path

import cv2
import numpy as np


def make(path: Path, frames: int = 60, fps: int = 25, w: int = 640, h: int = 360) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    wr = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    if not wr.isOpened():
        raise RuntimeError("mp4v writer unavailable")
    for i in range(frames):
        img = np.full((h, w, 3), (50, 140, 50), np.uint8)
        for k in range(3):
            x = 40 + k * 150 + i * 3
            cv2.rectangle(img, (x, 120 + k * 30), (x + 30, 200 + k * 30), (200, 60 + 50 * k, 60), -1)
        wr.write(img)
    wr.release()
    return path


if __name__ == "__main__":
    print(make(Path(sys.argv[1] if len(sys.argv) > 1 else "sample_data/synthetic_test.mp4")))
