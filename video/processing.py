"""Sequential video pipeline: decode -> detect/track -> annotate -> write -> CSV."""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from analytics.teams import TeamClassifier, torso_feature
from analytics.dataset import box_to_row, build_dataframe, export_csv, validate
from core.config import ProcessingConfig, safe_stem
from core.exceptions import OutputWriteError, VideoError
from detection.detector import PersonTracker, TrackedBox
from detection.auto_pitch import AutoPitch
from detection.identity import IdentityManager
from detection.pitch_filter import PitchFilter
from video.ingestion import VideoMetadata, read_metadata

log = logging.getLogger(__name__)
ProgressFn = Callable[[int, int, float], None]  # (frames_done, total, elapsed_s)


@dataclass(frozen=True)
class ProcessingResult:
    video_id: str
    annotated_video: Path
    tracking_csv: Path
    frames_read: int
    frames_inferred: int
    detections: int
    unique_tracks: int
    elapsed_seconds: float
    dropped_outside_pitch: int = 0
    median_people_per_frame: float = 0.0
    identity_merges: int = 0

    @property
    def throughput_fps(self) -> float:
        return self.frames_read / self.elapsed_seconds if self.elapsed_seconds else 0.0


def _color_for(track_id: int) -> tuple[int, int, int]:
    rng = np.random.default_rng(track_id * 7919)
    return tuple(int(v) for v in rng.integers(60, 255, 3))


def annotate(frame: np.ndarray, boxes: list[TrackedBox]) -> None:
    """Draw boxes and IDs in place."""
    for b in boxes:
        c = _color_for(b.track_id)
        p1, p2 = (int(b.x1), int(b.y1)), (int(b.x2), int(b.y2))
        cv2.rectangle(frame, p1, p2, c, 2)
        label = f"ID {b.track_id}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(frame, (p1[0], p1[1] - th - 6), (p1[0] + tw + 4, p1[1]), c, -1)
        cv2.putText(
            frame, label, (p1[0] + 2, p1[1] - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA
        )


def annotate_tactical(
    frame: np.ndarray,
    boxes: list[TrackedBox],
    labels: dict[int, str | None] | None = None,
    team_colors: dict[str, tuple[int, int, int]] | None = None,
) -> None:
    """Wyscout-style: translucent ellipse under each player's feet plus a small label, no boxes."""
    if not boxes:
        return
    overlay = frame.copy()
    shapes = []
    for b in boxes:
        w = max(b.x2 - b.x1, 8.0)
        centre = (int((b.x1 + b.x2) / 2), int(b.y2))
        axes = (max(int(w * 1.0), 6), max(int(w * 0.34), 3))
        lab = (labels or {}).get(b.track_id)
        ring = (team_colors or {}).get(lab) if lab != "other" else (170, 170, 170)
        ring = tuple(int(min(255, v * 1.15 + 20)) for v in ring) if ring else (255, 255, 255)
        shapes.append((centre, axes, ring, b.track_id))
        cv2.ellipse(overlay, centre, axes, 0, 0, 360, (255, 255, 255), -1, cv2.LINE_AA)
    cv2.addWeighted(overlay, 0.35, frame, 0.65, 0, dst=frame)
    for centre, axes, ring, tid in shapes:
        cv2.ellipse(frame, centre, axes, 0, 0, 360, ring, 2, cv2.LINE_AA)
        pos = (centre[0] + axes[0] + 4, centre[1] + 4)
        text = str(tid)
        cv2.putText(frame, text, pos, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(frame, text, pos, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)


def _fourcc_candidates() -> list[str]:
    return ["mp4v", "avc1", "XVID"]


def _open_writer(path: Path, fps: float, size: tuple[int, int]) -> cv2.VideoWriter:
    for code in _fourcc_candidates():
        w = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*code), fps, size)
        if w.isOpened():
            return w
        w.release()
    raise OutputWriteError(f"Could not create a video writer for {path} (tried {_fourcc_candidates()}).")


def process_video(
    input_path: str | Path,
    tracker: PersonTracker,
    cfg: ProcessingConfig,
    out_dir: str | Path,
    progress: ProgressFn | None = None,
    should_cancel: Callable[[], bool] | None = None,
) -> ProcessingResult:
    """Process one video. Never modifies the input; outputs go to out_dir."""
    cfg.validate()
    meta: VideoMetadata = read_metadata(input_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if out_dir.resolve() == meta.path.resolve().parent and meta.path.suffix == ".mp4":
        log.warning("Output directory equals input directory; using suffixed names to avoid overwrite.")

    video_id = safe_stem(meta.path.name)
    final_video = out_dir / f"{video_id}_annotated.mp4"
    final_csv = out_dir / f"{video_id}_tracking.csv"
    part_video = out_dir / f"{video_id}_annotated.part.mp4"
    for target in (final_video, final_csv):
        if target.resolve() == meta.path.resolve():
            raise OutputWriteError("Refusing to overwrite the input video.")

    scale = 1.0
    if cfg.max_width and meta.width > cfg.max_width:
        scale = cfg.max_width / meta.width
    out_size = (int(meta.width * scale) // 2 * 2, int(meta.height * scale) // 2 * 2)

    total = min(meta.frame_count, cfg.max_frames or meta.frame_count)
    cap = cv2.VideoCapture(str(meta.path))
    writer = None
    rows: list[dict] = []
    frames_read = frames_inferred = dropped = 0
    pitch: PitchFilter | AutoPitch | None = None
    if cfg.pitch_polygon:
        pitch = PitchFilter(cfg.pitch_polygon)
    elif cfg.auto_pitch:
        pitch = AutoPitch()
    ident = IdentityManager(meta.fps, cfg.reid_max_gap_seconds) if cfg.reid else None
    teams = TeamClassifier() if cfg.detect_teams else None
    t0 = time.perf_counter()
    completed = False
    try:
        if not cap.isOpened():
            raise VideoError(f"Could not open {meta.path}")
        writer = _open_writer(part_video, meta.fps, out_size)
        tracker.reset()
        last_boxes: list[TrackedBox] = []
        while frames_read < total:
            if should_cancel and should_cancel():
                raise InterruptedError("Processing cancelled by user.")
            ok, frame = cap.read()
            if not ok:
                log.warning("Stream ended early at frame %d of %d.", frames_read, total)
                break
            if scale != 1.0:
                frame = cv2.resize(frame, out_size, interpolation=cv2.INTER_AREA)
            if frames_read % cfg.frame_stride == 0:
                boxes = tracker.update(frame)
                if isinstance(pitch, AutoPitch):
                    pitch.update(frame)
                if pitch is not None:  # judged in the (possibly downscaled) frame the boxes came from
                    kept = pitch.filter(boxes, out_size)
                    dropped += len(boxes) - len(kept)
                    boxes = kept
                frames_inferred += 1
                if teams is not None or ident is not None:
                    feats = [torso_feature(frame, b) for b in boxes]  # computed once, shared
                    if ident is not None:
                        boxes, feats = ident.assign(frames_read, boxes, feats)  # stable IDs from here on
                    if teams is not None:
                        for b, f in zip(boxes, feats, strict=True):
                            teams.observe(b.track_id, f)
                last_boxes = boxes
                for b in boxes:  # record in ORIGINAL-resolution pixels
                    ob = TrackedBox(
                        b.track_id, b.confidence, b.x1 / scale, b.y1 / scale, b.x2 / scale, b.y2 / scale
                    )
                    rows.append(box_to_row(video_id, frames_read, meta.fps, ob))
            if cfg.style == "tactical":  # held boxes on skipped frames (display only)
                live = {b.track_id: teams.label(b.track_id) for b in last_boxes} if teams else None
                annotate_tactical(frame, last_boxes, live, teams.colors if teams else None)
            else:
                annotate(frame, last_boxes)
            if pitch is not None:
                pitch.draw(frame, out_size)
            writer.write(frame)
            frames_read += 1
            if progress and (frames_read % 10 == 0 or frames_read == total):
                progress(frames_read, total, time.perf_counter() - t0)
        completed = True
    finally:
        cap.release()
        if writer is not None:
            writer.release()
        if not completed and part_video.exists():
            log.error("Run incomplete; leaving %s marked as .part (not a finished output).", part_video)

    if frames_read == 0:
        part_video.unlink(missing_ok=True)
        raise VideoError("No frames were processed.")

    df = build_dataframe(rows)
    if teams is not None and len(df):
        df["team_label"] = df["track_id"].map(teams.final_labels()).astype("object")
    validate(df, meta.width, meta.height)
    export_csv(df, final_csv)
    part_video.replace(final_video)
    elapsed = time.perf_counter() - t0
    result = ProcessingResult(
        video_id,
        final_video,
        final_csv,
        frames_read,
        frames_inferred,
        len(df),
        int(df["track_id"].nunique()) if len(df) else 0,
        elapsed,
        dropped,
        float(df.groupby("frame_number").size().median()) if len(df) else 0.0,
        ident.merges if ident is not None else 0,
    )
    log.info(
        "Done: %d frames, %d detections, %d tracks, %.2fs (%.1f fps end-to-end)",
        frames_read,
        len(df),
        result.unique_tracks,
        elapsed,
        result.throughput_fps,
    )
    return result
