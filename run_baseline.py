"""Phase 1 CLI: process a video with YOLO + ByteTrack, write annotated MP4 + CSV."""

from __future__ import annotations

import argparse
import sys

from core.config import PROJECT_ROOT, PathsConfig, ProcessingConfig
from core.exceptions import PitchVisionError
from core.logging_config import setup_logging
from detection.detector import YoloByteTrack
from detection.pitch_filter import parse_polygon
from video.ingestion import read_metadata
from video.processing import process_video


PRESETS = {
    # fast: quick smoke tests. accurate: bigger model, higher resolution, camera-motion-aware tracker.
    "fast": {"weights": "yolov8n.pt", "imgsz": 640, "tracker": "bytetrack.yaml"},
    "accurate": {
        "weights": "yolov8m.pt",
        "imgsz": 960,
        "tracker": str(PROJECT_ROOT / "configs" / "botsort_pitch.yaml"),
    },
}


def resolve_preset(preset: str, weights: str | None, imgsz: int | None, tracker: str | None):
    """Explicit flags always win over the preset."""
    p = PRESETS[preset]
    return weights or p["weights"], imgsz or p["imgsz"], tracker or p["tracker"]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="PitchVision AI baseline pipeline")
    ap.add_argument("video")
    ap.add_argument("--preset", choices=list(PRESETS), default="accurate")
    ap.add_argument("--weights", default=None, help="overrides the preset's model")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--imgsz", type=int, default=None, help="overrides the preset's inference size")
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--max-frames", type=int, default=None)
    ap.add_argument("--max-width", type=int, default=None)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--out", default=None)
    ap.add_argument("--style", choices=["boxes", "tactical"], default="tactical")
    ap.add_argument("--teams", action=argparse.BooleanOptionalAction, default=True,
                    help="colour players by team from shirt colour (default on)")
    ap.add_argument("--reid", action=argparse.BooleanOptionalAction, default=True,
                    help="restore a player's old ID after they vanish and return (default on)")
    ap.add_argument("--reid-gap", type=float, default=6.0, help="max seconds a player may be gone and still be re-matched")
    ap.add_argument("--auto-pitch", action=argparse.BooleanOptionalAction, default=True,
                    help="detect the pitch from grass colour (default on; --pitch-polygon overrides)")
    ap.add_argument("--tracker", default=None, help="overrides the preset, e.g. configs/bytetrack_pitch.yaml")
    ap.add_argument(
        "--pitch-polygon",
        default=None,
        help="Pitch area as normalised 'x,y;x,y;...' (0..1). Boxes whose feet fall outside are dropped.",
    )
    a = ap.parse_args(argv)

    setup_logging()
    weights, imgsz, tracker_cfg = resolve_preset(a.preset, a.weights, a.imgsz, a.tracker)
    print(f"Preset: {a.preset} | model: {weights} | imgsz: {imgsz} | tracker: {tracker_cfg}")
    try:
        polygon = parse_polygon(a.pitch_polygon) if a.pitch_polygon else None
    except PitchVisionError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    cfg = ProcessingConfig(
        model_weights=weights,
        confidence=a.conf,
        image_size=imgsz,
        frame_stride=a.stride,
        max_frames=a.max_frames,
        max_width=a.max_width,
        device=a.device,
        tracker=tracker_cfg,
        style=a.style,
        detect_teams=a.teams,
        reid=a.reid,
        reid_max_gap_seconds=a.reid_gap,
        auto_pitch=a.auto_pitch,
        pitch_polygon=polygon,
    )
    try:
        meta = read_metadata(a.video)
        print(
            f"Video: {meta.width}x{meta.height} @ {meta.fps:.2f} fps, "
            f"{meta.frame_count} frames, {meta.duration_seconds:.1f}s"
        )
        tracker = YoloByteTrack(cfg)
        res = process_video(
            a.video,
            tracker,
            cfg,
            a.out or PathsConfig().outputs,
            progress=lambda d, t, e: print(f"\r  {d}/{t} frames  {e:.0f}s", end="", flush=True),
        )
    except PitchVisionError as exc:
        print(f"\nERROR: {exc}", file=sys.stderr)
        return 1
    print(
        f"\nAnnotated video: {res.annotated_video}\nTracking CSV:    {res.tracking_csv}\n"
        f"{res.detections} detections, {res.unique_tracks} track IDs, "
        f"{res.throughput_fps:.2f} fps end-to-end\n"
        f"Median people/frame: {res.median_people_per_frame:.0f}  |  "
        f"dropped outside pitch: {res.dropped_outside_pitch}  |  "
        f"IDs restored after returns: {res.identity_merges}"
    )
    if res.median_people_per_frame:
        # Ideal is ~1.0 (one ID per person). Higher = more fragmentation. Compare runs on this.
        print(f"IDs per median person: {res.unique_tracks / res.median_people_per_frame:.1f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
