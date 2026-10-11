# PitchVision AI

Football video analysis: detects and tracks every player on the pitch, colours them by team, and renders a tactical-style overlay (ellipses at the players' feet) plus a per-frame tracking CSV.

## Features

- **Player detection and tracking** with YOLOv8 and ByteTrack / BoT-SORT (Ultralytics).
- **Wyscout-style overlay**: translucent ellipse under each player's feet with their ID, instead of boxes.
- **Team grouping** from shirt colour (teams A / B, and "other" for referee and goalkeepers). No training data needed.
- **ID re-assignment**: when a player disappears and the tracker re-finds them under a new ID, the old ID is restored.
- **Automatic pitch detection** from the grass colour. It follows the pitch outline as the camera moves and drops the crowd, staff and ad boards.
- **Tracking dataset (CSV)** with positions per frame, ready for later analytics.

## Install

Requires Python 3.10 or newer.

```bash
git clone https://github.com/PanshulBhardwaj/PitchVisionAI.git
cd PitchVisionAI
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # macOS / Linux
pip install -r requirements.txt
```

The first run downloads the YOLO model weights automatically, so it needs an internet connection once.

## Usage

```bash
python run_baseline.py path/to/match.mp4 --out data/outputs/run1
```

Wrap the path in quotes if it contains spaces. Try a short clip first:

```bash
python run_baseline.py match.mp4 --max-frames 300 --out data/outputs/test
```

### Outputs

Written to the `--out` folder (default `data/outputs`):

| File | Contents |
| --- | --- |
| `<video>_annotated.mp4` | Video with the tactical overlay and the pitch outline |
| `<video>_tracking.csv` | One row per player per frame |

CSV columns: `video_id`, `frame_number`, `timestamp_seconds`, `track_id`, `detection_confidence`, `bounding_box_x1/y1/x2/y2`, `center_x/y`, `foot_point_x/y`, `team_label`, `pitch_x_meters`, `pitch_y_meters`.

Coordinates are image pixels of the original video (origin top-left). `pitch_x_meters` and `pitch_y_meters` stay empty until pitch calibration is added.

At the end of a run the CLI prints the median number of people per frame, how many detections fell outside the pitch, how many IDs were restored, and **IDs per median person**. Close to 1.0 means about one ID per player; higher means more fragmentation. Use it to compare settings.

## Options

| Flag | Default | What it does |
| --- | --- | --- |
| `--preset {fast,accurate}` | `accurate` | `accurate` = YOLOv8m, image size 960, BoT-SORT with camera-motion compensation. `fast` = YOLOv8n, 640, ByteTrack |
| `--weights`, `--imgsz`, `--tracker` | from preset | Override a single preset setting |
| `--conf` | 0.25 | Detection confidence threshold |
| `--style {tactical,boxes}` | `tactical` | Ellipses at the feet, or plain boxes for debugging |
| `--teams` / `--no-teams` | on | Team colouring from shirt colour |
| `--reid` / `--no-reid` | on | Restore a player's old ID after they return |
| `--reid-gap` | 6.0 | Max seconds a player can be gone and still be re-matched |
| `--auto-pitch` / `--no-auto-pitch` | on | Detect the pitch from grass colour |
| `--pitch-polygon` | none | Manual pitch area as normalised `"x,y;x,y;..."` (0..1). Overrides auto-pitch |
| `--stride` | 1 | Run detection on every Nth frame |
| `--max-frames`, `--max-width` | none | Limit frames, or downscale before inference |
| `--device` | auto | `auto` uses a GPU if available, otherwise CPU |

The accurate preset is much slower on a CPU. A GPU is strongly recommended for full-length videos.

## Project structure

```
run_baseline.py        CLI entry point
core/                  Config, logging, exceptions
video/                 Video reading and the processing pipeline
detection/             YOLO + tracker wrapper, pitch filters, ID re-assignment
analytics/             Tracking dataset and team classification
configs/               Tracker settings (bytetrack_pitch.yaml, botsort_pitch.yaml)
scripts/               Helper scripts (synthetic test video, sample download)
tests/                 Unit tests
backend/, dashboard/   Placeholders for later phases
```

## Tests

```bash
pytest
```

## Limitations

- Team grouping uses shirt colour only. A goalkeeper kit close to an outfield kit, or strong lighting changes, can cause wrong labels.
- ID re-assignment uses position, box size and shirt colour. Two team-mates who vanish and return together in the same spot can be swapped. Telling them apart reliably needs jersey-number reading or a learned ReID model.
- Auto pitch detection assumes green grass. Very dark or patchy pitches may need `--pitch-polygon`.
- Overlay labels show track IDs, not player names or shirt numbers.
- Positions are in image pixels. Metre coordinates need pitch calibration, which is not implemented yet.
- Results depend on the footage. Treat accuracy as unverified until checked on your own clips.

## Roadmap

- Pitch calibration (homography) for positions in metres
- Formation lines drawn between players
- Jersey-number reading for names and more reliable re-identification
- Dashboard and API (the `backend/` and `dashboard/` folders are placeholders)

## Footage and licensing

Do not commit match footage to the repository. Broadcast video is usually copyrighted. The `.gitignore` already excludes `sample_data/` videos, model weights and output files.
