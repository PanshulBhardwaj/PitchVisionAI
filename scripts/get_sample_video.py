"""Sample-video workflow.

No football clip URL is hard-coded: a URL cannot be trusted to exist or to be
redistributable without checking it. Supply your own, with its license:

    python scripts/get_sample_video.py --url <direct .mp4 URL> --license "CC BY 4.0" --source "<page URL>"

or copy a file you have rights to use into sample_data/ manually. Provenance is
written to sample_data/SOURCES.md.
"""

import argparse
import sys
import urllib.request
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from core.config import safe_stem
from core.exceptions import VideoError
from video.ingestion import read_metadata

DEST = Path(__file__).resolve().parent.parent / "sample_data"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", required=True)
    ap.add_argument("--license", required=True, help="license name, e.g. 'CC BY 4.0'")
    ap.add_argument("--source", required=True, help="page where the license is stated")
    a = ap.parse_args()
    name = safe_stem(a.url.rsplit("/", 1)[-1].split("?")[0]) + ".mp4"
    out = DEST / name
    tmp = out.with_suffix(".part")
    try:
        urllib.request.urlretrieve(a.url, tmp)
        tmp.replace(out)
        m = read_metadata(out)
    except (OSError, ValueError, VideoError) as exc:
        tmp.unlink(missing_ok=True)
        print(f"Download/validation failed: {exc}\nManual workflow: place a video you may use in {DEST}")
        return 1
    with open(DEST / "SOURCES.md", "a", encoding="utf-8") as f:
        f.write(
            f"- {name} | {a.url} | license: {a.license} | source page: {a.source} | fetched {date.today()}\n"
        )
    print(f"OK: {out} ({m.width}x{m.height}, {m.duration_seconds:.1f}s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
