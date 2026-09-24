import logging
import os
import subprocess
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


def download_video(video_url: str, output_dir: str, anforande_id: str) -> Optional[str]:
    """Download a video using yt-dlp.

    Returns the local file path on success, None on failure.
    yt-dlp handles both direct MP4 URLs and streaming pages (webb-tv).
    """
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    output_template = str(out_dir / f"{anforande_id}.%(ext)s")

    cmd = [
        "yt-dlp",
        "--no-playlist",
        "--output", output_template,
        "--quiet",
        "--no-warnings",
        video_url,
    ]

    logger.info("Downloading %s -> %s", video_url, output_template)
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        logger.error("yt-dlp failed for %s: %s", anforande_id, result.stderr.strip())
        return None

    # Find the file yt-dlp wrote (extension varies)
    for candidate in out_dir.iterdir():
        if candidate.stem == anforande_id:
            logger.info("Downloaded %s -> %s", anforande_id, candidate)
            return str(candidate)

    logger.error("yt-dlp reported success but no file found for %s", anforande_id)
    return None
