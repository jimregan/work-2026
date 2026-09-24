import json
import logging
import os
import requests
from sqlite3 import Row

logger = logging.getLogger(__name__)


def _webhook_urls() -> list[str]:
    raw = os.getenv("WEBHOOK_URLS", "")
    return [u.strip() for u in raw.split(",") if u.strip()]


def notify_webhooks(speech_row: Row, local_video: str) -> bool:
    urls = _webhook_urls()
    if not urls:
        logger.debug("No WEBHOOK_URLS configured; skipping notification")
        return True

    payload = {
        "anforande_id": speech_row["anforande_id"],
        "dok_id": speech_row["dok_id"],
        "talare": speech_row["talare"],
        "parti": speech_row["parti"],
        "datum": speech_row["datum"],
        "debattnamn": speech_row["debattnamn"],
        "anf_nummer": speech_row["anf_nummer"],
        "video_url": speech_row["video_url"],
        "local_video": local_video,
    }

    all_ok = True
    for url in urls:
        try:
            resp = requests.post(url, json=payload, timeout=15)
            resp.raise_for_status()
            logger.info("Notified %s for %s", url, speech_row["anforande_id"])
        except Exception as e:
            logger.error("Webhook %s failed for %s: %s", url, speech_row["anforande_id"], e)
            all_ok = False

    return all_ok
