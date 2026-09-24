import logging
import requests
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)

API_BASE = "https://data.riksdagen.se"


@dataclass
class Speech:
    anforande_id: str
    dok_id: str
    talare: str
    parti: str
    datum: str
    debattnamn: str
    anf_nummer: str
    anforande_url_xml: str


def list_speeches(page_size: int = 50, from_date: Optional[str] = None) -> list[Speech]:
    params: dict = {
        "sz": page_size,
        "utformat": "json",
        "sort": "datum",
        "sortorder": "desc",
    }
    if from_date:
        params["from"] = from_date

    resp = requests.get(f"{API_BASE}/anforandelista/", params=params, timeout=30)
    resp.raise_for_status()

    items = resp.json().get("anforandelista", {}).get("anforande", [])
    if isinstance(items, dict):
        items = [items]

    return [
        Speech(
            anforande_id=item.get("anforande_id", ""),
            dok_id=item.get("dok_id", ""),
            talare=item.get("talare", ""),
            parti=item.get("parti", ""),
            datum=item.get("datum", ""),
            debattnamn=item.get("debattnamn", ""),
            anf_nummer=item.get("anf_nummer", ""),
            anforande_url_xml=item.get("anforande_url_xml", ""),
        )
        for item in items
    ]


def get_speech_detail(anforande_id: str) -> Optional[dict]:
    url = f"{API_BASE}/anforande/{anforande_id}/json"
    try:
        resp = requests.get(url, timeout=30)
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        logger.error("Failed to fetch speech %s: %s", anforande_id, e)
        return None


def extract_video_url(speech_detail: dict) -> Optional[str]:
    """Extract video URL from speech detail JSON.

    The Riksdag API nests the speech under 'anforande'. The video link may be
    in 'video_url', 'webb_url', or similar fields.  If it's a riksdagen.se/webb-tv
    page URL rather than a direct media file, yt-dlp will handle it downstream.
    """
    anf = speech_detail.get("anforande", {})
    for key in ("video_url", "webb_url", "webburl", "media_url", "videourl"):
        val = anf.get(key)
        if val:
            return val
    return None
