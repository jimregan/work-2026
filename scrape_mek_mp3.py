#!/usr/bin/env python3
"""
Scrape MP3 files from a MEK audio collection page (live or Wayback Machine snapshot).
Downloads all MP3s and writes a JSON manifest with anchor text and original URLs.

Usage:
  python scrape_mek_mp3.py <url>             # fetch from URL (live or Wayback)
  python scrape_mek_mp3.py <url> <file.html> # parse locally saved HTML, use url for context
"""

import json
import re
import sys
import time
from pathlib import Path
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup

WAYBACK_RE = re.compile(r"https?://web\.archive\.org/web/(\d{14})/(https?://.+)")


def parse_url(url: str) -> tuple[str, str | None]:
    """Return (original_url, wayback_timestamp_or_None)."""
    m = WAYBACK_RE.match(url)
    if m:
        return m.group(2), m.group(1)
    return url, None


def collection_id(original_url: str) -> str:
    """Extract a short identifier from the URL path, e.g. '03309'."""
    parts = [p for p in urlparse(original_url).path.strip("/").split("/") if p]
    return parts[-2] if len(parts) >= 2 else parts[-1] if parts else "mek"


def make_download_url(filename: str, base_url: str, timestamp: str | None) -> str:
    original = urljoin(base_url if base_url.endswith("/") else base_url + "/", filename)
    if timestamp:
        return f"https://web.archive.org/web/{timestamp}/{original}"
    return original


def find_text_version(soup: BeautifulSoup) -> str | None:
    for tag in soup.find_all(string=re.compile(r"[Ss]z.veges?\s+v.ltozat|[Tt]ext\s+version")):
        parent = tag.parent
        a = parent.find_next("a") if parent else None
        if a and a.get("href"):
            href = a["href"]
            m = WAYBACK_RE.match(href)
            return m.group(2) if m else href
    return None


def parse_entries(html: str, base_url: str, timestamp: str | None) -> tuple[list[dict], str | None]:
    soup = BeautifulSoup(html, "html.parser")
    entries = []
    for a in soup.find_all("a", href=True):
        href = a["href"]
        m = WAYBACK_RE.match(href)
        if m:
            href = Path(urlparse(m.group(2)).path).name
        if not href.lower().endswith(".mp3"):
            continue
        filename = Path(urlparse(href).path).name
        anchor_text = a.get_text(" ", strip=True)
        original_url = urljoin(base_url if base_url.endswith("/") else base_url + "/", filename)
        entries.append({
            "anchor_text": anchor_text,
            "original_url": original_url,
            "download_url": make_download_url(filename, base_url, timestamp),
            "filename": filename,
        })
    text_version = find_text_version(soup)
    return entries, text_version


def fetch_html(url: str) -> str:
    session = requests.Session()
    session.headers["User-Agent"] = "mek-archiver/1.0 (research; contact: public)"
    resp = session.get(url, timeout=30)
    resp.raise_for_status()
    return resp.text


def download_all(entries: list[dict], output_dir: Path) -> list[dict]:
    output_dir.mkdir(parents=True, exist_ok=True)
    session = requests.Session()
    session.headers["User-Agent"] = "mek-archiver/1.0 (research; contact: public)"

    manifest = []
    for entry in entries:
        dest = output_dir / entry["filename"]
        if dest.exists():
            print(f"  skip (exists): {entry['filename']}")
        else:
            print(f"  downloading:   {entry['filename']}")
            resp = session.get(entry["download_url"], timeout=120, stream=True)
            resp.raise_for_status()
            with dest.open("wb") as f:
                for chunk in resp.iter_content(chunk_size=65536):
                    f.write(chunk)
            time.sleep(0.5)

        manifest.append({
            "anchor_text": entry["anchor_text"],
            "original_url": entry["original_url"],
            "filename": entry["filename"],
        })

    return manifest


def main():
    if len(sys.argv) < 2:
        print("Usage: scrape_mek_mp3.py <url> [file.html]")
        sys.exit(1)

    url = sys.argv[1]
    original_url, timestamp = parse_url(url)
    base_url = original_url if original_url.endswith("/") else original_url + "/"

    if len(sys.argv) > 2:
        html = Path(sys.argv[2]).read_text(encoding="iso-8859-2", errors="replace")
    else:
        print(f"Fetching {url}")
        html = fetch_html(url)

    entries, text_version = parse_entries(html, base_url, timestamp)
    if not entries:
        print("No MP3 links found.")
        sys.exit(1)

    print(f"Found {len(entries)} MP3 file(s)")

    cid = collection_id(original_url)
    output_dir = Path(f"mek_{cid}")
    manifest = download_all(entries, output_dir)

    manifest_data = {
        "source_url": url,
        "original_base_url": base_url,
        **({"text_version": text_version} if text_version else {}),
        "files": manifest,
    }
    manifest_file = output_dir / "manifest.json"
    manifest_file.write_text(json.dumps(manifest_data, ensure_ascii=False, indent=2))
    print(f"Manifest written to {manifest_file}")


if __name__ == "__main__":
    main()
