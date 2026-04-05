"""
parse_index.py
Parses a downloaded LibriVox HTML index page and extracts chapter/section
text URLs, writing a YAML config stub for the book.

Usage:
    python parse_index.py --html path/to/index.html --out book_config.yaml
"""

import re
import sys
import argparse
from pathlib import Path

import yaml
from bs4 import BeautifulSoup


# ── Heuristics for finding "read the text" links ─────────────────────────────

TEXT_HOSTS = re.compile(
    r"(gutenberg\.org|wikisource\.org|archive\.org|en\.wikisource|"
    r"standardebooks\.org|fadedpage\.com)",
    re.I,
)

CHAPTER_PATTERNS = re.compile(
    r"(chapter|section|part|book|canto|stanza|scene)\s*[\divxlc]+",
    re.I,
)


def extract_text_links(soup: BeautifulSoup) -> list[dict]:
    """Return a list of {label, url} dicts for plausible text source links."""
    links = []
    seen = set()
    for a in soup.find_all("a", href=True):
        href = a["href"].strip()
        text = a.get_text(strip=True)
        if href in seen:
            continue
        if TEXT_HOSTS.search(href):
            seen.add(href)
            links.append({"label": text or href, "url": href})
    return links


def extract_chapters(soup: BeautifulSoup) -> list[dict]:
    """
    Try to find a chapter/section listing with per-chapter audio + text links.
    Returns list of {chapter, audio_file, text_url} stubs.
    """
    chapters = []
    # LibriVox index pages often have a table or list with chapter info
    for row in soup.select("table tr, ul li, ol li"):
        cells = row.find_all(["td", "th"])
        text_nodes = cells if cells else [row]

        label = ""
        audio_url = ""
        text_url = ""

        for node in text_nodes:
            node_text = node.get_text(" ", strip=True)
            if CHAPTER_PATTERNS.search(node_text) and not label:
                label = node_text[:80]
            for a in node.find_all("a", href=True):
                href = a["href"]
                if href.endswith(".mp3") or "archive.org/download" in href:
                    if not audio_url:
                        audio_url = href
                elif TEXT_HOSTS.search(href):
                    if not text_url:
                        text_url = href

        if label or audio_url or text_url:
            chapters.append(
                {
                    "chapter": label or f"chapter_{len(chapters)+1}",
                    "audio_file": audio_url or "FILL_IN.mp3",
                    "text_url": text_url or "FILL_IN",
                    "whisperx_json": "FILL_IN.json",
                }
            )

    return chapters


def parse_index(html_path: Path) -> dict:
    soup = BeautifulSoup(html_path.read_text(encoding="utf-8", errors="replace"), "lxml")

    title_tag = soup.find("h1") or soup.find("title")
    title = title_tag.get_text(strip=True) if title_tag else html_path.stem

    top_links = extract_text_links(soup)
    chapters = extract_chapters(soup)

    return {
        "title": title,
        "source_html": str(html_path),
        # Top-level text source links found on the page
        "text_source_links": top_links,
        # Per-chapter mapping stubs (edit as needed)
        "chapters": chapters if chapters else [
            {
                "chapter": "chapter_1",
                "audio_file": "FILL_IN.mp3",
                "whisperx_json": "FILL_IN.json",
                "text_url": top_links[0]["url"] if top_links else "FILL_IN",
            }
        ],
    }


def main():
    parser = argparse.ArgumentParser(description="Parse LibriVox HTML index page.")
    parser.add_argument("--html", required=True, help="Path to downloaded index HTML")
    parser.add_argument("--out", default="book_config.yaml", help="Output YAML path")
    args = parser.parse_args()

    html_path = Path(args.html)
    if not html_path.exists():
        print(f"ERROR: {html_path} not found", file=sys.stderr)
        sys.exit(1)

    config = parse_index(html_path)
    out_path = Path(args.out)
    out_path.write_text(yaml.dump(config, allow_unicode=True, sort_keys=False), encoding="utf-8")
    print(f"Written: {out_path}")
    print(f"  Title   : {config['title']}")
    print(f"  Chapters: {len(config['chapters'])}")
    print(f"  Text links found: {len(config['text_source_links'])}")
    print("Edit the YAML to fill in any FILL_IN placeholders before running align.py.")


if __name__ == "__main__":
    main()
