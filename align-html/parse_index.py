"""
parse_index.py
Parses a downloaded LibriVox HTML index page and extracts chapter metadata plus
the most relevant online-text links, writing a YAML config stub for the book.

Usage:
    python parse_index.py --html path/to/index.html --out book_config.yaml
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from urllib.parse import urljoin

import yaml
from bs4 import BeautifulSoup


TEXT_HOSTS = re.compile(
    r"(gutenberg\.org|wikisource\.org|standardebooks\.org|fadedpage\.com)",
    re.I,
)


def normalise_space(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip()


def page_title(soup: BeautifulSoup, html_path: Path) -> str:
    for selector in (".page.book-page h1", ".book-page h1", "h1", "title"):
        node = soup.select_one(selector)
        if not node:
            continue
        text = normalise_space(node.get_text(" ", strip=True))
        if text and text.lower() != "librivox":
            return text
    return html_path.stem


def extract_text_links(soup: BeautifulSoup, html_path: Path) -> list[dict]:
    """Return top-level online text links, not generic external links."""
    links = []
    seen = set()

    sidebar = soup.select_one(".book-page-sidebar")
    candidates = sidebar.find_all("a", href=True) if sidebar else soup.find_all("a", href=True)

    for anchor in candidates:
        href = urljoin(html_path.as_uri(), anchor["href"].strip())
        label = normalise_space(anchor.get_text(" ", strip=True))
        label_lower = label.lower()

        if href in seen:
            continue
        if not TEXT_HOSTS.search(href):
            continue
        if "online text" not in label_lower and "project gutenberg" not in label_lower and "read" not in label_lower:
            continue

        seen.add(href)
        links.append({"label": label or href, "url": href})

    return links


def extract_chapters(soup: BeautifulSoup) -> list[dict]:
    """
    Extract chapters from the LibriVox chapter table.
    Returns list of {chapter, audio_file, text_url, whisperx_json}.
    """
    table = soup.select_one("table.chapter-download")
    if not table:
        return []

    chapters = []
    rows = table.select("tbody tr") or table.select("tr")
    for row in rows:
        cells = row.find_all("td", recursive=False) or row.find_all("td")
        if len(cells) < 2:
            continue

        section_text = normalise_space(cells[0].get_text(" ", strip=True))
        chapter_cell = cells[1]
        chapter_text = normalise_space(chapter_cell.get_text(" ", strip=True))
        audio_anchor = cells[0].find("a", href=True)

        if not chapter_text:
            continue

        section_match = re.search(r"\b(\d+[A-Za-z]?(?:-\d+[A-Za-z]?)?)\b\s*$", section_text)
        section = section_match.group(1) if section_match else None
        chapter_label = chapter_text
        if section and not chapter_text.lower().startswith(("chapter", "chapters", "section", "part", "book")):
            chapter_label = f"{section}: {chapter_text}"

        audio_url = audio_anchor["href"].strip() if audio_anchor else "FILL_IN.mp3"
        if audio_url and not audio_url.startswith(("http://", "https://")):
            audio_url = urljoin("https://archive.org/", audio_url)

        chapters.append(
            {
                "chapter": chapter_label,
                "audio_file": audio_url or "FILL_IN.mp3",
                "text_url": "FILL_IN",
                "whisperx_json": "FILL_IN.json",
            }
        )

    return chapters


def parse_index(html_path: Path) -> dict:
    soup = BeautifulSoup(html_path.read_text(encoding="utf-8", errors="replace"), "lxml")

    top_links = extract_text_links(soup, html_path)
    chapters = extract_chapters(soup)

    if not chapters:
        chapters = [{
            "chapter": "chapter_1",
            "audio_file": "FILL_IN.mp3",
            "whisperx_json": "FILL_IN.json",
            "text_url": top_links[0]["url"] if top_links else "FILL_IN",
        }]

    return {
        "title": page_title(soup, html_path),
        "source_html": str(html_path),
        "text_source_links": top_links,
        "chapters": chapters,
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
