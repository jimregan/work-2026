#!/usr/bin/env python3
"""
Find Project Gutenberg book IDs referenced by local index.html files, resolve the
best HTML reading URL for each book, and download the HTML file.

Examples:
    python3 gutenberg-scraper.py /path/to/book-dir
    python3 gutenberg-scraper.py /path/to/index.html --outdir downloaded-html
"""

from __future__ import annotations

import argparse
import io
import re
import sys
import time
import zipfile
from pathlib import Path
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup

BASE_GUTENBERG_URL = "https://www.gutenberg.org"
HEADERS = {
    "User-Agent": "Mozilla/5.0 (compatible; gutenberg-scraper/2.0; research script)"
}

BOOK_ID_PATTERNS = (
    re.compile(r"gutenberg\.org/ebooks/(\d+)(?:[/?#.]|$)", re.I),
    re.compile(r"gutenberg\.org/files/(\d+)(?:[/?#.]|$)", re.I),
    re.compile(r"gutenberg\.org/cache/epub/(\d+)(?:[/?#.]|$)", re.I),
)


def iter_index_files(path: Path) -> list[Path]:
    if path.is_file():
        if path.name != "index.html":
            raise ValueError(f"Expected an index.html file, got: {path}")
        return [path]

    if path.is_dir():
        return sorted(path.rglob("index.html"))

    raise ValueError(f"Path does not exist: {path}")


def extract_book_id_from_url(url: str) -> str | None:
    for pattern in BOOK_ID_PATTERNS:
        match = pattern.search(url)
        if match:
            return match.group(1)

    parsed = urlparse(url)
    path = parsed.path.rstrip("/")
    filename = path.split("/")[-1]

    if path.startswith("/ebooks/"):
        candidate = filename.split(".")[0]
        if candidate.isdigit():
            return candidate

    file_match = re.search(r"/(\d+)(?:-h|-images)?\.(?:html?|txt)", path, re.I)
    if file_match:
        return file_match.group(1)

    return None


def find_book_id_in_index(index_path: Path) -> tuple[str | None, str | None]:
    soup = BeautifulSoup(index_path.read_text(encoding="utf-8", errors="replace"), "html.parser")

    for anchor in soup.find_all("a", href=True):
        href = anchor["href"].strip()
        book_id = extract_book_id_from_url(href)
        if book_id:
            return book_id, href

    text = soup.get_text(" ", strip=True)
    for pattern in (
        re.compile(r"https?://www\.gutenberg\.org/ebooks/\d+", re.I),
        re.compile(r"https?://www\.gutenberg\.org/files/\d+/\S+", re.I),
    ):
        match = pattern.search(text)
        if match:
            href = match.group(0)
            return extract_book_id_from_url(href), href

    return None, None


def fetch_with_retries(session: requests.Session, url: str, timeout: int, retries: int = 3) -> requests.Response:
    last_error: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            response = session.get(url, timeout=timeout)
            response.raise_for_status()
            return response
        except requests.RequestException as exc:
            last_error = exc
            if attempt == retries:
                break
            time.sleep(1.5 * attempt)
    assert last_error is not None
    raise last_error


def choose_html_link(book_id: str, soup: BeautifulSoup, page_url: str) -> str | None:
    direct_candidates: list[str] = []
    zip_candidates: list[str] = []

    for anchor in soup.find_all("a", href=True):
        href = urljoin(page_url, anchor["href"].strip())
        lower_href = href.lower()
        label = anchor.get_text(" ", strip=True).lower()

        if f"/ebooks/{book_id}" not in lower_href and f"/cache/epub/{book_id}/" not in lower_href:
            continue

        if any(token in lower_href for token in (".html.images", ".htm.images", "-images.html", "-h.html")):
            direct_candidates.append(href)
            continue

        if lower_href.endswith((".html", ".htm")):
            direct_candidates.append(href)
            continue

        if "read now" in label and ".images" in lower_href:
            direct_candidates.append(href)
            continue

        if ".zip" in lower_href and "html" in label:
            zip_candidates.append(href)

    if direct_candidates:
        # Prefer the canonical /ebooks/<id>.html.images reader if present.
        direct_candidates.sort(key=lambda href: (f"/ebooks/{book_id}.html.images" not in href.lower(), len(href)))
        return direct_candidates[0]

    if zip_candidates:
        return zip_candidates[0]

    return None


def target_path_for(index_path: Path, outdir: Path | None, book_id: str) -> Path:
    filename = f"gutenberg-{book_id}.html"
    if outdir is not None:
        outdir.mkdir(parents=True, exist_ok=True)
        return outdir / filename
    return index_path.parent / filename


def download_book_html(
    session: requests.Session,
    book_id: str,
    destination: Path,
    timeout: int,
) -> str:
    ebook_page_url = f"{BASE_GUTENBERG_URL}/ebooks/{book_id}"
    landing_response = fetch_with_retries(session, ebook_page_url, timeout=timeout)
    soup = BeautifulSoup(landing_response.text, "html.parser")

    html_url = choose_html_link(book_id, soup, landing_response.url)
    if not html_url:
        raise RuntimeError(f"Could not find an HTML download/read link on {ebook_page_url}")

    html_response = fetch_with_retries(session, html_url, timeout=timeout)
    if html_response.url.lower().endswith(".zip"):
        with zipfile.ZipFile(io.BytesIO(html_response.content)) as archive:
            members = [name for name in archive.namelist() if name.lower().endswith((".html", ".htm"))]
            if not members:
                raise RuntimeError(f"No HTML file found in archive: {html_response.url}")
            members.sort(key=lambda name: ("-h." not in name.lower() and "-images." not in name.lower(), len(name)))
            destination.write_bytes(archive.read(members[0]))
    else:
        destination.write_bytes(html_response.content)
    return html_response.url


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Resolve Gutenberg IDs from local index.html files and download the HTML texts."
    )
    parser.add_argument("path", help="A directory to scan recursively or a single index.html file")
    parser.add_argument(
        "--outdir",
        type=Path,
        help="Optional directory for downloaded HTML files. Defaults to each index.html parent directory.",
    )
    parser.add_argument("--timeout", type=int, default=30, help="HTTP timeout in seconds")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing downloaded HTML files",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    scan_path = Path(args.path).expanduser()

    try:
        index_files = iter_index_files(scan_path)
    except ValueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    if not index_files:
        print("No index.html files found.", file=sys.stderr)
        return 1

    session = requests.Session()
    session.headers.update(HEADERS)

    failures = 0
    for index_path in index_files:
        print(f"\n==> {index_path}")
        book_id, source_url = find_book_id_in_index(index_path)
        if not book_id:
            print("  ERROR: No Project Gutenberg book ID found in this file.")
            failures += 1
            continue

        destination = target_path_for(index_path, args.outdir, book_id)
        print(f"  Book ID: {book_id}")
        if source_url:
            print(f"  Source link: {source_url}")
        print(f"  Output: {destination}")

        if destination.exists() and not args.overwrite:
            print("  Skipping existing file. Use --overwrite to replace it.")
            continue

        try:
            resolved_url = download_book_html(session, book_id, destination, timeout=args.timeout)
            print(f"  Downloaded from: {resolved_url}")
        except Exception as exc:
            print(f"  ERROR: {exc}")
            failures += 1

    print(f"\nProcessed {len(index_files)} index.html file(s); failures: {failures}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
