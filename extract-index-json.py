#!/usr/bin/env python3
"""
Extract structured metadata from downloaded index HTML files and write one JSON
file per input HTML file.

Expected fields per page:
  - title
  - author
  - online_text_link
  - chapters[]
  - cast_details[]

Examples:
    python3 extract-index-json.py
    python3 extract-index-json.py /data/index --outdir /data/json
    python3 extract-index-json.py /data/index/some-book/index.html
"""

from __future__ import annotations

import argparse
import html
import json
import re
import sys
from pathlib import Path
from urllib.parse import urljoin

from bs4 import BeautifulSoup, Tag


DEFAULT_INPUT_PATH = "/Users/joregan/Playing/librivox_mult/index"
CAST_LINE_RE = re.compile(
    r"(?:^|<br\s*/?>)\s*(?:<strong>\s*Cast\s*</strong>\s*<br\s*/?>\s*)?"
    r"([^:<]{1,120}?)\s*:\s*<a [^>]*href=[\"']([^\"']+)[\"'][^>]*>(.*?)</a>",
    re.I | re.S,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract structured JSON from index HTML files."
    )
    parser.add_argument(
        "path",
        nargs="?",
        default=DEFAULT_INPUT_PATH,
        help=f"Directory to scan recursively or a single HTML file. Defaults to {DEFAULT_INPUT_PATH}.",
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        help="Directory for JSON output. Defaults to writing beside each HTML file.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing JSON files.",
    )
    return parser.parse_args()


def iter_html_files(path: Path) -> list[Path]:
    if path.is_file():
        if path.suffix.lower() not in {".html", ".htm"}:
            raise ValueError(f"Expected an HTML file, got: {path}")
        return [path]

    if path.is_dir():
        return sorted(
            file_path
            for file_path in path.rglob("*")
            if file_path.is_file() and file_path.suffix.lower() in {".html", ".htm"}
        )

    raise ValueError(f"Path does not exist: {path}")


def normalise_space(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip()


def text_or_none(node: Tag | None) -> str | None:
    if not node:
        return None
    text = normalise_space(node.get_text(" ", strip=True))
    return text or None


def first_h1(soup: BeautifulSoup) -> str | None:
    selectors = (
        ".page.book-page h1",
        ".book-page h1",
        "div.page h1",
        "h1",
    )
    for selector in selectors:
        node = soup.select_one(selector)
        text = text_or_none(node)
        if text and text.lower() != "librivox":
            return text
    return None


def author_from_page(soup: BeautifulSoup) -> str | None:
    author_node = soup.select_one(".page.book-page .book-page-author, .book-page .book-page-author")
    return text_or_none(author_node)


def online_text_link(soup: BeautifulSoup, page_path: Path) -> str | None:
    for anchor in soup.find_all("a", href=True):
        label = normalise_space(anchor.get_text(" ", strip=True)).lower()
        if "online text" in label:
            return urljoin(page_path.as_uri(), anchor["href"].strip())
    return None


def row_cells(row: Tag) -> list[Tag]:
    return row.find_all(["th", "td"], recursive=False) or row.find_all(["th", "td"])


def extract_reader_id(href: str | None) -> str | None:
    if not href:
        return None
    match = re.search(r"/reader/(\d+)\b", href)
    if match:
        return match.group(1)
    match = re.search(r"peopleid=(\d+)\b", href)
    if match:
        return match.group(1)
    return None


def extract_reader_from_cell(cell: Tag | None, fallback_text: str | None = None) -> tuple[str | None, str | None]:
    if not cell:
        return fallback_text, None
    anchor = cell.find("a", href=True)
    if anchor:
        reader_name = normalise_space(anchor.get_text(" ", strip=True)) or fallback_text
        reader_id = extract_reader_id(anchor["href"].strip())
        return reader_name, reader_id
    return fallback_text, None


def chapter_headers(table: Tag) -> tuple[list[str], list[Tag]]:
    rows = table.find_all("tr")
    if not rows:
        return [], []

    first_cells = row_cells(rows[0])
    headers = [normalise_space(cell.get_text(" ", strip=True)).lower() for cell in first_cells]
    has_header = any(
        header in {"section", "chapter", "reader", "time", "duration"} for header in headers
    )
    return (headers if has_header else []), rows[1:] if has_header else rows


def chapter_entry_from_cells(cells: list[Tag], headers: list[str]) -> dict[str, str | None]:
    values = [normalise_space(cell.get_text(" ", strip=True)) or None for cell in cells]

    if headers and len(headers) == len(values):
        mapping = dict(zip(headers, values))
        reader_index = headers.index("reader") if "reader" in headers else None
        reader_name, reader_id = extract_reader_from_cell(
            cells[reader_index] if reader_index is not None else None,
            mapping.get("reader"),
        )
        return {
            "section": extract_section_value(mapping.get("section")),
            "chapter": mapping.get("chapter") or mapping.get("title"),
            "reader": reader_name,
            "reader_id": reader_id,
            "time": mapping.get("time") or mapping.get("duration"),
        }

    padded = values + [None] * (4 - len(values))
    reader_name, reader_id = extract_reader_from_cell(cells[2] if len(cells) > 2 else None, padded[2])
    return {
        "section": extract_section_value(padded[0]),
        "chapter": padded[1] if len(values) > 1 else padded[0],
        "reader": reader_name,
        "reader_id": reader_id,
        "time": padded[3] if len(values) > 3 else None,
    }


def extract_section_value(value: str | None) -> str | None:
    if not value:
        return None
    match = re.search(r"\b(\d+[A-Za-z]?)\b\s*$", value)
    if match:
        return match.group(1)
    return value


def extract_chapters(soup: BeautifulSoup) -> list[dict[str, str | None]]:
    table = soup.find("table", class_="chapter-download")
    if not table:
        return []

    headers, rows = chapter_headers(table)
    chapters: list[dict[str, str | None]] = []
    for row in rows:
        if row.find("th"):
            continue
        cells = row_cells(row)
        if not cells:
            continue
        entry = chapter_entry_from_cells(cells, headers)
        if any(value for value in entry.values()):
            chapters.append(entry)
    return chapters


def extract_cast_details(soup: BeautifulSoup, page_path: Path) -> list[dict[str, str | None]]:
    cast_details: list[dict[str, str | None]] = []
    seen: set[tuple[str, str, str]] = set()

    candidate_blocks = list(soup.select("p.description")) or list(soup.find_all(["p", "div"]))
    for node in candidate_blocks:
        if not node.find("a", href=True):
            continue

        content = node.decode_contents()
        for match in CAST_LINE_RE.finditer(content):
            character = normalise_space(html.unescape(match.group(1)))
            reader_link = urljoin(page_path.as_uri(), match.group(2).strip())
            reader_name = normalise_space(
                BeautifulSoup(match.group(3), "html.parser").get_text(" ", strip=True)
            )
            if not character or not reader_name:
                continue
            if len(character.split()) > 8:
                continue

            key = (character, reader_name, reader_link)
            if key in seen:
                continue
            seen.add(key)
            cast_details.append(
                {
                    "character": character,
                    "reader_name": reader_name,
                    "reader_id": extract_reader_id(match.group(2).strip()),
                    "reader_link": reader_link,
                }
            )

    return cast_details


def extract_page_data(html_path: Path) -> dict[str, object]:
    soup = BeautifulSoup(html_path.read_text(encoding="utf-8", errors="replace"), "html.parser")
    return {
        "source_html": str(html_path),
        "title": first_h1(soup),
        "author": author_from_page(soup),
        "online_text_link": online_text_link(soup, html_path),
        "chapters": extract_chapters(soup),
        "cast_details": extract_cast_details(soup, html_path),
    }


def output_path_for(html_path: Path, outdir: Path | None, root_path: Path) -> Path:
    if outdir is None:
        if "index" in html_path.parts:
            parts = list(html_path.parts)
            parts[parts.index("index")] = "json"
            target = Path(*parts).with_suffix(".json")
            target.parent.mkdir(parents=True, exist_ok=True)
            return target
        return html_path.with_suffix(".json")
    outdir.mkdir(parents=True, exist_ok=True)
    if root_path.is_dir():
        relative_path = html_path.relative_to(root_path).with_suffix(".json")
        target = outdir / relative_path
        target.parent.mkdir(parents=True, exist_ok=True)
        return target
    return outdir / f"{html_path.stem}.json"


def main() -> int:
    args = parse_args()
    input_path = Path(args.path).expanduser()

    try:
        html_files = iter_html_files(input_path)
    except ValueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    if not html_files:
        print("No HTML files found.", file=sys.stderr)
        return 1

    failures = 0
    for html_path in html_files:
        out_path = output_path_for(html_path, args.outdir, input_path)
        print(f"\n==> {html_path}")
        print(f"  Output: {out_path}")

        if out_path.exists() and not args.overwrite:
            print("  Skipping existing file. Use --overwrite to replace it.")
            continue

        try:
            payload = extract_page_data(html_path)
            out_path.write_text(
                json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            print(
                "  Extracted:"
                f" title={bool(payload['title'])},"
                f" author={bool(payload['author'])},"
                f" chapters={len(payload['chapters'])},"
                f" cast={len(payload['cast_details'])}"
            )
        except Exception as exc:
            print(f"  ERROR: {exc}")
            failures += 1

    print(f"\nProcessed {len(html_files)} HTML file(s); failures: {failures}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
