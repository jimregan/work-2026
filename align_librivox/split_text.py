"""
split_text.py

Split one downloaded book HTML into smaller alignable text chunks using the
chapter list from a book config. It handles several common conventions:

- explicit TOC anchors in Project Gutenberg HTML
- chapter headings with numbers and titles
- story/essay collections where section titles match TOC entries
- combined LibriVox labels like "Chapters 1-2"

Usage:
    python split_text.py --config book_config.yaml
    python split_text.py --config book_config.yaml --html /path/to/book.html
    python split_text.py --config book_config.yaml --outdir texts/
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from urllib.parse import urlparse

import yaml
from bs4 import BeautifulSoup, NavigableString, Tag
from difflib import SequenceMatcher

BLOCK_TAGS = {
    "p", "div", "section", "article", "header", "footer", "blockquote", "pre",
    "ul", "ol", "li", "table", "tr", "td", "th", "h1", "h2", "h3", "h4", "h5", "h6",
    "br", "hr",
}
HEADING_TAGS = {"h1", "h2", "h3", "h4", "h5", "h6"}
ROMAN_NUMERAL_RE = r"(?:[ivxlcdm]+|\d+)"
SECTION_PREFIX_RE = re.compile(
    rf"^(?:chapter|chapters|book|part|section|scene|letter|stave)\s+{ROMAN_NUMERAL_RE}"
    rf"(?:\s*[-–—.:]\s*|\s+from\s+|\s+)",
    re.I,
)
SECTION_RANGE_RE = re.compile(
    rf"\b(chapter|chapters|book|part|section|scene|letter|stave)\s+"
    rf"({ROMAN_NUMERAL_RE})(?:\s*[-–—]\s*({ROMAN_NUMERAL_RE}))?\b",
    re.I,
)


def normalise_space(value: str) -> str:
    return re.sub(r"\s+", " ", value.replace("\xa0", " ")).strip()


def clean_text(text: str) -> str:
    text = re.sub(r"\r\n?", "\n", text)
    text = re.sub(r"[^\S\n]+", " ", text)
    text = re.sub(r" *\n *", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def normalise_key(value: str) -> str:
    value = value.lower()
    value = value.replace("’", "'").replace("—", "-").replace("–", "-")
    value = re.sub(r"[^a-z0-9]+", " ", value)
    return normalise_space(value)


def roman_to_int(value: str) -> int | None:
    value = value.lower()
    if value.isdigit():
        return int(value)
    numerals = {"i": 1, "v": 5, "x": 10, "l": 50, "c": 100, "d": 500, "m": 1000}
    total = 0
    prev = 0
    for char in reversed(value):
        current = numerals.get(char)
        if current is None:
            return None
        if current < prev:
            total -= current
        else:
            total += current
            prev = current
    return total


def extract_section_numbers(label: str) -> tuple[int | None, int | None]:
    match = SECTION_RANGE_RE.search(label)
    if not match:
        return None, None
    start = roman_to_int(match.group(2))
    end = roman_to_int(match.group(3)) if match.group(3) else start
    return start, end


def label_variants(label: str) -> list[str]:
    label = normalise_space(label)
    variants = {label}
    variants.add(re.sub(r"^\d+\s*:\s*", "", label))

    stripped = label
    while True:
        updated = SECTION_PREFIX_RE.sub("", stripped).strip()
        if updated == stripped:
            break
        variants.add(updated)
        stripped = updated

    if " from " in label.lower():
        variants.add(re.split(r"\bfrom\b", label, flags=re.I, maxsplit=1)[1].strip())

    return [v for v in variants if v]


def score_text_match(label: str, candidate: str) -> float:
    label_key = normalise_key(label)
    candidate_key = normalise_key(candidate)
    if not label_key or not candidate_key:
        return 0.0
    if label_key == candidate_key:
        return 1.0
    if label_key in candidate_key or candidate_key in label_key:
        return 0.9
    return SequenceMatcher(None, label_key, candidate_key).ratio()


def element_index_map(body: Tag) -> dict[int, int]:
    return {id(tag): idx for idx, tag in enumerate(body.find_all(True))}


def toc_candidates(soup: BeautifulSoup, index_map: dict[int, int]) -> list[dict]:
    candidates = []
    seen = set()
    for anchor in soup.select('a[href^="#"]'):
        href = anchor.get("href", "").strip()
        text = normalise_space(anchor.get_text(" ", strip=True))
        if not href or href == "#" or not text:
            continue
        if text.isdigit() or re.fullmatch(r"\[\d+\]", text):
            continue
        target_id = href[1:]
        target = soup.find(id=target_id) or soup.find(attrs={"name": target_id})
        if not isinstance(target, Tag):
            continue
        idx = index_map.get(id(target))
        if idx is None:
            continue
        key = (idx, normalise_key(text))
        if key in seen:
            continue
        seen.add(key)
        candidates.append({
            "element": target,
            "index": idx,
            "text": text,
            "num_start": extract_section_numbers(text)[0],
            "source": "toc",
        })
    return candidates


def heading_candidates(body: Tag, index_map: dict[int, int]) -> list[dict]:
    candidates = []
    for tag in body.find_all(True):
        if tag.name not in HEADING_TAGS and not tag.get("id") and not tag.get("name"):
            continue

        text = normalise_space(tag.get_text(" ", strip=True))
        if not text:
            ident = tag.get("id") or tag.get("name") or ""
            text = normalise_space(ident.replace("_", " ").replace("-", " "))
        if not text:
            continue
        if text.lower() in {"contents", "content", "index"}:
            continue

        idx = index_map.get(id(tag))
        if idx is None:
            continue
        candidates.append({
            "element": tag,
            "index": idx,
            "text": text,
            "num_start": extract_section_numbers(text)[0],
            "source": "heading",
        })
    return candidates


def candidate_score(chapter_label: str, chapter_num: int | None, candidate: dict) -> float:
    score = max(score_text_match(variant, candidate["text"]) for variant in label_variants(chapter_label))
    candidate_num = candidate.get("num_start")
    if chapter_num is not None and candidate_num is not None:
        if chapter_num == candidate_num:
            score += 0.35
        else:
            score -= 0.2
    if candidate["source"] == "toc":
        score += 0.05
    return score


def choose_markers(chapters: list[dict], candidates: list[dict]) -> list[dict | None]:
    ordered_candidates = sorted(candidates, key=lambda item: item["index"])
    chosen: list[dict | None] = []
    previous_index = -1

    for chapter in chapters:
        chapter_label = chapter.get("chapter", "")
        chapter_num, _chapter_end = extract_section_numbers(chapter_label)
        best = None
        best_score = 0.0
        for candidate in ordered_candidates:
            if candidate["index"] <= previous_index:
                continue
            score = candidate_score(chapter_label, chapter_num, candidate)
            if score > best_score:
                best = candidate
                best_score = score

        threshold = 0.72 if chapter_num is None else 0.58
        if best is not None and best_score >= threshold:
            chosen.append(best)
            previous_index = best["index"]
        else:
            chosen.append(None)

    return chosen


def inject_markers(body: Tag, chosen_markers: list[dict | None]) -> list[str]:
    marker_names = []
    inserted = set()
    for idx, marker in enumerate(chosen_markers):
        name = f"__ALIGN_MARKER_{idx}__"
        marker_names.append(name)
        if marker is None:
            continue
        element = marker["element"]
        if id(element) in inserted:
            continue
        element.insert_before(NavigableString(f"\n{name}\n"))
        inserted.add(id(element))
    return marker_names


def split_body_text(body: Tag, marker_names: list[str]) -> list[str]:
    text = body.get_text("\n")
    text = clean_text(text)
    marker_re = re.compile(r"(__ALIGN_MARKER_\d+__)")
    parts = marker_re.split(text)

    segments = {}
    current_marker = None
    for part in parts:
        part = clean_text(part)
        if not part:
            continue
        if marker_re.fullmatch(part):
            current_marker = part
            segments.setdefault(current_marker, [])
        elif current_marker is not None:
            segments.setdefault(current_marker, []).append(part)

    return [clean_text("\n\n".join(segments.get(name, []))) for name in marker_names]


def derive_text_html(config_path: Path, config: dict) -> Path:
    configured = config.get("text_html")
    if configured:
        path = Path(configured)
        if path.exists():
            return path

    source_html = config.get("source_html")
    if source_html:
        source_path = Path(source_html)
        if "index" in source_path.parts:
            parts = list(source_path.parts)
            parts[parts.index("index")] = "text"
            candidate_dir = Path(*parts).parent
            html_files = sorted(candidate_dir.glob("*.html"))
            if len(html_files) == 1:
                return html_files[0]

            online = config.get("text_source_links", [])[:1]
            if online:
                url = online[0].get("url", "")
                book_id_match = re.search(r"/(\d+)(?:[/?#.]|$)", url)
                if book_id_match:
                    book_id = book_id_match.group(1)
                    for file_path in html_files:
                        if book_id in file_path.name:
                            return file_path
            if html_files:
                return html_files[0]

    raise FileNotFoundError("Could not infer downloaded text HTML; pass --html or add text_html to config.")


def clean_html_for_splitting(soup: BeautifulSoup) -> Tag:
    for selector in ("div#pg-machine-header", "section.pg-boilerplate", "#pg-start-separator"):
        for node in soup.select(selector):
            node.decompose()
    body = soup.find("body") or soup
    return body


def split_book(html_path: Path, chapters: list[dict]) -> tuple[list[str], list[dict | None]]:
    soup = BeautifulSoup(html_path.read_text(encoding="utf-8", errors="replace"), "lxml")
    body = clean_html_for_splitting(soup)
    index_map = element_index_map(body)
    candidates = toc_candidates(soup, index_map) + heading_candidates(body, index_map)
    chosen = choose_markers(chapters, candidates)
    marker_names = inject_markers(body, chosen)
    segments = split_body_text(body, marker_names)
    return segments, chosen


def slugify(value: str, fallback: str) -> str:
    slug = re.sub(r"[^\w]+", "_", value).strip("_")
    return slug[:80] or fallback


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Split downloaded book HTML into alignable chunks.")
    parser.add_argument("--config", required=True, help="book_config.yaml to update")
    parser.add_argument("--html", help="Downloaded book HTML. If omitted, infer from the config/source_html path.")
    parser.add_argument("--outdir", help="Output directory for chunk text files. Defaults to config dir/texts.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing chunk files.")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    config_path = Path(args.config)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    html_path = Path(args.html) if args.html else derive_text_html(config_path, config)

    chapters = config.get("chapters", [])
    if not chapters:
        print("ERROR: config has no chapters", file=sys.stderr)
        return 1

    outdir = Path(args.outdir) if args.outdir else config_path.parent / "texts"
    outdir.mkdir(parents=True, exist_ok=True)

    segments, chosen = split_book(html_path, chapters)
    config["text_html"] = str(html_path)

    failures = 0
    for i, chapter in enumerate(chapters):
        label = chapter.get("chapter", f"chapter_{i+1}")
        segment = segments[i] if i < len(segments) else ""
        if not segment:
            print(f"[{i+1}/{len(chapters)}] No split found for '{label}'", file=sys.stderr)
            failures += 1
            continue

        filename = slugify(label, f"chapter_{i+1}") + ".txt"
        out_path = outdir / filename
        if out_path.exists() and not args.overwrite:
            print(f"[{i+1}/{len(chapters)}] Skipping existing {out_path}")
        else:
            out_path.write_text(segment, encoding="utf-8")
            print(f"[{i+1}/{len(chapters)}] Wrote {out_path}")

        chapter["text_file"] = str(out_path)
        if chosen[i] is not None:
            chapter["split_match"] = {
                "source": chosen[i]["source"],
                "matched_text": chosen[i]["text"],
            }

    config_path.write_text(yaml.dump(config, allow_unicode=True, sort_keys=False), encoding="utf-8")
    print(f"\nConfig updated: {config_path}")
    if failures:
        print(f"Unmatched chapters: {failures}", file=sys.stderr)
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
