"""
split_html.py

Split one downloaded book HTML into per-chapter HTML files, preserving markup.

Usage:
    python split_html.py --config book_config.yaml
    python split_html.py --config book_config.yaml --html /path/to/book.html
    python split_html.py --config book_config.yaml --outdir chapters/
"""

from __future__ import annotations

import argparse
import re
import sys
from difflib import SequenceMatcher
from pathlib import Path

import yaml
from bs4 import BeautifulSoup, Comment, Tag


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
    previous = 0
    for char in reversed(value):
        current = numerals.get(char)
        if current is None:
            return None
        if current < previous:
            total -= current
        else:
            total += current
            previous = current
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

    return [variant for variant in variants if variant]


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


def document_order_map(body: Tag) -> dict[int, int]:
    return {id(tag): index for index, tag in enumerate(body.find_all(True))}


def split_element(tag: Tag) -> Tag:
    current = tag
    while isinstance(current.parent, Tag) and current.parent.name != "body":
        if current.name in HEADING_TAGS:
            return current
        current = current.parent
    return current


def toc_candidates(soup: BeautifulSoup, order_map: dict[int, int]) -> list[dict]:
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

        element = split_element(target)
        index = order_map.get(id(element), order_map.get(id(target)))
        if index is None:
            continue

        key = (index, normalise_key(text))
        if key in seen:
            continue
        seen.add(key)
        candidates.append({
            "element": element,
            "index": index,
            "text": text,
            "num_start": extract_section_numbers(text)[0],
            "source": "toc",
        })
    return candidates


def heading_candidates(body: Tag, order_map: dict[int, int]) -> list[dict]:
    candidates = []
    for tag in body.find_all(True):
        if tag.name not in HEADING_TAGS:
            continue

        text = normalise_space(tag.get_text(" ", strip=True))
        if not text or text.lower() in {"contents", "content", "index"}:
            continue

        index = order_map.get(id(tag))
        if index is None:
            continue
        candidates.append({
            "element": tag,
            "index": index,
            "text": text,
            "num_start": extract_section_numbers(text)[0],
            "source": "heading",
        })
    return candidates


def candidate_score(chapter_label: str, chapter_num: int | None, candidate: dict) -> float:
    score = max(score_text_match(variant, candidate["text"]) for variant in label_variants(chapter_label))
    candidate_num = candidate.get("num_start")
    if chapter_num is not None and candidate_num is not None:
        score += 0.35 if chapter_num == candidate_num else -0.2
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


def inject_markers(chosen_markers: list[dict | None]) -> list[str]:
    marker_names = []
    for index, marker in enumerate(chosen_markers):
        name = f"__ALIGN_HTML_SPLIT_{index}__"
        marker_names.append(name)
        if marker is None:
            continue
        marker["element"].insert_before(Comment(name))
    return marker_names


def split_body_html(body: Tag, marker_names: list[str]) -> list[str]:
    html = body.decode_contents(formatter="html")
    marker_re = re.compile(r"<!--(__ALIGN_HTML_SPLIT_\d+__)-->")
    parts = marker_re.split(html)
    segments = {}
    current_marker = None

    for part in parts:
        if marker_re.fullmatch(f"<!--{part}-->"):
            current_marker = part
            segments.setdefault(current_marker, [])
        elif current_marker is not None:
            segments.setdefault(current_marker, []).append(part)

    return ["".join(segments.get(name, [])).strip() for name in marker_names]


def derive_text_html(config_path: Path, config: dict) -> Path:
    configured = config.get("text_html")
    if configured:
        path = Path(configured)
        if path.exists():
            return path

    source_html = config.get("source_html")
    if source_html:
        source_path = Path(source_html)
        candidate_dir = config_path.parent
        if "index" in source_path.parts:
            parts = list(source_path.parts)
            parts[parts.index("index")] = "text"
            candidate_dir = Path(*parts).parent

        html_files = sorted(candidate_dir.glob("*.html"))
        if len(html_files) == 1:
            return html_files[0]

        online = config.get("text_source_links", [])[:1]
        if not online and config.get("online_text_link"):
            online = [{"url": config["online_text_link"]}]
        if online:
            url = online[0].get("url", "")
            book_id_match = re.search(r"/(?:etext/)?(\d+)(?:[/?#.]|$)", url)
            if book_id_match:
                book_id = book_id_match.group(1)
                for file_path in html_files:
                    if book_id in file_path.name:
                        return file_path

        if html_files:
            return html_files[0]

    raise FileNotFoundError("Could not infer downloaded text HTML; pass --html or add text_html to config.")


def wrap_html_document(source: BeautifulSoup, body_html: str) -> str:
    head = source.find("head")
    head_html = head.decode_contents(formatter="html") if head else ""
    return (
        "<!doctype html>\n"
        "<html>\n"
        "<head>\n"
        f"{head_html}\n"
        "</head>\n"
        "<body>\n"
        f"{body_html}\n"
        "</body>\n"
        "</html>\n"
    )


def chapter_title(element: Tag, fallback: str) -> str:
    heading = element.find(HEADING_TAGS)
    if heading:
        text = normalise_space(heading.get_text(" ", strip=True))
        if text:
            return text
    text = normalise_space(element.get_text(" ", strip=True))
    return text[:80] if text else fallback


def chapter_heading_text(element: Tag) -> str:
    heading = element.find(HEADING_TAGS)
    if heading:
        return normalise_space(heading.get_text(" ", strip=True))
    return normalise_space(element.get_text(" ", strip=True))


def expand_chapter_group(item) -> list[int]:
    if isinstance(item, int):
        return [item]
    if isinstance(item, str):
        match = re.fullmatch(r"\s*(\d+)\s*-\s*(\d+)\s*", item)
        if match:
            start, end = int(match.group(1)), int(match.group(2))
            return list(range(start, end + 1))
        if item.strip().isdigit():
            return [int(item)]
    if isinstance(item, dict):
        if "chapters" in item:
            return expand_chapter_group(item["chapters"])
        if "from" in item and "to" in item:
            return list(range(int(item["from"]), int(item["to"]) + 1))
    if isinstance(item, list):
        if len(item) == 2 and all(isinstance(value, int) for value in item):
            start, end = item
            if start <= end:
                return list(range(start, end + 1))
        return [int(value) for value in item]
    raise ValueError(f"Unsupported combine entry: {item!r}")


def build_groups(total: int, combine_entries: list) -> list[list[int]]:
    combined_by_start = {group[0]: group for group in (expand_chapter_group(item) for item in combine_entries)}
    groups = []
    index = 1
    while index <= total:
        group = combined_by_start.get(index, [index])
        groups.append(group)
        index = max(group) + 1
    return groups


def split_by_chapter_divs(
    html_path: Path,
    selector: str,
    heading_pattern: str,
    combine_entries: list,
) -> tuple[list[str], list[dict]]:
    soup = BeautifulSoup(html_path.read_text(encoding="utf-8", errors="replace"), "lxml")
    elements = soup.select(selector)
    if not elements:
        raise ValueError(f"No chapter elements matched selector: {selector}")
    heading_re = re.compile(heading_pattern, re.I)
    elements = [element for element in elements if heading_re.search(chapter_heading_text(element))]
    if not elements:
        raise ValueError(f"No chapter elements matched heading pattern: {heading_pattern}")
    groups = build_groups(len(elements), combine_entries)
    documents = []
    chapters = []

    for output_index, group in enumerate(groups, start=1):
        selected = [elements[source_index - 1] for source_index in group]
        body_html = "\n".join(element.decode(formatter="html") for element in selected)
        documents.append(wrap_html_document(soup, body_html))
        label = chapter_title(selected[0], f"Chapter {group[0]}")
        chapters.append({
            "chapter": label,
            "source_chapters": group,
            "split_match": {
                "source": selector,
                "matched_text": label,
            },
        })

    return documents, chapters


def split_book(html_path: Path, chapters: list[dict]) -> tuple[list[str], list[dict | None]]:
    soup = BeautifulSoup(html_path.read_text(encoding="utf-8", errors="replace"), "lxml")
    body = soup.find("body") or soup
    order_map = document_order_map(body)
    candidates = toc_candidates(soup, order_map) + heading_candidates(body, order_map)
    chosen = choose_markers(chapters, candidates)
    marker_names = inject_markers(chosen)
    body_segments = split_body_html(body, marker_names)
    documents = []
    for index, segment in enumerate(body_segments):
        documents.append(wrap_html_document(soup, segment))
    return documents, chosen


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Split downloaded book HTML into per-chapter HTML files.")
    parser.add_argument("--config", required=True, help="book_config.yaml to update")
    parser.add_argument("--html", help="Downloaded book HTML. If omitted, infer from the config/source_html path.")
    parser.add_argument("--outdir", help="Output directory for chapter HTML files. Defaults to config dir/html.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing chapter HTML files.")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    config_path = Path(args.config)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    html_path = Path(args.html) if args.html else derive_text_html(config_path, config)

    chapters = config.get("chapters", [])
    outdir = Path(args.outdir) if args.outdir else config_path.parent / "html"
    outdir.mkdir(parents=True, exist_ok=True)

    if chapters:
        segments, chosen = split_book(html_path, chapters)
    else:
        selector = config.get("chapter_selector", "div.chapter")
        heading_pattern = config.get("match_regex", config.get("chapter_heading", r"\bchapter\s+\d+\b"))
        segments, chapters = split_by_chapter_divs(
            html_path,
            selector,
            heading_pattern,
            config.get("combine", []),
        )
        chosen = [chapter.get("split_match") for chapter in chapters]
        config["chapters"] = chapters

    config["text_html"] = str(html_path)

    failures = 0
    for index, chapter in enumerate(chapters):
        label = chapter.get("chapter", f"chapter_{index + 1}")
        segment = segments[index] if index < len(segments) else ""
        if not segment:
            print(f"[{index + 1}/{len(chapters)}] No split found for '{label}'", file=sys.stderr)
            failures += 1
            continue

        filename = f"{index + 1:03d}.html"
        out_path = outdir / filename
        if out_path.exists() and not args.overwrite:
            print(f"[{index + 1}/{len(chapters)}] Skipping existing {out_path}")
        else:
            out_path.write_text(segment, encoding="utf-8")
            print(f"[{index + 1}/{len(chapters)}] Wrote {out_path}")

        chapter["html_file"] = str(out_path)
        if chosen[index] is not None and "split_match" not in chapter:
            chapter["split_match"] = {
                "source": chosen[index]["source"],
                "matched_text": chosen[index]["text"],
            }

    config_path.write_text(yaml.dump(config, allow_unicode=True, sort_keys=False), encoding="utf-8")
    print(f"\nConfig updated: {config_path}")
    if failures:
        print(f"Unmatched chapters: {failures}", file=sys.stderr)
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
