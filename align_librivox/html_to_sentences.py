"""
html_to_sentences.py

Clean per-chapter HTML into paragraph-level sentence lists.

Usage:
    python html_to_sentences.py --config book_config.yaml
    python html_to_sentences.py --html html/001.html --out sentences/001.json
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import yaml
from bs4 import BeautifulSoup, NavigableString, Tag

try:
    import spacy
    _SPACY_AVAILABLE = True
except ImportError:
    _SPACY_AVAILABLE = False


BLOCK_TAGS = "p, blockquote, li"
REMOVE_SELECTOR = "script, style, nav, header, footer, .noprint"
DEFAULT_PRESERVE_TAGS = {"i", "em", "b", "strong"}
TAG_MARKDOWN = {
    "i": ("*", "*"),
    "em": ("*", "*"),
    "b": ("**", "**"),
    "strong": ("**", "**"),
    "s": ("~~", "~~"),
    "del": ("~~", "~~"),
    "code": ("`", "`"),
    "sup": ("^", "^"),
    "sub": ("~", "~"),
}
ABBREVIATIONS = {
    “adm”, “capt”, “col”, “dr”, “gen”, “hon”, “jr”, “m”, “mme”, “mlle”, “mr”,
    “mrs”, “ms”, “prof”, “rev”, “sr”, “st”,
}

_SENTENCISER_NORM_TABLE = str.maketrans({
    ““”: ‘”’, “””: ‘”’,
    “‘”: “’”, “’”: “’”,
    “*”: “ “,
    “[“: “(“,
    “]”: “)”,
})


def normalise_space(value: str) -> str:
    value = value.replace(“\xa0”, “ “)
    value = re.sub(r”\s+”, “ “, value)
    return value.strip()


def _normalise_for_sentenciser(text: str) -> str:
    “””Return a same-length normalised copy suitable for sentence boundary detection.

    Inline markup markers (*) and editorial brackets ([]) are replaced with
    neutral punctuation so they don’t mislead the splitter.  Curly quotes become
    their ASCII equivalents.  All replacements are one-for-one so spaCy’s
    start_char/end_char offsets remain valid against the original text.”””
    return text.translate(_SENTENCISER_NORM_TABLE)


def get_sentence_splitter(model: str | None = None):
    if _SPACY_AVAILABLE:
        models = [model] if model else [“en_core_web_sm”, “xx_sent_ud_sm”, “xx_ent_wiki_sm”]
        for name in models:
            if not name:
                continue
            try:
                nlp = spacy.load(name, disable=[“ner”, “tagger”, “lemmatizer”])
                if “parser” in nlp.pipe_names:
                    nlp.disable_pipe(“parser”)
                if “sentencizer” not in nlp.pipe_names:
                    nlp.add_pipe(“sentencizer”)
                print(f”Using spaCy model: {name}”, file=sys.stderr)
                def _spacy_split(text, _nlp=nlp):
                    doc = _nlp(_normalise_for_sentenciser(text))
                    return [text[s.start_char:s.end_char].strip() for s in doc.sents
                            if text[s.start_char:s.end_char].strip()]
                return _spacy_split
            except OSError:
                continue

    return split_sentences_regex


def is_abbreviation(text: str, period_index: int) -> bool:
    prefix = text[:period_index].rstrip()
    match = re.search(r”([A-Za-zÀ-ÖØ-öø-ÿ]+)$”, prefix)
    return bool(match and match.group(1).lower() in ABBREVIATIONS)


def starts_sentence(text: str, index: int) -> bool:
    while index < len(text) and text[index].isspace():
        index += 1
    while index < len(text) and text[index] in “\”“”’”’«”:
        index += 1
    if index >= len(text):
        return True
    return text[index].isupper() or text[index].isdigit()


def split_sentences_regex(text: str) -> list[str]:
    sentences = []
    start = 0
    index = 0
    end_punctuation = “.!?。！？”
    closers = “\”’”’”’»)”

    while index < len(text):
        char = text[index]
        if char not in end_punctuation:
            index += 1
            continue
        if char == "." and is_abbreviation(text, index):
            index += 1
            continue

        end = index + 1
        while end < len(text) and text[end] in closers:
            end += 1

        if end >= len(text) or starts_sentence(text, end):
            sentence = text[start:end].strip()
            if sentence:
                sentences.append(sentence)
            start = end
            index = end
        else:
            index = end

    tail = text[start:].strip()
    if tail:
        sentences.append(tail)
    return sentences


def text_with_format(node, inline_format: str, preserve_tags: set[str]) -> str:
    if isinstance(node, NavigableString):
        return str(node)
    if not isinstance(node, Tag):
        return ""
    if node.name == "br":
        return " "

    inner = "".join(text_with_format(child, inline_format, preserve_tags) for child in node.children)
    if node.name not in preserve_tags:
        return inner

    inner = inner.strip()
    if not inner or inline_format == "plain":
        return inner
    if inline_format == "html":
        return f"<{node.name}>{inner}</{node.name}>"
    open_m, close_m = TAG_MARKDOWN.get(node.name, (f"<{node.name}>", f"</{node.name}>"))
    return f"{open_m}{inner}{close_m}"


def clean_element_text(element: Tag, inline_format: str, preserve_tags: set[str]) -> str:
    return normalise_space(text_with_format(element, inline_format, preserve_tags))


def html_to_paragraphs(
    html_path: Path,
    *,
    paragraph_selector: str = BLOCK_TAGS,
    remove_selector: str = REMOVE_SELECTOR,
    sentence_model: str | None = None,
    inline_format: str = "markdown",
    preserve_tags: set[str] | None = None,
) -> list[dict]:
    if preserve_tags is None:
        preserve_tags = DEFAULT_PRESERVE_TAGS
    soup = BeautifulSoup(html_path.read_text(encoding="utf-8", errors="replace"), "lxml")
    for node in soup.select(remove_selector):
        node.decompose()

    split_sentences = get_sentence_splitter(sentence_model)
    paragraphs = []
    for element in soup.select(paragraph_selector):
        text = clean_element_text(element, inline_format, preserve_tags)
        if not text:
            continue
        sentences = split_sentences(text)
        if not sentences:
            continue
        paragraphs.append({
            "paragraph": len(paragraphs) + 1,
            "text": text,
            "sentences": sentences,
        })
    return paragraphs


def convert_file(
    html_path: Path,
    out_path: Path,
    *,
    paragraph_selector: str = BLOCK_TAGS,
    remove_selector: str = REMOVE_SELECTOR,
    sentence_model: str | None = None,
    inline_format: str = "markdown",
    preserve_tags: set[str] | None = None,
) -> dict:
    paragraphs = html_to_paragraphs(
        html_path,
        paragraph_selector=paragraph_selector,
        remove_selector=remove_selector,
        sentence_model=sentence_model,
        inline_format=inline_format,
        preserve_tags=preserve_tags,
    )
    output = {
        "source_html": str(html_path),
        "paragraphs": paragraphs,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    return output


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Clean chapter HTML into paragraph-level sentence JSON.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--config", help="YAML config containing chapters with html_file paths")
    group.add_argument("--html", help="Single HTML file")
    parser.add_argument("--out", help="Output JSON path for single-file mode")
    parser.add_argument("--outdir", default="sentences", help="Output directory for config mode")
    parser.add_argument("--paragraph-selector", default=None)
    parser.add_argument("--remove-selector", default=None)
    parser.add_argument("--sentence-model", default=None)
    parser.add_argument("--inline-format", choices=["markdown", "plain", "html"], default=None)
    parser.add_argument(
        "--preserve-tags",
        nargs="*",
        metavar="TAG",
        default=None,
        help="Inline HTML tags to keep in output (e.g. i em b). Pass nothing to strip all markup.",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser


def _resolve_preserve_tags(cli_value, config_value=None) -> set[str] | None:
    if cli_value is not None:
        return set(cli_value)
    if config_value is not None:
        return set(config_value)
    return None


def main() -> int:
    args = build_parser().parse_args()

    if args.html:
        html_path = Path(args.html)
        out_path = Path(args.out) if args.out else html_path.with_suffix(".sentences.json")
        convert_file(
            html_path,
            out_path,
            paragraph_selector=args.paragraph_selector or BLOCK_TAGS,
            remove_selector=args.remove_selector or REMOVE_SELECTOR,
            sentence_model=args.sentence_model,
            inline_format=args.inline_format or "markdown",
            preserve_tags=_resolve_preserve_tags(args.preserve_tags),
        )
        print(f"Written: {out_path}")
        return 0

    config_path = Path(args.config)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    outdir = Path(args.outdir)
    paragraph_selector = args.paragraph_selector or config.get("paragraph_selector", BLOCK_TAGS)
    remove_selector = args.remove_selector or config.get("remove_selector", REMOVE_SELECTOR)
    sentence_model = args.sentence_model or config.get("sentence_model")
    inline_format = args.inline_format or config.get("inline_format", "markdown")
    preserve_tags = _resolve_preserve_tags(args.preserve_tags, config.get("preserve_tags"))

    chapters = config.get("chapters", [])
    if not chapters:
        print("ERROR: config has no chapters", file=sys.stderr)
        return 1

    failures = 0
    for index, chapter in enumerate(chapters, start=1):
        html_file = chapter.get("html_file")
        if not html_file:
            print(f"[{index}/{len(chapters)}] Skipping: no html_file", file=sys.stderr)
            failures += 1
            continue

        html_path = Path(html_file)
        out_path = outdir / f"{index:03d}.json"
        if out_path.exists() and not args.overwrite:
            print(f"[{index}/{len(chapters)}] Skipping existing {out_path}")
        else:
            chapter_preserve = _resolve_preserve_tags(None, chapter.get("preserve_tags")) or preserve_tags
            output = convert_file(
                html_path,
                out_path,
                paragraph_selector=paragraph_selector,
                remove_selector=remove_selector,
                sentence_model=sentence_model,
                inline_format=chapter.get("inline_format", inline_format),
                preserve_tags=chapter_preserve,
            )
            print(f"[{index}/{len(chapters)}] Wrote {out_path} ({len(output['paragraphs'])} paragraphs)")

        chapter["sentence_json"] = str(out_path)

    config_path.write_text(yaml.dump(config, allow_unicode=True, sort_keys=False), encoding="utf-8")
    print(f"\nConfig updated: {config_path}")
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
