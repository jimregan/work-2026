"""
fetch_text.py
Fetches a text URL (Project Gutenberg, Wikisource, Standard Ebooks, etc.)
and returns clean plain text suitable for alignment.

Usage:
    python fetch_text.py --url "https://..." --out chapter1.txt

    # Or process an entire book config:
    python fetch_text.py --config book_config.yaml --outdir texts/
"""

import argparse
import re
import sys
import time
from pathlib import Path
from urllib.parse import urlparse

import requests
import yaml
from bs4 import BeautifulSoup, NavigableString

HEADERS = {
    "User-Agent": "Mozilla/5.0 (compatible; librivox-aligner/1.0; research)"
}


# ── Per-host cleaning strategies ─────────────────────────────────────────────

def _text_from_gutenberg(soup: BeautifulSoup) -> str:
    # PG wraps the body in <div class="chapter"> or just the body
    for div in soup.select("div.chapter, div#pg-machine-header"):
        div.decompose()
    # Remove PG boilerplate at top/bottom
    body = soup.find("body") or soup
    return _clean(body.get_text("\n"))


def _text_from_wikisource(soup: BeautifulSoup) -> str:
    content = soup.select_one("div#mw-content-text, div.mw-parser-output")
    if content:
        # Remove edit links, navigation boxes
        for el in content.select(".mw-editsection, .noprint, .navigation-box"):
            el.decompose()
        return _clean(content.get_text("\n"))
    return _clean(soup.get_text("\n"))


def _text_generic(soup: BeautifulSoup) -> str:
    # Remove obvious chrome
    for tag in soup.select("nav, header, footer, script, style, .sidebar, #sidebar"):
        tag.decompose()
    main = soup.select_one("main, article, div#content, div.content, div#main")
    target = main if main else soup.find("body") or soup
    return _clean(target.get_text("\n"))


HOST_STRATEGIES = {
    "gutenberg.org": _text_from_gutenberg,
    "wikisource.org": _text_from_wikisource,
}


def _clean(text: str) -> str:
    """Normalise whitespace, collapse blank lines, strip control chars."""
    text = re.sub(r"\r\n?", "\n", text)
    text = re.sub(r"[^\S\n]+", " ", text)          # collapse horizontal whitespace
    text = re.sub(r" *\n *", "\n", text)            # trim line edges
    text = re.sub(r"\n{3,}", "\n\n", text)          # max 2 consecutive blank lines
    return text.strip()


# ── Core fetch ────────────────────────────────────────────────────────────────

def fetch_and_clean(url: str, retries: int = 3, delay: float = 2.0) -> str:
    """Fetch a URL and return clean plain text."""
    for attempt in range(retries):
        try:
            resp = requests.get(url, headers=HEADERS, timeout=30)
            resp.raise_for_status()
            break
        except requests.RequestException as e:
            if attempt == retries - 1:
                raise
            print(f"  Retry {attempt+1}/{retries} after error: {e}", file=sys.stderr)
            time.sleep(delay)

    # Detect encoding
    resp.encoding = resp.apparent_encoding or "utf-8"

    # Plain text URLs (e.g. Gutenberg .txt)
    ct = resp.headers.get("content-type", "")
    if "text/plain" in ct or url.endswith(".txt"):
        return _clean(resp.text)

    soup = BeautifulSoup(resp.text, "lxml")

    host = urlparse(url).netloc.lower()
    for key, fn in HOST_STRATEGIES.items():
        if key in host:
            return fn(soup)

    return _text_generic(soup)


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Fetch and clean text for alignment.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--url", help="Single URL to fetch")
    group.add_argument("--config", help="book_config.yaml — fetch all chapters")
    parser.add_argument("--out", help="Output file path (single URL mode)")
    parser.add_argument("--outdir", default="texts", help="Output dir (config mode)")
    parser.add_argument("--delay", type=float, default=1.5,
                        help="Seconds between requests (config mode)")
    args = parser.parse_args()

    if args.url:
        out_path = Path(args.out) if args.out else Path("fetched_text.txt")
        print(f"Fetching: {args.url}")
        text = fetch_and_clean(args.url)
        out_path.write_text(text, encoding="utf-8")
        print(f"Written {len(text)} chars → {out_path}")

    else:
        config_path = Path(args.config)
        config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        outdir = Path(args.outdir)
        outdir.mkdir(parents=True, exist_ok=True)

        chapters = config.get("chapters", [])
        for i, ch in enumerate(chapters):
            url = ch.get("text_url", "")
            if not url or "FILL_IN" in url:
                print(f"[{i+1}/{len(chapters)}] Skipping '{ch.get('chapter')}' — no text_url")
                continue

            slug = re.sub(r"[^\w]+", "_", ch.get("chapter", f"chapter_{i+1}"))[:50]
            out_path = outdir / f"{slug}.txt"

            print(f"[{i+1}/{len(chapters)}] Fetching '{ch.get('chapter')}' → {out_path}")
            try:
                text = fetch_and_clean(url)
                out_path.write_text(text, encoding="utf-8")
                # Write the local path back into the config for align.py
                ch["text_file"] = str(out_path)
                print(f"  {len(text)} chars")
            except Exception as e:
                print(f"  ERROR: {e}", file=sys.stderr)

            if i < len(chapters) - 1:
                time.sleep(args.delay)

        # Update config with local text_file paths
        config_path.write_text(yaml.dump(config, allow_unicode=True, sort_keys=False),
                                encoding="utf-8")
        print(f"\nConfig updated with text_file paths: {config_path}")


if __name__ == "__main__":
    main()
