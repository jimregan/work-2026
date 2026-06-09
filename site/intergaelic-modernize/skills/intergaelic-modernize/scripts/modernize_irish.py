#!/usr/bin/env python3
"""
Modernize Irish text via the Cadhán intergaelic API.
Usage: echo "sean-téacs" | python modernize_irish.py
       python modernize_irish.py < file.txt
       python modernize_irish.py "inline text here"
"""
import urllib.parse
import urllib.error
import urllib.request
import json
import sys
import hashlib
from pathlib import Path

CACHE_DIR = Path.home() / '.cache' / 'intergaelic'


def cache_get(key: str):
    path = CACHE_DIR / f'{key}.json'
    if path.exists():
        return json.loads(path.read_text())
    return None


def cache_set(key: str, value):
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    (CACHE_DIR / f'{key}.json').write_text(json.dumps(value))


def make_request(text: str) -> list:
    key = hashlib.sha256(text.encode()).hexdigest()
    cached = cache_get(key)
    if cached is not None:
        return cached

    params = urllib.parse.urlencode({'foinse': 'ga', 'teacs': text})
    data = params.encode('ascii')
    headers = {
        'Content-Type': 'application/x-www-form-urlencoded',
        'Accept': 'application/json',
    }
    req = urllib.request.Request(
        'https://cadhan.com/api/intergaelic/3.0', data, headers=headers
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            pairs = json.loads(resp.read())
            cache_set(key, pairs)
            return pairs
    except urllib.error.HTTPError as e:
        print(f'HTTP error: {e.code}', file=sys.stderr)
    except urllib.error.URLError as e:
        print(f'Connection error: {e.reason}', file=sys.stderr)
    except ValueError:
        print('Malformed JSON from server', file=sys.stderr)
    return []


def modernize(text: str) -> str:
    """Return modernized Irish, falling back to original on failure."""
    pairs = make_request(text)
    # pairs is a list of [original, translation] — ga→ga gives modernized form
    if not pairs:
        return text
    # Collect modernized tokens and rejoin
    return ' '.join(modern for _, modern in pairs)


if __name__ == '__main__':
    if len(sys.argv) > 1:
        text = ' '.join(sys.argv[1:])
    else:
        text = sys.stdin.read()
    text = text.strip()
    if not text:
        print('No input provided.', file=sys.stderr)
        sys.exit(1)
    result = modernize(text)
    print(result)
