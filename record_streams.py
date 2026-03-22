#!/usr/bin/env python3
"""Record HLS (m3u8) streams, saving raw segments as downloaded."""

import argparse
import hashlib
import io
import os
import signal
import threading
from pathlib import Path
from urllib.parse import urljoin, urlparse

import requests

try:
    from warcio.warcwriter import WARCWriter
    from warcio.statusandheaders import StatusAndHeaders
    WARCIO_AVAILABLE = True
except ImportError:
    WARCIO_AVAILABLE = False

STOP_EVENT = threading.Event()


def signal_handler(*_):
    print("\nStopping...", flush=True)
    STOP_EVENT.set()


def parse_master_m3u8(text, base_url):
    """If text is a master playlist, return the highest-bandwidth variant URL; else None."""
    variants = []
    lines = text.splitlines()
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if line.startswith("#EXT-X-STREAM-INF:"):
            bandwidth = 0
            for part in line.split(","):
                if part.upper().startswith("BANDWIDTH="):
                    try:
                        bandwidth = int(part.split("=", 1)[1])
                    except ValueError:
                        pass
            i += 1
            if i < len(lines):
                uri = lines[i].strip()
                if uri and not uri.startswith("#"):
                    variants.append((bandwidth, urljoin(base_url, uri)))
        i += 1
    if not variants:
        return None
    return max(variants, key=lambda x: x[0])[1]


def parse_m3u8(text, base_url):
    """Return (segments, target_duration) from a media playlist."""
    segments = []
    target_duration = 2
    lines = text.splitlines()
    for line in lines:
        line = line.strip()
        if line.startswith("#EXT-X-TARGETDURATION:"):
            try:
                target_duration = int(line.split(":", 1)[1])
            except ValueError:
                pass
        elif line and not line.startswith("#"):
            segments.append(urljoin(base_url, line))
    return segments, target_duration


def segment_name(url):
    """Extract filename from segment URL, stripping query string."""
    path = urlparse(url).path
    return os.path.basename(path) or hashlib.md5(url.encode()).hexdigest()


def make_warc_record(writer, resp, url):
    """Write a requests.Response to an open WARCWriter."""
    status_line = f"{resp.status_code} {resp.reason}"
    http_headers = StatusAndHeaders(status_line, list(resp.headers.items()), protocol="HTTP/1.1")
    payload = resp.content
    record = writer.create_warc_record(
        url,
        "response",
        payload=io.BytesIO(payload),
        length=len(payload),
        http_headers=http_headers,
    )
    writer.write_record(record)


def record_stream(url, out_dir, session, index, save_warc=False):
    """Poll a live m3u8 stream and save new segments to out_dir."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    seen = set()
    poll_interval = 2
    label = f"[stream {index}]"

    print(f"{label} Recording to {out_dir}", flush=True)

    warc_path = out_dir / "stream.warc.gz"
    warc_fh = open(warc_path, "ab") if save_warc else None
    writer = WARCWriter(warc_fh, gzip=True) if save_warc else None
    if save_warc:
        print(f"{label} WARC: {warc_path}", flush=True)

    try:
        resp = session.get(url, timeout=15)
        resp.raise_for_status()
        if writer is not None:
            make_warc_record(writer, resp, resp.url)
        variant_url = parse_master_m3u8(resp.text, resp.url)
        if variant_url is not None:
            print(f"{label} Master playlist → highest bitrate: {variant_url}", flush=True)
            url = variant_url
    except requests.RequestException as e:
        print(f"{label} Initial fetch error: {e}", flush=True)
        if warc_fh is not None:
            warc_fh.close()
        return

    try:
        while not STOP_EVENT.is_set():
            try:
                resp = session.get(url, timeout=15)
                resp.raise_for_status()
            except requests.RequestException as e:
                print(f"{label} Playlist fetch error: {e}", flush=True)
                STOP_EVENT.wait(poll_interval)
                continue

            if writer is not None:
                make_warc_record(writer, resp, resp.url)

            segments, target_duration = parse_m3u8(resp.text, resp.url)
            poll_interval = max(1, target_duration // 2)

            new_segments = [s for s in segments if s not in seen]
            for seg_url in new_segments:
                if STOP_EVENT.is_set():
                    break
                name = segment_name(seg_url)
                dest = out_dir / name
                if dest.exists():
                    seen.add(seg_url)
                    continue
                try:
                    seg_resp = session.get(seg_url, timeout=30)
                    seg_resp.raise_for_status()
                    dest.write_bytes(seg_resp.content)
                    if writer is not None:
                        make_warc_record(writer, seg_resp, seg_url)
                    print(f"{label} {name} ({len(seg_resp.content)} bytes)", flush=True)
                    seen.add(seg_url)
                except requests.RequestException as e:
                    print(f"{label} Segment error {name}: {e}", flush=True)

            STOP_EVENT.wait(poll_interval)
    finally:
        if warc_fh is not None:
            warc_fh.close()

    print(f"{label} Stopped.", flush=True)


def load_urls_from_file(path):
    with open(path) as f:
        return [line.strip() for line in f if line.strip() and not line.startswith("#")]


def main():
    parser = argparse.ArgumentParser(description="Record HLS streams to disk (raw segments).")
    parser.add_argument("urls", nargs="*", help="m3u8 stream URL(s)")
    parser.add_argument("-f", "--file", help="File with one m3u8 URL per line")
    parser.add_argument("-o", "--outdir", default="recordings", help="Base output directory (default: recordings)")
    parser.add_argument("--warc", action="store_true", help="Save WARC archive of all HTTP responses alongside segments")
    args = parser.parse_args()

    if args.warc and not WARCIO_AVAILABLE:
        parser.error("--warc requires the 'warcio' package: pip install warcio")

    urls = list(args.urls)
    if args.file:
        urls.extend(load_urls_from_file(args.file))

    if not urls:
        parser.error("Provide at least one URL or a --file.")

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    session = requests.Session()
    session.headers["User-Agent"] = "Mozilla/5.0"

    threads = []
    for i, url in enumerate(urls):
        name = f"stream_{i:02d}"
        out_dir = Path(args.outdir) / name
        t = threading.Thread(target=record_stream, args=(url, out_dir, session, i, args.warc), daemon=True)
        t.start()
        threads.append(t)

    for t in threads:
        t.join()


if __name__ == "__main__":
    main()
