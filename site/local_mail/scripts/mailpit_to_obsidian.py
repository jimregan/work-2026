#!/usr/bin/env python3
"""
mailpit_to_obsidian.py — Export Mailpit threads as Obsidian markdown files.

Each message becomes one .md file with YAML front matter and Obsidian wiki
links to the previous (and next, if known) message in the thread.

Usage:
    python mailpit_to_obsidian.py --output-dir /path/to/threads
    python mailpit_to_obsidian.py --output-dir /path/to/threads --mailpit-url http://localhost:8025
"""
import argparse
import json
import re
from datetime import datetime
from pathlib import Path
from urllib.request import urlopen


def fetch_json(url):
    with urlopen(url) as r:
        return json.load(r)


def get_all_messages(base_url):
    messages = []
    page = 1
    while True:
        data = fetch_json(f"{base_url}/api/v1/messages?limit=100&page={page}")
        batch = data.get("messages") or []
        if not batch:
            break
        messages.extend(batch)
        if len(batch) < 100:
            break
        page += 1
    return messages


def make_filename(msg, from_name):
    dt = datetime.fromisoformat(msg["Created"].replace("Z", "+00:00"))
    date_str = dt.strftime("%Y-%m-%d %H-%M")
    subject = re.sub(r"^Re:\s*", "", msg["Subject"])
    subject_slug = re.sub(r'[<>:"/\\|?*]', "", subject)[:50].strip()
    return f"{date_str} [{from_name}] {subject_slug}.md"


def main():
    parser = argparse.ArgumentParser(
        description="Export Mailpit threads as Obsidian markdown."
    )
    parser.add_argument(
        "--output-dir", required=True, help="Directory to write markdown files"
    )
    parser.add_argument(
        "--mailpit-url",
        default="http://localhost:8025",
        help="Mailpit base URL (default: http://localhost:8025)",
    )
    args = parser.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    base = args.mailpit_url.rstrip("/")

    print("Fetching message list...")
    messages = get_all_messages(base)
    messages.sort(key=lambda m: m["Created"])
    print(f"  {len(messages)} messages found")

    # First pass: collect headers and assign filenames
    records = {}
    for msg in messages:
        mailpit_id = msg["ID"]
        from_addr = msg["From"]["Address"]
        from_name = "claude" if "claude" in from_addr else "jim"
        fname = make_filename(msg, from_name)

        headers = fetch_json(f"{base}/api/v1/message/{mailpit_id}/headers")
        raw_id = headers.get("Message-Id", [mailpit_id])[0]
        msg_id = raw_id.strip("<> ")

        in_reply_to_raw = headers.get("In-Reply-To", [])
        parent_id = in_reply_to_raw[0].strip("<> ") if in_reply_to_raw else None

        records[msg_id] = {
            "mailpit_id": mailpit_id,
            "msg": msg,
            "filename": fname,
            "msg_id": msg_id,
            "parent_id": parent_id,
        }

    # Build reverse map: parent_id → child msg_ids (for forward links)
    children = {}
    for msg_id, rec in records.items():
        p = rec["parent_id"]
        if p and p in records:
            children.setdefault(p, []).append(msg_id)

    # Second pass: fetch bodies and write files
    print("Writing markdown files...")
    for msg_id, rec in records.items():
        msg = rec["msg"]
        fname = rec["filename"]
        from_addr = msg["From"]["Address"]
        display_from = "Jim O'Regan" if "you@" in from_addr else "Claude"

        dt = datetime.fromisoformat(msg["Created"].replace("Z", "+00:00"))

        nav_lines = []
        parent = rec["parent_id"]
        if parent and parent in records:
            prev_stem = Path(records[parent]["filename"]).stem
            nav_lines.append(f"← [[{prev_stem}]]")
        for child_id in children.get(msg_id, []):
            next_stem = Path(records[child_id]["filename"]).stem
            nav_lines.append(f"→ [[{next_stem}]]")

        nav_block = "\n".join(nav_lines) + "\n\n" if nav_lines else ""

        full = fetch_json(f"{base}/api/v1/message/{rec['mailpit_id']}")
        body = full.get("Text", "")

        content = (
            f"---\n"
            f'subject: "{msg["Subject"]}"\n'
            f"from: {display_from}\n"
            f"date: {dt.strftime('%Y-%m-%d %H:%M')}\n"
            f"message_id: {msg_id}\n"
            f"---\n\n"
            f"{nav_block}"
            f"{body}\n"
        )

        filepath = out / fname
        filepath.write_text(content, encoding="utf-8")
        print(f"  {fname}")

    print(f"\nDone — {len(records)} files written to {out}")


if __name__ == "__main__":
    main()
