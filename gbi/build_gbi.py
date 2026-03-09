"""Build a HuggingFace Dataset from local GBI audio files + metadata CSV.

GBI CSV format (comma-separated, no header)
--------------------------------------------
<sentence_id>, <utterance_id>, <text>

where utterance_id = {we,mi,ir,no,sc,so}{m,f}_{speaker_id}_{utterance_num}

Example row::

    s001, wef_042_001, Please call Stella.

Output HF dataset columns
--------------------------
speaker_id       str   e.g. "gbi_wef_042"
utterance_id     str   original utterance ID for traceability
sentence_id      str   GBI sentence identifier (= content_label before alignment)
text             str
dialect          str   canonical label from dialect_map.GBI_CODE_TO_DIALECT
gender           str   "M" or "F"
audio            Audio loaded from <audio_dir>/<utterance_id>.wav (or .flac)
"""

from __future__ import annotations

import re
from pathlib import Path

from dialect_map import GBI_CODE_TO_DIALECT

UTTERANCE_RE = re.compile(r"^(we|mi|ir|no|sc|so)(m|f)_(.+)_(\d+)$")


def parse_utterance_id(uid: str) -> dict:
    m = UTTERANCE_RE.match(uid)
    if not m:
        raise ValueError(f"Unrecognised utterance_id format: {uid!r}")
    dialect_code, gender_char, speaker_part, utt_num = m.groups()
    return {
        "dialect":    GBI_CODE_TO_DIALECT[dialect_code],
        "gender":     "M" if gender_char == "m" else "F",
        "speaker_id": f"gbi_{dialect_code}{gender_char}_{speaker_part}",
    }


def read_gbi_csv(path: Path) -> list[dict]:
    """Parse a GBI metadata CSV file.  Returns one dict per row."""
    rows = []
    with open(path, encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            parts = line.split(",", 2)
            if len(parts) != 3:
                raise ValueError(f"Line {lineno}: expected 3 comma-separated fields, got {len(parts)}: {line!r}")
            sentence_id, utterance_id, text = (p.strip() for p in parts)
            rows.append({
                "sentence_id":  sentence_id,
                "utterance_id": utterance_id,
                "text":         text,
                **parse_utterance_id(utterance_id),
            })
    return rows


def find_audio(audio_dir: Path, utterance_id: str) -> Path | None:
    """Locate the audio file for an utterance (tries .wav then .flac)."""
    for ext in (".wav", ".flac"):
        p = audio_dir / f"{utterance_id}{ext}"
        if p.exists():
            return p
    return None


def build_dataset(csv_path: Path, audio_dir: Path, sentence_table_path: Path | None = None):
    """Build and return a HuggingFace Dataset from a GBI CSV + audio directory.

    Parameters
    ----------
    csv_path:
        Path to the GBI metadata CSV.
    audio_dir:
        Directory containing audio files named ``{utterance_id}.wav`` or ``.flac``.
    sentence_table_path:
        Optional TSV produced by sentence_align.py.  When provided, a
        ``content_label`` column is added (canonical cross-dataset sentence ID).
    """
    import datasets

    rows = read_gbi_csv(csv_path)

    content_label_map: dict[str, str] = {}
    if sentence_table_path is not None:
        import csv
        with open(sentence_table_path) as f:
            for row in csv.DictReader(f, delimiter="\t"):
                if row["gbi_sentence_id"]:
                    content_label_map[row["gbi_sentence_id"]] = row["canonical_id"]

    records = []
    missing_audio = 0
    for row in rows:
        audio_path = find_audio(audio_dir, row["utterance_id"])
        if audio_path is None:
            missing_audio += 1
            continue
        record = {
            "speaker_id":    row["speaker_id"],
            "utterance_id":  row["utterance_id"],
            "sentence_id":   row["sentence_id"],
            "text":          row["text"],
            "dialect":       row["dialect"],
            "gender":        row["gender"],
            "audio":         str(audio_path),
        }
        if content_label_map:
            record["content_label"] = content_label_map.get(row["sentence_id"], f"gbi_{row['sentence_id']}")
        records.append(record)

    if missing_audio:
        print(f"Warning: {missing_audio} utterances had no matching audio file.")

    features = datasets.Features({
        "speaker_id":   datasets.Value("string"),
        "utterance_id": datasets.Value("string"),
        "sentence_id":  datasets.Value("string"),
        "text":         datasets.Value("string"),
        "dialect":      datasets.Value("string"),
        "gender":       datasets.Value("string"),
        "audio":        datasets.Audio(),
        **({"content_label": datasets.Value("string")} if content_label_map else {}),
    })

    return datasets.Dataset.from_list(records, features=features)


def enrich_hf_dataset(repo_id: str, sentence_table_path: Path) -> "datasets.Dataset":
    """Load an existing GBI HF dataset and add a ``content_label`` column.

    Parameters
    ----------
    repo_id:
        HuggingFace dataset repo ID, e.g. ``"kth-tmh/google-britain-ireland"``.
    sentence_table_path:
        sentence_table.tsv produced by sentence_align.py.
    """
    import csv
    import datasets

    ds = datasets.load_dataset(repo_id, split="train")

    content_label_map: dict[str, str] = {}
    with open(sentence_table_path) as f:
        for row in csv.DictReader(f, delimiter="\t"):
            if row["gbi_sentence_id"]:
                content_label_map[row["gbi_sentence_id"]] = row["canonical_id"]

    def _add_label(row):
        row["content_label"] = content_label_map.get(row["sentence_id"],
                                                      f"gbi_{row['sentence_id']}")
        return row

    return ds.map(_add_label)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Build or enrich GBI HF dataset.")
    mode = parser.add_subparsers(dest="mode", required=True)

    p_build = mode.add_parser("build", help="Build dataset from local CSV + audio.")
    p_build.add_argument("csv", type=Path, help="GBI metadata CSV file.")
    p_build.add_argument("audio_dir", type=Path, help="Directory containing audio files.")
    p_build.add_argument("--sentence-table", type=Path, default=None,
                         help="sentence_table.tsv (adds content_label).")
    p_build.add_argument("--out", type=Path, default=Path("gbi_dataset"))
    p_build.add_argument("--push-to-hub", metavar="REPO_ID", default=None)

    p_enrich = mode.add_parser("enrich", help="Add content_label to existing HF dataset.")
    p_enrich.add_argument("repo_id", help="HF repo ID, e.g. kth-tmh/google-britain-ireland.")
    p_enrich.add_argument("sentence_table", type=Path, help="sentence_table.tsv.")
    p_enrich.add_argument("--out", type=Path, default=Path("gbi_enriched"))
    p_enrich.add_argument("--push-to-hub", metavar="REPO_ID", default=None)

    args = parser.parse_args()

    if args.mode == "build":
        ds = build_dataset(args.csv, args.audio_dir, args.sentence_table)
    else:
        ds = enrich_hf_dataset(args.repo_id, args.sentence_table)

    print(ds)
    ds.save_to_disk(str(args.out))
    print(f"Saved to {args.out}")

    if args.push_to_hub:
        ds.push_to_hub(args.push_to_hub)
        print(f"Pushed to {args.push_to_hub}")
