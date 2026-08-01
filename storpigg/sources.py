"""Load every view of each announcement so the slots can be cross-checked.

Five sources per id:
  transcript  the partially-corrected target (numerals spelled out)
  messate     the metadata sheet's own text column (numerals as digits; itself noisy)
  whisperx    ASR, cased/punctuated, digits
  wav2vec     ASR, uppercase, numerals spelled out, truncates word-finally
plus two structured metadata fields that are genuine ground truth:
  narrator    "Lastname, Firstname"
  media_year
"""
import csv
import os

import slots

ROOT = "/tmp/storpigg"
TSV = os.path.join(ROOT, "Untitled spreadsheet - Sheet1.tsv")
TRANSCRIPT = os.path.join(ROOT, "storspigg_transcript.txt")


def narrator_to_spoken(narrator):
    """'Andersson, Sus' -> ['sus', 'andersson'] (spoken order)."""
    if not narrator:
        return None
    if "," in narrator:
        last, first = narrator.split(",", 1)
        name = f"{first.strip()} {last.strip()}"
    else:
        name = narrator
    return slots.norm_tokens(name)


def load():
    rows = {}

    with open(TSV, newline="") as fh:
        for r in csv.DictReader(fh, delimiter="\t"):
            rows[r["id"]] = {
                "narrator": r["narrator"],
                "narrator_tokens": narrator_to_spoken(r["narrator"]),
                "media_year": r["media_year"],
                "messate_raw": r["messate"],
                "messate": slots.norm_tokens(r["messate"]),
            }

    with open(TRANSCRIPT) as fh:
        for line in fh:
            if not line.strip():
                continue
            key, _, text = line.partition("\t")
            rows.setdefault(key, {})
            rows[key]["transcript_raw"] = text.strip()
            rows[key]["transcript"] = slots.norm_tokens(text)

    asr = slots.load_all(ROOT)
    for key, v in asr.items():
        rows.setdefault(key, {})
        rows[key]["whisperx"] = v["whisperx"]
        rows[key]["wav2vec"] = v["wav2vec"]

    return rows


TEXT_SOURCES = ("transcript", "messate", "whisperx", "wav2vec")
