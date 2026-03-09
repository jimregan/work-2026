"""Cross-dataset sentence alignment by normalised text.

Produces a canonical sentence table that maps sentence text to a stable
``content_label`` shared across GBI, VCTK, and CMU Arctic.

Canonical ID assignment
-----------------------
* Sentences present in GBI → the GBI sentence ID is the canonical ID.
* Sentences in VCTK only   → ``vctk_{text_id}``.
* Sentences in Arctic only → ``arctic_{arctic_id}``.

The canonical table is a dict keyed by canonical ID and suitable for
serialising to a TSV / JSON used at dataset-build time.
"""

from __future__ import annotations

import re
import unicodedata
from pathlib import Path


def normalize_text(text: str) -> str:
    """Canonical form for matching: NFD → ASCII fold, lower, strip punctuation."""
    text = unicodedata.normalize("NFD", text)
    text = text.encode("ascii", "ignore").decode("ascii")
    text = text.lower()
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def build_vctk_index(hf_dataset) -> dict[str, str]:
    """Return {normalised_text: text_id} from a loaded VCTK HF dataset.

    Only the first occurrence of each text is kept (VCTK has many speakers
    per sentence; they all share the same text_id).
    """
    index: dict[str, str] = {}
    for row in hf_dataset:
        norm = normalize_text(row["text"])
        if norm and norm not in index:
            index[norm] = row["text_id"]
    return index


def build_arctic_index(hf_dataset) -> tuple[dict[str, str], dict[str, str]]:
    """Return ({normalised_text: sentence_id}, {normalised_text: original_text}).

    Arctic sentence IDs are not globally unique content identifiers (the same
    sentence can appear under different IDs across speakers), so we match by
    normalised text and keep only the first sentence_id seen for each text.
    """
    sid_index: dict[str, str] = {}
    text_index: dict[str, str] = {}
    for row in hf_dataset:
        norm = normalize_text(row["text"])
        if norm and norm not in sid_index:
            sid_index[norm] = row["sentence_id"]
            text_index[norm] = row["text"]
    return sid_index, text_index


def align_sentences(
    vctk_index: dict[str, str],
    vctk_text_by_id: dict[str, str],
    gbi_rows: list[dict],
    arctic_index: dict[str, str] | None = None,
    arctic_text_by_norm: dict[str, str] | None = None,
) -> dict[str, dict]:
    """Build the canonical sentence table.

    Parameters
    ----------
    vctk_index:
        {normalised_text: text_id} from :func:`build_vctk_index`.
    vctk_text_by_id:
        {text_id: original_text} — needed to record the text for VCTK-only rows.
    gbi_rows:
        List of dicts with at least ``sentence_id`` and ``text`` keys,
        as returned by :func:`build_gbi.read_gbi_csv`.
    arctic_index:
        Optional {normalised_text: sentence_id} from :func:`build_arctic_index`.

    Returns
    -------
    dict keyed by canonical_id, each value a dict with keys:
        text, vctk_text_id, gbi_sentence_id, arctic_id
    """
    table: dict[str, dict] = {}
    matched_vctk_ids: set[str] = set()
    matched_arctic_norms: set[str] = set()

    for row in gbi_rows:
        gbi_sid = row["sentence_id"]
        norm = normalize_text(row["text"])
        vctk_tid = vctk_index.get(norm)
        arctic_sid = arctic_index.get(norm) if arctic_index else None

        table[gbi_sid] = {
            "text":            row["text"],
            "vctk_text_id":    vctk_tid,
            "gbi_sentence_id": gbi_sid,
            "arctic_id":       arctic_sid,
        }
        if vctk_tid:
            matched_vctk_ids.add(vctk_tid)
        if arctic_sid:
            matched_arctic_norms.add(norm)

    for norm, vctk_tid in vctk_index.items():
        if vctk_tid not in matched_vctk_ids:
            arctic_sid = arctic_index.get(norm) if arctic_index else None
            canonical_id = f"vctk_{vctk_tid}"
            table[canonical_id] = {
                "text":            vctk_text_by_id.get(vctk_tid, ""),
                "vctk_text_id":    vctk_tid,
                "gbi_sentence_id": None,
                "arctic_id":       arctic_sid,
            }
            if arctic_sid:
                matched_arctic_norms.add(norm)

    if arctic_index:
        for norm, arctic_sid in arctic_index.items():
            if norm not in matched_arctic_norms:
                canonical_id = f"arctic_{arctic_sid}"
                table[canonical_id] = {
                    "text":            arctic_text_by_norm.get(norm, "") if arctic_text_by_norm else "",
                    "vctk_text_id":    None,
                    "gbi_sentence_id": None,
                    "arctic_id":       arctic_sid,
                }

    return table


def write_tsv(table: dict[str, dict], path: Path) -> None:
    import csv
    fieldnames = ["canonical_id", "text", "vctk_text_id", "gbi_sentence_id", "arctic_id"]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t")
        w.writeheader()
        for canonical_id, row in sorted(table.items()):
            w.writerow({"canonical_id": canonical_id, **row})


def build_gbi_index_from_hf(hf_dataset) -> list[dict]:
    """Extract unique GBI sentence rows from a loaded HF dataset.

    Returns one dict per unique sentence_id with keys ``sentence_id`` and ``text``.
    """
    seen: dict[str, dict] = {}
    for row in hf_dataset:
        sid = row["sentence_id"]
        if sid not in seen:
            seen[sid] = {"sentence_id": sid, "text": row["text"]}
    return list(seen.values())


if __name__ == "__main__":
    import argparse
    from datasets import load_dataset

    parser = argparse.ArgumentParser(description="Align sentences across GBI and VCTK.")
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--gbi-csv", type=Path, metavar="CSV",
                     help="Path to GBI metadata CSV (line_index_all.csv).")
    src.add_argument("--gbi-hf", metavar="REPO_ID", default=None,
                     help="Load GBI from a HuggingFace dataset (e.g. kth-tmh/google-britain-ireland).")
    parser.add_argument("--arctic-hf", metavar="REPO_ID", default="kth-tmh/cmu_arctic",
                        help="CMU Arctic HF dataset (default: kth-tmh/cmu_arctic). Pass empty string to skip.")
    parser.add_argument("--out", type=Path, default=Path("sentence_table.tsv"))
    args = parser.parse_args()

    print("Loading VCTK from HuggingFace…")
    vctk = load_dataset("kth-tmh/vctk", split="train")

    vctk_index = build_vctk_index(vctk)
    vctk_text_by_id: dict[str, str] = {}
    for row in vctk:
        vctk_text_by_id.setdefault(row["text_id"], row["text"])

    print(f"VCTK: {len(vctk_index)} unique sentences.")

    if args.gbi_hf:
        print(f"Loading GBI from HuggingFace ({args.gbi_hf})…")
        gbi_ds = load_dataset(args.gbi_hf, split="train")
        unique_gbi_rows = build_gbi_index_from_hf(gbi_ds)
    else:
        from build_gbi import read_gbi_csv
        unique_gbi_rows = list({r["sentence_id"]: r for r in read_gbi_csv(args.gbi_csv)}.values())

    print(f"GBI: {len(unique_gbi_rows)} unique sentence IDs.")

    arctic_index = arctic_text_by_norm = None
    if args.arctic_hf:
        print(f"Loading CMU Arctic from HuggingFace ({args.arctic_hf})…")
        arctic_ds = load_dataset(args.arctic_hf, split="train")
        arctic_index, arctic_text_by_norm = build_arctic_index(arctic_ds)
        print(f"Arctic: {len(arctic_index)} unique sentences.")

    table = align_sentences(vctk_index, vctk_text_by_id, unique_gbi_rows,
                            arctic_index=arctic_index,
                            arctic_text_by_norm=arctic_text_by_norm)

    matched      = sum(1 for v in table.values() if v["vctk_text_id"] and v["gbi_sentence_id"])
    vctk_only    = sum(1 for v in table.values() if v["vctk_text_id"] and not v["gbi_sentence_id"])
    gbi_only     = sum(1 for v in table.values() if not v["vctk_text_id"] and v["gbi_sentence_id"])
    arctic_only  = sum(1 for v in table.values() if v["arctic_id"] and not v["gbi_sentence_id"] and not v["vctk_text_id"])
    print(f"Matched (GBI∩VCTK): {matched}  |  VCTK-only: {vctk_only}  |  GBI-only: {gbi_only}  |  Arctic-only: {arctic_only}")

    write_tsv(table, args.out)
    print(f"Written to {args.out}")
