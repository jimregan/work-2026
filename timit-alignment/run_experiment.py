from __future__ import annotations

import csv
import json
import random
import tempfile
from pathlib import Path

from alignment_runners import run_charsiu, run_mfa
from error_introduction import apply_edits
from timit_utils import (
    extract_file_lexicon,
    get_word_sequence,
    iter_timit_files,
    load_timit_lexicon,
)

RESULT_FIELDS = [
    "file",
    "aligner",
    "condition",
    "n_edits",
    "trial",
    "success",
    "edit_log_json",
]


def _oracle_phones_flat(oracle_lexicon: dict[str, list[list[str]]]) -> list[str]:
    """Flatten all oracle pronunciations into a single phone sequence (for Charsiu)."""
    return [p for prons in oracle_lexicon.values() for phones in prons for p in phones]


def _run_both(
    wav_path: Path,
    transcript_words: list[str],
    lexicon: dict[str, list[list[str]]],
    output_base: Path,
    label: str,
) -> dict[str, bool]:
    mfa_out = output_base / "mfa" / label
    charsiu_out = output_base / "charsiu" / label / wav_path.with_suffix(".TextGrid").name

    flat_phones = [p for prons in lexicon.values() for phones in prons for p in phones]

    return {
        "mfa": run_mfa(wav_path, transcript_words, lexicon, mfa_out),
        "charsiu": run_charsiu(wav_path, flat_phones, charsiu_out),
    }


def run_all(
    timit_root: str | Path,
    timit_lexicon_path: str | Path,
    output_csv: str | Path,
    n_trials_per_n_edits: int = 10,
    max_edits: int = 20,
    seed: int = 42,
    max_files: int | None = None,
) -> None:
    """
    Run the full degradation experiment over TIMIT.

    Conditions per file:
      canonical  – global TIMIT dictionary
      oracle     – per-file extracted lexicon (ground truth phones)
      degraded   – oracle with N random edits (N=1..max_edits,
                   repeated n_trials_per_n_edits times each)

    Results are appended to output_csv after each file so the run is resumable.
    """
    output_csv = Path(output_csv)
    global_lexicon = load_timit_lexicon(timit_lexicon_path)
    rng = random.Random(seed)

    write_header = not output_csv.exists()
    csv_file = open(output_csv, "a", newline="")
    writer = csv.DictWriter(csv_file, fieldnames=RESULT_FIELDS)
    if write_header:
        writer.writeheader()
        csv_file.flush()

    with tempfile.TemporaryDirectory() as tmp_root:
        tmp_root = Path(tmp_root)

        for i, (wav, phn, wrd) in enumerate(iter_timit_files(timit_root)):
            if max_files is not None and i >= max_files:
                break

            file_id = wav.stem
            words = get_word_sequence(wrd)
            oracle_lexicon = extract_file_lexicon(phn, wrd)

            # only keep words that appear in the transcript
            canonical_lexicon = {
                w: global_lexicon[w]
                for w in words
                if w in global_lexicon
            }

            output_base = tmp_root / file_id

            # --- canonical condition ---
            for aligner, success in _run_both(
                wav, words, canonical_lexicon, output_base, "canonical"
            ).items():
                writer.writerow({
                    "file": file_id,
                    "aligner": aligner,
                    "condition": "canonical",
                    "n_edits": 0,
                    "trial": 0,
                    "success": int(success),
                    "edit_log_json": "[]",
                })

            # --- oracle condition ---
            for aligner, success in _run_both(
                wav, words, oracle_lexicon, output_base, "oracle"
            ).items():
                writer.writerow({
                    "file": file_id,
                    "aligner": aligner,
                    "condition": "oracle",
                    "n_edits": 0,
                    "trial": 0,
                    "success": int(success),
                    "edit_log_json": "[]",
                })

            # --- degraded conditions ---
            for n_edits in range(1, max_edits + 1):
                for trial in range(n_trials_per_n_edits):
                    modified, edit_log = apply_edits(oracle_lexicon, n_edits, rng)
                    label = f"degraded_{n_edits}_{trial}"
                    for aligner, success in _run_both(
                        wav, words, modified, output_base, label
                    ).items():
                        writer.writerow({
                            "file": file_id,
                            "aligner": aligner,
                            "condition": "degraded",
                            "n_edits": n_edits,
                            "trial": trial,
                            "success": int(success),
                            "edit_log_json": json.dumps(edit_log),
                        })

            csv_file.flush()
            print(f"[{i+1}] {file_id} done")

    csv_file.close()
    print(f"Results written to {output_csv}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("timit_root")
    parser.add_argument("timit_lexicon")
    parser.add_argument("output_csv")
    parser.add_argument("--trials", type=int, default=10)
    parser.add_argument("--max-edits", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-files", type=int, default=None)
    args = parser.parse_args()

    run_all(
        timit_root=args.timit_root,
        timit_lexicon_path=args.timit_lexicon,
        output_csv=args.output_csv,
        n_trials_per_n_edits=args.trials,
        max_edits=args.max_edits,
        seed=args.seed,
        max_files=args.max_files,
    )
