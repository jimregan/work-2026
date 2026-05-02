from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from pathlib import Path

from timit_utils import TIMIT_TO_IPA


def _write_mfa_dict(lexicon: dict[str, list[list[str]]], path: Path) -> None:
    with open(path, "w") as f:
        for word, pronunciations in sorted(lexicon.items()):
            for phones in pronunciations:
                f.write(f"{word}\t{' '.join(phones)}\n")


def _write_lab(words: list[str], path: Path) -> None:
    path.write_text(" ".join(words) + "\n")


def run_mfa(
    wav_path: str | Path,
    transcript_words: list[str],
    lexicon: dict[str, list[list[str]]],
    output_dir: str | Path,
    acoustic_model: str = "timit",
    tmp_root: str | Path | None = None,
) -> bool:
    """
    Run MFA forced alignment on a single utterance.

    Writes a temporary corpus directory with the .wav and .lab files,
    plus a per-run dictionary, then calls mfa align.

    Returns True iff a non-empty .TextGrid is produced for the file.
    """
    wav_path = Path(wav_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(dir=tmp_root) as tmp:
        tmp = Path(tmp)
        corpus_dir = tmp / "corpus"
        corpus_dir.mkdir()

        shutil.copy(wav_path, corpus_dir / wav_path.name)
        _write_lab(transcript_words, corpus_dir / wav_path.with_suffix(".lab").name)

        dict_path = tmp / "lexicon.dict"
        _write_mfa_dict(lexicon, dict_path)

        cmd = [
            "mfa", "align",
            "--clean",
            "--quiet",
            str(corpus_dir),
            str(dict_path),
            acoustic_model,
            str(output_dir),
        ]
        try:
            subprocess.run(cmd, check=True, capture_output=True)
        except (subprocess.CalledProcessError, FileNotFoundError):
            return False

    textgrid = output_dir / wav_path.with_suffix(".TextGrid").name
    return textgrid.exists() and textgrid.stat().st_size > 0


def _phones_to_ipa(phones: list[str]) -> list[str]:
    return [TIMIT_TO_IPA.get(p, p) for p in phones if TIMIT_TO_IPA.get(p, p) != "∅"]


def run_charsiu(
    wav_path: str | Path,
    phones: list[str],
    output_path: str | Path,
) -> bool:
    """
    Run Charsiu forced alignment with an explicit phone sequence.

    Converts TIMIT phones to IPA, then calls the charsiu aligner API.
    Returns True iff a non-empty TextGrid is produced.
    """
    try:
        from charsiu import charsiu_forced_aligner  # type: ignore
    except ImportError:
        raise RuntimeError("charsiu is not installed: pip install charsiu")

    ipa_phones = _phones_to_ipa(phones)
    if not ipa_phones:
        return False

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        aligner = charsiu_forced_aligner(aligner="charsiu/en_w2v2_fc_10ms")
        aligner.serve(
            audio=str(wav_path),
            phones=ipa_phones,
            output=str(output_path),
        )
    except Exception:
        return False

    return output_path.exists() and output_path.stat().st_size > 0
