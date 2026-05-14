"""Convert a fairseq TSV manifest plus transcript file into a HF dataset."""

import argparse
from pathlib import Path

from datasets import Audio, Dataset, DatasetDict


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tsv", required=True, help="Fairseq manifest .tsv file")
    parser.add_argument("--transcripts", required=True, help="Transcript file matching the TSV order")
    parser.add_argument("--output_dir", required=True, help="Where to save the dataset with save_to_disk()")
    parser.add_argument("--split", default="train", help="Dataset split name to write")
    parser.add_argument("--sampling_rate", type=int, default=None, help="Optional sampling rate for the Audio feature")
    return parser.parse_args()


def parse_fairseq_transcript(line: str) -> list[str]:
    return line.strip().split()


def format_fairseq_transcript(tokens: list[str]) -> str:
    return " ".join(tokens)


def read_fairseq_manifest(tsv_path: str | Path, transcripts_path: str | Path) -> list[dict]:
    tsv_path = Path(tsv_path)
    transcripts_path = Path(transcripts_path)

    with tsv_path.open("r", encoding="utf-8") as tsv_file:
        rows = [line.rstrip("\n") for line in tsv_file]

    if not rows:
        raise ValueError(f"Manifest is empty: {tsv_path}")

    audio_root = Path(rows[0])
    if not audio_root.is_absolute():
        raise ValueError(f"First TSV line must be an absolute path, got: {audio_root}")

    manifest_entries = []
    for line_number, row in enumerate(rows[1:], start=2):
        if not row:
            continue
        try:
            rel_path, num_samples = row.split("\t")
        except ValueError as exc:
            raise ValueError(
                f"Invalid TSV row at line {line_number}: expected '<filename><TAB><number of samples>'"
            ) from exc

        audio_path = audio_root / rel_path
        manifest_entries.append(
            {
                "audio": str(audio_path),
                "path": str(audio_path),
                "num_samples": int(num_samples),
            }
        )

    with transcripts_path.open("r", encoding="utf-8") as transcripts_file:
        transcripts = [parse_fairseq_transcript(line) for line in transcripts_file]

    if len(manifest_entries) != len(transcripts):
        raise ValueError(
            f"Manifest/transcript length mismatch: {len(manifest_entries)} entries vs {len(transcripts)} transcripts"
        )

    for entry, transcript_tokens in zip(manifest_entries, transcripts, strict=True):
        entry["text"] = format_fairseq_transcript(transcript_tokens)
        entry["phonemes"] = transcript_tokens

    return manifest_entries


def build_dataset(entries: list[dict], sampling_rate: int | None = None) -> Dataset:
    dataset = Dataset.from_list(entries)
    audio_kwargs = {} if sampling_rate is None else {"sampling_rate": sampling_rate}
    return dataset.cast_column("audio", Audio(**audio_kwargs))


def main():
    args = parse_args()
    entries = read_fairseq_manifest(args.tsv, args.transcripts)
    dataset = build_dataset(entries, sampling_rate=args.sampling_rate)
    DatasetDict({args.split: dataset}).save_to_disk(args.output_dir)


if __name__ == "__main__":
    main()
