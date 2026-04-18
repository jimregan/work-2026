from pathlib import Path
import wave

import pytest

datasets = pytest.importorskip("datasets")
load_from_disk = datasets.load_from_disk
from convert_fairseq_to_hf_dataset import (
    build_dataset,
    decode_fairseq_transcript,
    read_fairseq_manifest,
)


def write_test_wav(path: Path, sample_rate: int = 16000, num_samples: int = 160):
    with wave.open(str(path), "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(b"\x00\x00" * num_samples)


def test_decode_fairseq_transcript():
    assert decode_fairseq_transcript("h e l l o | w o r l d\n") == "hello world"


def test_manifest_round_trip(tmp_path):
    audio_root = tmp_path / "audio"
    audio_root.mkdir()

    wav_path = audio_root / "sample.wav"
    write_test_wav(wav_path, num_samples=160)

    tsv_path = tmp_path / "train.tsv"
    tsv_path.write_text(f"{audio_root}\nsample.wav\t160\n", encoding="utf-8")

    transcripts_path = tmp_path / "train.ltr"
    transcripts_path.write_text("h i | t h e r e\n", encoding="utf-8")

    entries = read_fairseq_manifest(tsv_path, transcripts_path)
    assert entries == [
        {
            "audio": str(wav_path),
            "path": str(wav_path),
            "num_samples": 160,
            "text": "hi there",
        }
    ]

    dataset = build_dataset(entries, sampling_rate=16000)
    output_dir = tmp_path / "hf_dataset"
    dataset.save_to_disk(output_dir)

    loaded = load_from_disk(output_dir)
    row = loaded[0]
    assert row["path"] == str(wav_path)
    assert row["num_samples"] == 160
    assert row["text"] == "hi there"
    assert row["audio"]["sampling_rate"] == 16000
