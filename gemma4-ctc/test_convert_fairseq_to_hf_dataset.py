from pathlib import Path
import wave

import pytest
from transformers import Wav2Vec2CTCTokenizer

datasets = pytest.importorskip("datasets")
load_from_disk = datasets.load_from_disk
from convert_fairseq_to_hf_dataset import (
    build_dataset,
    format_fairseq_transcript,
    parse_fairseq_transcript,
    read_fairseq_manifest,
)


def write_test_wav(path: Path, sample_rate: int = 16000, num_samples: int = 160):
    with wave.open(str(path), "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(b"\x00\x00" * num_samples)


def test_parse_fairseq_transcript():
    assert parse_fairseq_transcript("h e l l o | w o r l d\n") == [
        "h", "e", "l", "l", "o", "|", "w", "o", "r", "l", "d"
    ]


def test_format_fairseq_transcript():
    assert format_fairseq_transcript(["ɑː", "b", "|", "d"]) == "ɑː b | d"


def test_pretokenized_labels_work_with_local_tokenizer():
    tokenizer = Wav2Vec2CTCTokenizer.from_pretrained(".")
    encoded = tokenizer(["ɑː", "b", "|", "d"], is_split_into_words=True).input_ids
    assert tokenizer.convert_ids_to_tokens(encoded) == ["ɑː", "b", "|", "d"]


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
            "text": "h i | t h e r e",
            "phonemes": ["h", "i", "|", "t", "h", "e", "r", "e"],
        }
    ]

    dataset = build_dataset(entries, sampling_rate=16000)
    output_dir = tmp_path / "hf_dataset"
    dataset.save_to_disk(output_dir)

    loaded = load_from_disk(output_dir)
    row = loaded[0]
    assert row["path"] == str(wav_path)
    assert row["num_samples"] == 160
    assert row["text"] == "h i | t h e r e"
    assert row["phonemes"] == ["h", "i", "|", "t", "h", "e", "r", "e"]
    assert row["audio"]["sampling_rate"] == 16000
