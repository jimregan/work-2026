"""Utilities for building CTC vocabularies from dataset label columns."""

import json
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path

from transformers import Wav2Vec2CTCTokenizer


SPECIAL_TOKEN_NAMES = ("pad_token", "bos_token", "eos_token", "unk_token")


def _iter_label_units(value) -> list[str]:
    if isinstance(value, str):
        return list(value.replace(" ", "|"))

    if isinstance(value, Sequence):
        units = []
        for item in value:
            if not isinstance(item, str):
                raise TypeError(f"Unsupported label unit type: {type(item).__name__}")
            units.append(item)
        return units

    raise TypeError(f"Unsupported label value type: {type(value).__name__}")


def collect_ctc_units(examples: Iterable, text_column: str) -> list[str]:
    units = set()

    for example in examples:
        if isinstance(example, Mapping):
            value = example[text_column]
        else:
            value = example
        units.update(_iter_label_units(value))

    return sorted(units)


def build_ctc_vocab(units: Iterable[str], tokenizer_template: Wav2Vec2CTCTokenizer) -> dict[str, int]:
    vocab = {}

    for token_name in SPECIAL_TOKEN_NAMES:
        token = getattr(tokenizer_template, token_name)
        if token is not None and token not in vocab:
            vocab[token] = len(vocab)

    for unit in sorted(set(units)):
        if unit not in vocab:
            vocab[unit] = len(vocab)

    word_delimiter = tokenizer_template.word_delimiter_token
    if word_delimiter is not None and word_delimiter not in vocab:
        vocab[word_delimiter] = len(vocab)

    return vocab


def save_ctc_tokenizer(
    vocab: dict[str, int],
    tokenizer_template: Wav2Vec2CTCTokenizer,
    output_dir: str | Path,
) -> Wav2Vec2CTCTokenizer:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    vocab_path = output_dir / "vocab.json"
    vocab_path.write_text(
        json.dumps(vocab, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    tokenizer = Wav2Vec2CTCTokenizer(
        str(vocab_path),
        bos_token=tokenizer_template.bos_token,
        eos_token=tokenizer_template.eos_token,
        unk_token=tokenizer_template.unk_token,
        pad_token=tokenizer_template.pad_token,
        word_delimiter_token=tokenizer_template.word_delimiter_token,
        replace_word_delimiter_char=tokenizer_template.replace_word_delimiter_char,
        do_lower_case=tokenizer_template.do_lower_case,
    )
    tokenizer.save_pretrained(output_dir)
    return tokenizer
