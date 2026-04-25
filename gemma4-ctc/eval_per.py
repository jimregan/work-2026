"""Evaluate a Gemma4 CTC checkpoint with phoneme error rate (PER)."""

import argparse
from pathlib import Path

import torch
from datasets import Dataset, DatasetDict, load_dataset, load_from_disk
from torch.utils.data import DataLoader
from transformers import AutoModelForCTC, Wav2Vec2CTCTokenizer
from transformers.models.gemma4.feature_extraction_gemma4 import Gemma4AudioFeatureExtractor

from collator import DataCollatorCTCWithPadding
from train import prepare_dataset


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", required=True)
    parser.add_argument("--dataset_name", required=True)
    parser.add_argument("--dataset_config", default=None)
    parser.add_argument("--split", default="test")
    parser.add_argument("--audio_column", default="audio")
    parser.add_argument("--text_column", default="phonemes")
    parser.add_argument("--audio_channel_strategy", default="error", choices=["error", "first"])
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--max_examples", type=int, default=None)
    return parser.parse_args()


def load_eval_dataset(dataset_name: str, dataset_config: str | None, split: str) -> Dataset:
    dataset_path = Path(dataset_name).expanduser()
    if dataset_path.exists():
        loaded = load_from_disk(str(dataset_path))
        if isinstance(loaded, DatasetDict):
            return loaded[split]
        if isinstance(loaded, Dataset):
            return loaded
        raise TypeError(f"Unsupported dataset type: {type(loaded).__name__}")
    return load_dataset(dataset_name, dataset_config, split=split)


def normalize_tokens(value) -> list[str]:
    if isinstance(value, str):
        return value.split()
    return list(value)


def edit_distance(ref: list[str], hyp: list[str]) -> int:
    rows = len(ref) + 1
    cols = len(hyp) + 1
    dp = [[0] * cols for _ in range(rows)]

    for i in range(rows):
        dp[i][0] = i
    for j in range(cols):
        dp[0][j] = j

    for i in range(1, rows):
        for j in range(1, cols):
            cost = 0 if ref[i - 1] == hyp[j - 1] else 1
            dp[i][j] = min(
                dp[i - 1][j] + 1,
                dp[i][j - 1] + 1,
                dp[i - 1][j - 1] + cost,
            )

    return dp[-1][-1]


def decode_prediction_ids(tokenizer, pred_ids: torch.Tensor) -> list[list[str]]:
    pad_token = tokenizer.pad_token
    word_delimiter = tokenizer.word_delimiter_token
    results = []

    for seq in pred_ids.tolist():
        tokens = tokenizer.convert_ids_to_tokens(seq)
        collapsed = []
        prev = None
        for token in tokens:
            if token == prev:
                continue
            prev = token
            if token == pad_token:
                continue
            collapsed.append(token)

        results.append([token for token in collapsed if token != word_delimiter])

    return results


def main():
    args = parse_args()

    dataset = load_eval_dataset(args.dataset_name, args.dataset_config, args.split)
    if args.max_examples is not None:
        dataset = dataset.select(range(min(args.max_examples, len(dataset))))

    feature_extractor = Gemma4AudioFeatureExtractor.from_pretrained(args.model_dir)
    tokenizer = Wav2Vec2CTCTokenizer.from_pretrained(args.model_dir)
    model = AutoModelForCTC.from_pretrained(args.model_dir, trust_remote_code=True)
    model.to(args.device)
    model.eval()

    processed = dataset.map(
        prepare_dataset,
        fn_kwargs={
            "feature_extractor": feature_extractor,
            "tokenizer": tokenizer,
            "audio_column": args.audio_column,
            "text_column": args.text_column,
            "audio_channel_strategy": args.audio_channel_strategy,
        },
        remove_columns=dataset.column_names,
        num_proc=4,
    )

    collator = DataCollatorCTCWithPadding(
        feature_extractor=feature_extractor,
        tokenizer=tokenizer,
    )
    dataloader = DataLoader(
        processed,
        batch_size=args.batch_size,
        collate_fn=collator,
    )

    total_edits = 0
    total_ref_tokens = 0

    offset = 0
    with torch.no_grad():
        for batch in dataloader:
            batch_size = batch["labels"].shape[0]
            refs = [
                normalize_tokens(dataset[offset + i][args.text_column])
                for i in range(batch_size)
            ]
            offset += batch_size

            batch = {
                key: value.to(args.device) if isinstance(value, torch.Tensor) else value
                for key, value in batch.items()
            }
            logits = model(**batch).logits
            pred_ids = torch.argmax(logits, dim=-1).cpu()
            hyps = decode_prediction_ids(tokenizer, pred_ids)

            for ref, hyp in zip(refs, hyps, strict=True):
                total_edits += edit_distance(ref, hyp)
                total_ref_tokens += len(ref)

    per = total_edits / total_ref_tokens if total_ref_tokens else 0.0
    print(f"split={args.split}")
    print(f"examples={len(dataset)}")
    print(f"ref_tokens={total_ref_tokens}")
    print(f"edits={total_edits}")
    print(f"per={per:.6f}")


if __name__ == "__main__":
    main()
