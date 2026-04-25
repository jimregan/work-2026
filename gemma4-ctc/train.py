"""
Fine-tune Gemma4ForCTC for CTC ASR using HuggingFace Accelerate.

Usage:
    accelerate launch train.py [options]

The encoder is frozen by default; only the CTC head is trained.
Pass --unfreeze_norms to also train the encoder's layer norms.

On the first run, the Gemma 4 multimodal checkpoint is downloaded to
extract the audio encoder.  config.gemma4_audio_model_id accepts either
a Hub repo ID or a local path.  Checkpoints saved to output_dir are
self-contained: reloading them does not require the upstream model.
"""

import argparse
import json
import math
import os
import sys
from pathlib import Path
from types import SimpleNamespace

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import numpy as np
import torch
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from datasets import Dataset, DatasetDict, load_dataset, load_from_disk
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import AutoConfig, AutoModelForCTC, Wav2Vec2CTCTokenizer, get_scheduler
from transformers.models.gemma4.feature_extraction_gemma4 import Gemma4AudioFeatureExtractor

from collator import DataCollatorCTCWithPadding
from ctc_vocab import build_ctc_vocab, collect_ctc_units, save_ctc_tokenizer


logger = get_logger(__name__)
DEFAULT_MODEL_DIR = str(SCRIPT_DIR)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", default=DEFAULT_MODEL_DIR,
                        help="Repo dir containing config.json, vocab.json, etc.")
    parser.add_argument("--dataset_name", required=True)
    parser.add_argument("--dataset_config", default=None)
    parser.add_argument("--base_model_id", default=None,
                        help="Optional Gemma checkpoint override for the audio tower")
    parser.add_argument("--train_split", default="train")
    parser.add_argument("--eval_split", default="validation")
    parser.add_argument("--audio_column", default="audio")
    parser.add_argument("--text_column", default="text")
    parser.add_argument("--audio_channel_strategy", default="error",
                        choices=["error", "first"],
                        help="How to handle multi-channel audio: fail or take the first channel")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--rebuild_vocab", action="store_true",
                        help="Rebuild the CTC tokenizer vocab from the dataset text column")
    parser.add_argument("--num_train_epochs", type=int, default=10)
    parser.add_argument("--per_device_train_batch_size", type=int, default=8)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=8)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--warmup_ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save_steps", type=int, default=500)
    parser.add_argument("--eval_steps", type=int, default=500)
    parser.add_argument("--unfreeze_norms", action="store_true",
                        help="Keep encoder layer norms trainable instead of freezing everything")
    parser.add_argument("--gradient_checkpointing", action="store_true")
    return parser.parse_args()


def prepare_dataset(
    batch,
    feature_extractor,
    tokenizer,
    audio_column,
    text_column,
    audio_channel_strategy,
):
    audio = batch[audio_column]
    labels = batch[text_column]
    audio_path = batch.get("path", "<unknown>")

    if hasattr(audio, "get_all_samples"):
        samples = audio.get_all_samples()
        speech = samples.data.detach().cpu().numpy().astype(np.float32, copy=False)
        sampling_rate = int(samples.sample_rate)
    elif isinstance(audio, dict) and "array" in audio and "sampling_rate" in audio:
        speech = np.asarray(audio["array"], dtype=np.float32)
        sampling_rate = int(audio["sampling_rate"])
    else:
        raise TypeError(
            f"Unsupported audio object for {audio_path}: {type(audio).__name__}"
        )

    if speech.ndim == 2:
        if speech.shape[0] == 1 or speech.shape[1] == 1:
            speech = speech.reshape(-1)
        elif audio_channel_strategy == "first":
            speech = speech[0] if speech.shape[0] <= speech.shape[1] else speech[:, 0]
        else:
            raise ValueError(
                f"Multi-channel decoded audio encountered: path={audio_path}, sampling_rate={sampling_rate}, shape={speech.shape}"
            )
    elif speech.ndim != 1:
        raise ValueError(
            f"Unexpected decoded audio rank: path={audio_path}, sampling_rate={sampling_rate}, shape={speech.shape}"
        )

    batch["input_features"] = feature_extractor(
        speech, sampling_rate=sampling_rate
    ).input_features[0]

    if isinstance(labels, str):
        batch["labels"] = tokenizer(labels).input_ids
    else:
        batch["labels"] = tokenizer(labels, is_split_into_words=True).input_ids

    return batch


def is_ctc_model_dir(model_dir: str | os.PathLike) -> bool:
    config_path = Path(model_dir) / "config.json"
    if not config_path.exists():
        return False

    try:
        config = json.loads(config_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return False

    return config.get("model_type") == "gemma4_ctc"


def load_tokenizer_template(model_dir: str | os.PathLike):
    model_dir = Path(model_dir)
    vocab_path = model_dir / "vocab.json"
    if vocab_path.exists():
        return Wav2Vec2CTCTokenizer.from_pretrained(str(model_dir))

    config = {}
    tokenizer_config_path = model_dir / "tokenizer_config.json"
    if tokenizer_config_path.exists():
        config = json.loads(tokenizer_config_path.read_text(encoding="utf-8"))

    return SimpleNamespace(
        bos_token=config.get("bos_token", "<s>"),
        eos_token=config.get("eos_token", "</s>"),
        unk_token=config.get("unk_token", "<unk>"),
        pad_token=config.get("pad_token", "<pad>"),
        word_delimiter_token=config.get("word_delimiter_token", "|"),
        replace_word_delimiter_char=config.get("replace_word_delimiter_char", " "),
        do_lower_case=config.get("do_lower_case", False),
        pad_token_id=0,
    )


def load_training_dataset(dataset_name: str, dataset_config: str | None, train_split: str) -> DatasetDict:
    dataset_path = Path(dataset_name).expanduser()

    if dataset_path.exists():
        loaded = load_from_disk(str(dataset_path))
        if isinstance(loaded, DatasetDict):
            return loaded
        if isinstance(loaded, Dataset):
            return DatasetDict({train_split: loaded})
        raise TypeError(f"Unsupported dataset type returned from load_from_disk: {type(loaded).__name__}")

    return load_dataset(dataset_name, dataset_config)


def iter_text_column_values(raw_datasets: DatasetDict, text_column: str):
    for split in raw_datasets.values():
        for value in split[text_column]:
            yield value


def build_model(
    *,
    model_dir: str,
    ctc_repo_dir: str,
    tokenizer,
    rebuild_vocab: bool,
    base_model_id: str | None,
):
    from configuration_gemma4_ctc import Gemma4CTCConfig
    from modeling_gemma4_ctc import Gemma4ForCTC

    if not hasattr(tokenizer, "__len__"):
        raise ValueError(
            "No concrete CTC tokenizer is available. "
            "Pass --rebuild_vocab to build one from the dataset, or provide a CTC model repo with vocab.json."
        )

    if is_ctc_model_dir(model_dir):
        config = AutoConfig.from_pretrained(model_dir, trust_remote_code=True)
        if base_model_id is not None:
            config.gemma4_audio_model_id = base_model_id

        if rebuild_vocab:
            config.vocab_size = len(tokenizer)
            config.pad_token_id = tokenizer.pad_token_id
            return AutoModelForCTC.from_pretrained(
                model_dir,
                config=config,
                trust_remote_code=True,
                ignore_mismatched_sizes=True,
            )

        return AutoModelForCTC.from_pretrained(model_dir, config=config, trust_remote_code=True)

    config = Gemma4CTCConfig.from_pretrained(ctc_repo_dir)
    config.vocab_size = len(tokenizer)
    config.pad_token_id = tokenizer.pad_token_id
    config.gemma4_audio_model_id = base_model_id or model_dir
    return Gemma4ForCTC(config)


def main():
    args = parse_args()
    accelerator = Accelerator(gradient_accumulation_steps=args.gradient_accumulation_steps)
    set_seed(args.seed)

    logger.info(accelerator.state)

    raw_datasets = load_training_dataset(args.dataset_name, args.dataset_config, args.train_split)

    ctc_repo_dir = args.model_dir if is_ctc_model_dir(args.model_dir) else str(SCRIPT_DIR)

    feature_extractor = Gemma4AudioFeatureExtractor.from_pretrained(ctc_repo_dir)
    tokenizer_template = load_tokenizer_template(ctc_repo_dir)

    if args.rebuild_vocab:
        vocab_units = collect_ctc_units(
            iter_text_column_values(raw_datasets, args.text_column),
            args.text_column,
        )
        vocab = build_ctc_vocab(vocab_units, tokenizer_template)
        tokenizer_dir = Path(args.output_dir) / "rebuilt_tokenizer"
        tokenizer = save_ctc_tokenizer(vocab, tokenizer_template, tokenizer_dir)

        if accelerator.is_main_process:
            logger.info(
                f"Rebuilt tokenizer with {len(tokenizer)} entries from column '{args.text_column}'"
            )
            logger.info(f"Saved rebuilt tokenizer to {tokenizer_dir}")
    else:
        tokenizer = tokenizer_template

    model = build_model(
        model_dir=args.model_dir,
        ctc_repo_dir=ctc_repo_dir,
        tokenizer=tokenizer,
        rebuild_vocab=args.rebuild_vocab,
        base_model_id=args.base_model_id,
    )

    if accelerator.is_main_process and ctc_repo_dir != args.model_dir:
        logger.info(f"Using local CTC repo assets from {ctc_repo_dir}")
        logger.info(f"Using base Gemma checkpoint from {args.base_model_id or args.model_dir}")

    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()

    if args.unfreeze_norms:
        model.freeze_audio_encoder_except_norm()
    else:
        model.freeze_audio_encoder()

    if accelerator.is_main_process:
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in model.parameters())
        logger.info(f"Trainable parameters: {trainable:,} / {total:,}")

    with accelerator.main_process_first():
        processed = raw_datasets.map(
            prepare_dataset,
            fn_kwargs={
                "feature_extractor": feature_extractor,
                "tokenizer": tokenizer,
                "audio_column": args.audio_column,
                "text_column": args.text_column,
                "audio_channel_strategy": args.audio_channel_strategy,
            },
            remove_columns=raw_datasets[args.train_split].column_names,
            num_proc=4,
        )

    collator = DataCollatorCTCWithPadding(
        feature_extractor=feature_extractor,
        tokenizer=tokenizer,
    )

    train_dataloader = DataLoader(
        processed[args.train_split],
        batch_size=args.per_device_train_batch_size,
        shuffle=True,
        collate_fn=collator,
    )
    eval_dataloader = DataLoader(
        processed[args.eval_split],
        batch_size=args.per_device_eval_batch_size,
        collate_fn=collator,
    )

    optimizer = AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.learning_rate,
    )

    num_update_steps = math.ceil(
        len(train_dataloader) / args.gradient_accumulation_steps
    ) * args.num_train_epochs
    warmup_steps = int(num_update_steps * args.warmup_ratio)

    lr_scheduler = get_scheduler(
        "cosine",
        optimizer=optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=num_update_steps,
    )

    model, optimizer, train_dataloader, eval_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, eval_dataloader, lr_scheduler
    )

    global_step = 0

    for epoch in range(args.num_train_epochs):
        model.train()
        for batch in train_dataloader:
            with accelerator.accumulate(model):
                outputs = model(**batch)
                loss = outputs.loss
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            if accelerator.sync_gradients:
                global_step += 1

                if global_step % args.eval_steps == 0:
                    model.eval()
                    eval_loss = 0.0
                    for eval_batch in eval_dataloader:
                        with torch.no_grad():
                            eval_outputs = model(**eval_batch)
                        eval_loss += accelerator.gather(eval_outputs.loss).mean().item()
                    eval_loss /= len(eval_dataloader)
                    logger.info(f"step {global_step} | eval_loss {eval_loss:.4f}")
                    model.train()

                if global_step % args.save_steps == 0 and accelerator.is_main_process:
                    unwrapped = accelerator.unwrap_model(model)
                    unwrapped.save_pretrained(
                        f"{args.output_dir}/checkpoint-{global_step}",
                        save_function=accelerator.save,
                    )

        logger.info(f"Epoch {epoch + 1} done")

    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        unwrapped = accelerator.unwrap_model(model)
        unwrapped.save_pretrained(args.output_dir, save_function=accelerator.save)
        feature_extractor.save_pretrained(args.output_dir)
        tokenizer.save_pretrained(args.output_dir)
        logger.info(f"Saved to {args.output_dir}")


if __name__ == "__main__":
    main()
