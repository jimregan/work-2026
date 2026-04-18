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
import math
import os
import sys

import torch
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from datasets import load_dataset
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import AutoConfig, AutoModelForCTC, Wav2Vec2CTCTokenizer, get_scheduler
from transformers.models.gemma4.feature_extraction_gemma4 import Gemma4AudioFeatureExtractor

sys.path.insert(0, os.path.dirname(__file__))
from collator import DataCollatorCTCWithPadding


logger = get_logger(__name__)
DEFAULT_MODEL_DIR = os.path.dirname(os.path.abspath(__file__))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", default=DEFAULT_MODEL_DIR,
                        help="Repo dir containing config.json, vocab.json, etc.")
    parser.add_argument("--dataset_name", required=True)
    parser.add_argument("--dataset_config", default=None)
    parser.add_argument("--train_split", default="train")
    parser.add_argument("--eval_split", default="validation")
    parser.add_argument("--audio_column", default="audio")
    parser.add_argument("--text_column", default="text")
    parser.add_argument("--output_dir", required=True)
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


def prepare_dataset(batch, feature_extractor, tokenizer, audio_column, text_column):
    audio = batch[audio_column]
    batch["input_features"] = feature_extractor(
        audio["array"], sampling_rate=audio["sampling_rate"]
    ).input_features[0]
    batch["labels"] = tokenizer(batch[text_column]).input_ids
    return batch


def main():
    args = parse_args()
    accelerator = Accelerator(gradient_accumulation_steps=args.gradient_accumulation_steps)
    set_seed(args.seed)

    logger.info(accelerator.state)

    feature_extractor = Gemma4AudioFeatureExtractor.from_pretrained(args.model_dir)
    tokenizer = Wav2Vec2CTCTokenizer.from_pretrained(args.model_dir)

    model = AutoModelForCTC.from_pretrained(args.model_dir, trust_remote_code=True)

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

    raw_datasets = load_dataset(args.dataset_name, args.dataset_config)

    with accelerator.main_process_first():
        processed = raw_datasets.map(
            prepare_dataset,
            fn_kwargs={
                "feature_extractor": feature_extractor,
                "tokenizer": tokenizer,
                "audio_column": args.audio_column,
                "text_column": args.text_column,
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
