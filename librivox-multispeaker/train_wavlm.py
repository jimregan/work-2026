"""End-to-end training of WavLM + MultiAxisProjection via the
MultiAxisProjectionTrainer.

Loads a pre-processed HuggingFace dataset from disk (produced by the
notebook) and fine-tunes WavLM jointly with the projection heads.

Usage::

    python train_wavlm.py \
        --dataset_dir ./merged_dataset \
        --output_dir  ./output/wavlm-multiaxis \
        --model_id    microsoft/wavlm-base-plus \
        --encoder_dim 768

To run in tmux on a server::

    tmux new -s train
    python train_wavlm.py --dataset_dir /path/to/merged_dataset ...
    # Ctrl-b d  to detach
"""

from __future__ import annotations

import argparse

import torch
from datasets import load_from_disk
from sentence_transformers.models import Pooling
from sentence_transformers.training_args import SentenceTransformerTrainingArguments

from spoken_sentence_transformers import (
    MultiAxisInfoNCELoss,
    MultiAxisProjection,
    MultiAxisProjectionTrainer,
    MultiAxisNoDuplicatesBatchSampler,
    MultiAxisSentenceTransformer,
)
from spoken_sentence_transformers.encoders import HFAcousticEncoder


def parse_axes(specs: list[str]) -> dict[str, int]:
    """Parse 'name:dim' axis specs, e.g. 'semantic:256 speaker_id:128'."""
    result = {}
    for s in specs:
        name, dim = s.split(":")
        result[name.strip()] = int(dim.strip())
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fine-tune WavLM + MultiAxisProjection heads end-to-end."
    )
    parser.add_argument(
        "--dataset_dir",
        required=True,
        help="Path to merged dataset saved with save_to_disk().",
    )
    parser.add_argument(
        "--output_dir",
        default="./output/wavlm-multiaxis",
        help="Directory for checkpoints and final model.",
    )
    parser.add_argument(
        "--model_id",
        default="microsoft/wavlm-base-plus",
        help="HuggingFace model ID for the acoustic encoder.",
    )
    parser.add_argument(
        "--encoder_dim",
        type=int,
        default=768,
        help="Hidden size of the acoustic encoder (768 for wavlm-base-plus, 1024 for wavlm-large).",
    )
    parser.add_argument(
        "--axes",
        nargs="+",
        default=["semantic:256", "speaker_id:128", "dialect:64", "gender:32"],
        help="Projection axis specs as name:dim pairs.",
    )
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--warmup_ratio", type=float, default=0.1)
    parser.add_argument("--logging_steps", type=int, default=100)
    parser.add_argument(
        "--fp16",
        action="store_true",
        default=torch.cuda.is_available(),
        help="Use fp16 mixed precision (default: on if CUDA available).",
    )
    args = parser.parse_args()

    axes = parse_axes(args.axes)

    print(f"Loading dataset from {args.dataset_dir} ...")
    dataset = load_from_disk(args.dataset_dir)

    print(f"Building model: {args.model_id}, axes={axes}")
    encoder = HFAcousticEncoder(args.model_id)
    pooling = Pooling(args.encoder_dim, pooling_mode="mean")
    proj    = MultiAxisProjection(in_features=args.encoder_dim, axes=axes)
    model   = MultiAxisSentenceTransformer(modules=[encoder, pooling, proj])

    loss = MultiAxisInfoNCELoss(model)

    training_args = SentenceTransformerTrainingArguments(
        output_dir=args.output_dir,
        batch_sampler=MultiAxisNoDuplicatesBatchSampler,
        per_device_train_batch_size=args.batch_size,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        warmup_ratio=args.warmup_ratio,
        fp16=args.fp16,
        save_strategy="epoch",
        logging_steps=args.logging_steps,
    )

    trainer = MultiAxisProjectionTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        loss=loss,
    )

    print("Training ...")
    trainer.train()
    model.save_pretrained(args.output_dir)
    print(f"Model saved to {args.output_dir}")


if __name__ == "__main__":
    main()
