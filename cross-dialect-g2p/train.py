"""Fine-tune T5 for dialect-conditioned G2P.

Input:  "g2p ga: hello"
Output: "həloʊ"

Usage:
    python train.py \\
        --data data/g2p.tsv \\
        --model google/byt5-small \\
        --out models/g2p-byt5
"""

import argparse
import csv
from pathlib import Path

import torch
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    DataCollatorForSeq2Seq,
)


def load_dataset_from_tsv(path: Path) -> Dataset:
    inputs, targets = [], []
    with open(path, encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            inputs.append(row["input"])
            targets.append(row["target"])
    return Dataset.from_dict({"input": inputs, "target": targets})


def tokenize(batch, tokenizer, max_input=64, max_target=64):
    model_inputs = tokenizer(
        batch["input"],
        max_length=max_input,
        truncation=True,
        padding=False,
    )
    labels = tokenizer(
        text_target=batch["target"],
        max_length=max_target,
        truncation=True,
        padding=False,
    )
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--model", default="google/byt5-small")
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--val-split", type=float, default=0.02)
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model)

    dataset = load_dataset_from_tsv(args.data)
    split = dataset.train_test_split(test_size=args.val_split, seed=42)
    train_ds = split["train"]
    eval_ds = split["test"]

    fn = lambda batch: tokenize(batch, tokenizer)
    train_ds = train_ds.map(fn, batched=True, remove_columns=["input", "target"])
    eval_ds = eval_ds.map(fn, batched=True, remove_columns=["input", "target"])

    collator = DataCollatorForSeq2Seq(tokenizer, model=model, padding=True)

    training_args = Seq2SeqTrainingArguments(
        output_dir=str(args.out),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=args.lr,
        warmup_ratio=0.05,
        predict_with_generate=True,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        fp16=torch.cuda.is_available(),
        logging_steps=200,
        report_to="none",
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        tokenizer=tokenizer,
        data_collator=collator,
    )

    trainer.train()
    trainer.save_model(str(args.out))
    tokenizer.save_pretrained(str(args.out))
    print(f"Model saved to {args.out}")


if __name__ == "__main__":
    main()
