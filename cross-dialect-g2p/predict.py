"""Run inference / few-shot eval on a trained G2P model.

Usage:
    # Single word
    python predict.py --model models/g2p-byt5 --dialect ga --word hello

    # Evaluate from TSV (same format as training data)
    python predict.py --model models/g2p-byt5 --eval data/dialect_X.tsv
"""

import argparse
from pathlib import Path
import csv

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch


def predict(model, tokenizer, inputs: list[str], batch_size=64) -> list[str]:
    results = []
    for i in range(0, len(inputs), batch_size):
        batch = inputs[i:i + batch_size]
        enc = tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=64)
        enc = {k: v.to(model.device) for k, v in enc.items()}
        with torch.no_grad():
            out = model.generate(**enc, max_new_tokens=64)
        results.extend(tokenizer.batch_decode(out, skip_special_tokens=True))
    return results


def per_char_error(pred: str, ref: str) -> float:
    """Character Error Rate (simple edit distance / len(ref))."""
    from rapidfuzz.distance import Levenshtein
    return Levenshtein.distance(pred, ref) / max(len(ref), 1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--dialect", help="Dialect prefix for single-word mode (ga/rp/...)")
    parser.add_argument("--word", help="Single word to transcribe")
    parser.add_argument("--eval", type=Path, help="TSV file to evaluate (input/target columns)")
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model)
    model.eval()
    if torch.cuda.is_available():
        model = model.cuda()

    if args.word:
        inp = f"g2p en {args.dialect}: {args.word}"
        result = predict(model, tokenizer, [inp])[0]
        print(result)

    if args.eval:
        inputs, refs = [], []
        with open(args.eval, encoding="utf-8") as f:
            reader = csv.DictReader(f, delimiter="\t")
            for row in reader:
                inputs.append(row["input"])
                refs.append(row["target"])

        preds = predict(model, tokenizer, inputs)

        exact = sum(p == r for p, r in zip(preds, refs))
        print(f"Word accuracy: {exact}/{len(refs)} = {exact/len(refs):.3f}")

        try:
            cer = sum(per_char_error(p, r) for p, r in zip(preds, refs)) / len(refs)
            print(f"Mean CER:      {cer:.4f}")
        except ImportError:
            print("(install rapidfuzz for CER)")

        # Show first 20 errors
        errors = [(i, r, p) for i, (p, r) in enumerate(zip(preds, refs)) if p != r]
        for i, ref, pred in errors[:20]:
            print(f"  [{inputs[i]}]  ref={ref}  pred={pred}")


if __name__ == "__main__":
    main()
