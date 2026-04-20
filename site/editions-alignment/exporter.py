import json


def to_json(path, sentences_a, sentences_b, alignment, meta):
    rows = [
        {
            "edition_a": {"index": i, "sentence": sentences_a[i] if i is not None else None},
            "edition_b": {"index": j, "sentence": sentences_b[j] if j is not None else None},
        }
        for (i, j) in alignment
    ]
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"meta": meta, "alignment": rows}, f, ensure_ascii=False, indent=2)
