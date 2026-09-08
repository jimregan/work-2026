#!/usr/bin/env python3
"""Find open-vocabulary transcript candidates for vowel-context feasibility."""

import argparse
import json
import re
from collections import defaultdict


VOWEL_GRAPHEMES = {
    "iː": "i",
    "yː": "y",
    "ʉː": "u",
    "eː": "e",
    "øː": "ö",
    "ɛː": "ä",
    "ɑː": "a",
    "oː": "å",
    "uː": "o",
}
WORD_RE = re.compile(r"[^\W\d_]+", re.UNICODE)


def load_candidates(paths):
    candidates = {}
    for path in paths:
        city = path.split(".")[1]
        with open(path, encoding="utf-8") as handle:
            for line in handle:
                row = json.loads(line)
                candidate = candidates.setdefault(row["riksdagen_id"], dict(row))
                candidate["cities"] = sorted(set(candidate.get("cities", [])) | {city})
    return candidates


def add_hit(hits, candidate_id, vowel, context, word, speech_id, year):
    key = (candidate_id, vowel, context)
    result = hits[key]
    result["occurrence_count"] += 1
    result["word_types"].add(word)
    if speech_id:
        result["speech_ids"].add(speech_id)
    if year is not None:
        result["years"].add(year)
    if len(result["examples"]) < 5:
        result["examples"].append({"word": word, "speech_id": speech_id, "year": year})


def scan_word(word, candidate_id, speech_id, year, hits):
    if word == "på":
        word_for_final_context = ""
    else:
        word_for_final_context = word
    for vowel, grapheme in VOWEL_GRAPHEMES.items():
        for match in re.finditer(re.escape(grapheme), word):
            start, end = match.span()
            before = word[start - 1] if start else None
            after = word[end] if end < len(word) else None
            after_after = word[end + 1] if end + 1 < len(word) else None
            if after is not None and after in "td" and (after_after is None or after_after in "aeiouyåäö"):
                add_hit(hits, candidate_id, vowel, "t,d", word, speech_id, year)
            if after is not None and after in "kg" and (after_after is None or after_after in "aeiouyåäö"):
                add_hit(hits, candidate_id, vowel, "k,g", word, speech_id, year)
            if end == len(word_for_final_context):
                add_hit(hits, candidate_id, vowel, "word-final", word, speech_id, year)
            if vowel == "iː":
                if before is not None and before in "bpfv" and after is not None and after in "pb" and (after_after is None or after_after in "aeiouyåäö"):
                    add_hit(hits, candidate_id, vowel, "preceding-bpfv-following-p,b", word, speech_id, year)
                if before is not None and before in "vf" and after is not None and after in "lr":
                    add_hit(hits, candidate_id, vowel, f"preceding-{before}-following-{after}", word, speech_id, year)
                if before == "v" and after is not None and after in "sn":
                    add_hit(hits, candidate_id, vowel, f"preceding-v-following-{after}", word, speech_id, year)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("jsonl")
    parser.add_argument("candidates", nargs="+")
    parser.add_argument("output")
    args = parser.parse_args()

    candidates = load_candidates(args.candidates)
    hits = defaultdict(lambda: {
        "occurrence_count": 0,
        "word_types": set(),
        "speech_ids": set(),
        "years": set(),
        "examples": [],
    })
    seen_metadata = set()
    with open(args.jsonl, encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            candidate_id = row.get("riksdagen_id")
            if candidate_id not in candidates:
                continue
            if candidate_id not in seen_metadata:
                candidates[candidate_id].update({
                    "gender": row.get("gender"),
                    "party": row.get("party"),
                })
                seen_metadata.add(candidate_id)
            for word in WORD_RE.findall((row.get("text") or "").lower()):
                scan_word(word, candidate_id, row.get("speech_id"), row.get("year"), hits)

    with open(args.output, "w", encoding="utf-8") as handle:
        for (candidate_id, vowel, context), result in sorted(hits.items()):
            candidate = candidates[candidate_id]
            row = {
                "riksdagen_id": candidate_id,
                "name": candidate.get("name"),
                "gender": candidate.get("gender"),
                "party": candidate.get("party"),
                "district": candidate.get("district"),
                "cities": candidate.get("cities"),
                "centre_distance_km": candidate.get("centre_distance_km"),
                "centre_name": candidate.get("centre_name"),
                "vowel": vowel,
                "context": context,
                "occurrence_count": result["occurrence_count"],
                "word_type_count": len(result["word_types"]),
                "word_types": sorted(result["word_types"]),
                "speech_ids": sorted(result["speech_ids"]),
                "years": sorted(result["years"]),
                "examples": result["examples"],
            }
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
