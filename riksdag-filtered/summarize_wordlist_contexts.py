#!/usr/bin/env python3
"""Summarize phonetic-context coverage from word-list match rows."""

import argparse
import json
from collections import defaultdict


VOWELS = ["iː", "yː", "ʉː", "eː", "øː", "ɛː", "ɑː", "oː", "uː"]
COMMON_CONTEXTS = ["t,d", "k,g", "word-final"]
I_ONLY_TOKENS = [
    "BI", "PI", "BIBEL", "PIPA", "VITA", "BITA", "PITA", "VIKA",
    "FIKA", "BIGA", "PIGA", "VILA", "FILA", "VIRA", "FIRA", "VISA", "VINA",
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("matches")
    parser.add_argument("output")
    args = parser.parse_args()

    candidates = {}
    common = defaultdict(set)
    i_only = defaultdict(set)
    counts = defaultdict(int)
    for line in open(args.matches, encoding="utf-8"):
        row = json.loads(line)
        candidate_id = row["riksdagen_id"]
        candidates[candidate_id] = {
            key: row.get(key)
            for key in (
                "riksdagen_id", "name", "gender", "party", "district", "cities",
                "centre_distance_km", "centre_name",
            )
        }
        if row["token"] == "PÅ":
            continue
        counts[candidate_id] += row["occurrence_count"]
        if row["following_context"] in COMMON_CONTEXTS:
            common[candidate_id].add((row["vowel"], row["following_context"]))
        if "iː_only" in row["wordlist_tables"]:
            i_only[candidate_id].add(row["token"])

    all_common = [f"{vowel}:{context}" for vowel in VOWELS for context in COMMON_CONTEXTS]
    with open(args.output, "w", encoding="utf-8") as handle:
        for candidate_id in sorted(candidates):
            common_contexts = sorted(f"{vowel}:{context}" for vowel, context in common[candidate_id])
            missing_common = [cell for cell in all_common if cell not in common_contexts]
            i_contexts = sorted(i_only[candidate_id], key=I_ONLY_TOKENS.index)
            row = {
                **candidates[candidate_id],
                "total_occurrences_excluding_på": counts[candidate_id],
                "common_context_count": len(common_contexts),
                "common_contexts": common_contexts,
                "missing_common_contexts": missing_common,
                "essential_i_common_context_count": sum(cell.startswith("iː:") for cell in common_contexts),
                "essential_i_common_contexts": [cell for cell in common_contexts if cell.startswith("iː:")],
                "i_only_token_context_count": len(i_contexts),
                "i_only_token_contexts": i_contexts,
            }
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
