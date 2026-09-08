#!/usr/bin/env python3
"""Search candidate speeches for the Westerberg word-list tokens."""

import argparse
import json
import re
from collections import defaultdict


TOKENS = {
    "BITA": ("iː", "t,d", "all_nine_long_vowels", "iː_only"),
    "BIGA": ("iː", "k,g", "all_nine_long_vowels", "iː_only"),
    "BI": ("iː", "word-final", "all_nine_long_vowels", "iː_only"),
    "PITA": ("iː", "t,d", "all_nine_long_vowels", "iː_only"),
    "PIGA": ("iː", "k,g", "all_nine_long_vowels", "iː_only"),
    "PI": ("iː", "word-final", "all_nine_long_vowels", "iː_only"),
    "BYTA": ("yː", "t,d", "all_nine_long_vowels"),
    "BYKA": ("yː", "k,g", "all_nine_long_vowels"),
    "BY": ("yː", "word-final", "all_nine_long_vowels"),
    "BUDA": ("ʉː", "t,d", "all_nine_long_vowels"),
    "BUGA": ("ʉː", "k,g", "all_nine_long_vowels"),
    "BU": ("ʉː", "word-final", "all_nine_long_vowels"),
    "BETA": ("eː", "t,d", "all_nine_long_vowels"),
    "PEKA": ("eː", "k,g", "all_nine_long_vowels"),
    "BE": ("eː", "word-final", "all_nine_long_vowels"),
    "BÖTA": ("øː", "t,d", "all_nine_long_vowels"),
    "BÖKA": ("øː", "k,g", "all_nine_long_vowels"),
    "HÖ": ("øː", "word-final", "all_nine_long_vowels"),
    "VÄTE": ("ɛː", "t,d", "all_nine_long_vowels"),
    "VÄGA": ("ɛː", "k,g", "all_nine_long_vowels"),
    "BÄ": ("ɛː", "word-final", "all_nine_long_vowels"),
    "BADA": ("ɑː", "t,d", "all_nine_long_vowels"),
    "BAKA": ("ɑː", "k,g", "all_nine_long_vowels"),
    "HA": ("ɑː", "word-final", "all_nine_long_vowels"),
    "BÅDA": ("oː", "t,d", "all_nine_long_vowels"),
    "BÅGE": ("oː", "k,g", "all_nine_long_vowels"),
    "PÅ": ("oː", "word-final", "all_nine_long_vowels"),
    "BOTA": ("uː", "t,d", "all_nine_long_vowels"),
    "BOKA": ("uː", "k,g", "all_nine_long_vowels"),
    "BO": ("uː", "word-final", "all_nine_long_vowels"),
    "VITA": ("iː", "t,d", "iː_only"),
    "FIKA": ("iː", "k,g", "iː_only"),
    "VIKA": ("iː", "k,g", "iː_only"),
    "FILA": ("iː", "l", "iː_only"),
    "VILA": ("iː", "l", "iː_only"),
    "FIRA": ("iː", "r", "iː_only"),
    "VIRA": ("iː", "r", "iː_only"),
    "VISA": ("iː", "s", "iː_only"),
    "VINA": ("iː", "n", "iː_only"),
    "BIBEL": ("iː", "p,b", "iː_only"),
    "PIPA": ("iː", "p,b", "iː_only"),
}

PRECEDING_CONTEXTS = {
    "BIBEL": "b",
    "PIPA": "p",
}


def load_candidates(paths):
    candidates = {}
    for path in paths:
        city = path.split(".")[1]
        with open(path, encoding="utf-8") as handle:
            for line in handle:
                row = json.loads(line)
                candidate = candidates.setdefault(row["riksdagen_id"], {
                    "riksdagen_id": row["riksdagen_id"],
                    "name": row.get("name"),
                    "gender": row.get("gender"),
                    "party": row.get("party"),
                    "district": row.get("district"),
                    "cities": [],
                    "centre_distance_km": row.get("centre_distance_km"),
                    "centre_name": row.get("centre_name"),
                })
                if city not in candidate["cities"]:
                    candidate["cities"].append(city)
    return candidates


def context(text, start, end, radius=90):
    left = max(0, start - radius)
    right = min(len(text), end + radius)
    return text[left:right].replace("\n", " ")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("jsonl")
    parser.add_argument("candidates", nargs="+", help="*by-distance.jsonl files")
    parser.add_argument("output")
    args = parser.parse_args()

    candidates = load_candidates(args.candidates)
    patterns = {
        token: re.compile(r"(?<!\w)" + re.escape(token.lower()) + r"(?!\w)", re.IGNORECASE)
        for token in TOKENS
    }
    matches = defaultdict(lambda: {
        "occurrence_count": 0,
        "speech_ids": set(),
        "years": set(),
        "examples": [],
    })

    with open(args.jsonl, encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            candidate_id = row.get("riksdagen_id")
            if candidate_id not in candidates:
                continue
            candidates[candidate_id]["gender"] = candidates[candidate_id].get("gender") or row.get("gender")
            candidates[candidate_id]["party"] = candidates[candidate_id].get("party") or row.get("party")
            text = row.get("text") or ""
            for token, pattern in patterns.items():
                found = list(pattern.finditer(text))
                if not found:
                    continue
                result = matches[(candidate_id, token)]
                result["occurrence_count"] += len(found)
                if row.get("speech_id"):
                    result["speech_ids"].add(row["speech_id"])
                if row.get("year") is not None:
                    result["years"].add(row["year"])
                for match in found[:3 - len(result["examples"])]:
                    result["examples"].append({
                        "speech_id": row.get("speech_id"),
                        "year": row.get("year"),
                        "context": context(text, match.start(), match.end()),
                    })

    with open(args.output, "w", encoding="utf-8") as handle:
        for (candidate_id, token), result in sorted(matches.items()):
            candidate = candidates[candidate_id]
            vowel, following, *tables = TOKENS[token]
            row = {
                **candidate,
                "token": token,
                "vowel": vowel,
                "following_context": following,
                "preceding_context": PRECEDING_CONTEXTS.get(token),
                "wordlist_tables": tables,
                "occurrence_count": result["occurrence_count"],
                "speech_ids": sorted(result["speech_ids"]),
                "years": sorted(result["years"]),
                "examples": result["examples"],
            }
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
