"""Prepare CMU dict + Britfone as T5 seq2seq examples.

Tasks generated:
    g2p en ga: hello          -> həˈloʊ
    g2p en rp: hello          -> həˈləʊ
    p2g en ga: həˈloʊ         -> hello
    p2g en rp: həˈləʊ         -> hello
    transduce en ga to en rp: həˈloʊ  -> həˈləʊ
    transduce en rp to en ga: həˈləʊ  -> həˈloʊ
    identify dialect: həˈloʊ  -> en ga

Transduction and dialect ID require words present in both dictionaries.
For transduction, all GA×RP variant pairs are generated.
Multiple pronunciations: all variants are kept as separate rows (use
--first-only to restrict to the first variant per word).
Words with variant markers like (1)/(2) in Britfone are merged under the
base word.
"""

import argparse
import re
import csv
from pathlib import Path
from convert import arpabet_to_ipa
from rapidfuzz.distance import Levenshtein


def best_pairs(ga_ipas: list[str], rp_ipas: list[str]) -> set[tuple[str, str]]:
    """Pair each GA variant with its closest RP variant and vice versa."""
    pairs = set()
    for ga in ga_ipas:
        best_rp = min(rp_ipas, key=lambda r: Levenshtein.distance(ga, r))
        pairs.add((ga, best_rp))
    for rp in rp_ipas:
        best_ga = min(ga_ipas, key=lambda g: Levenshtein.distance(rp, g))
        pairs.add((best_ga, rp))
    return pairs


def load_cmudict(path: Path) -> dict[str, list[str]]:
    """Return {word: [ipa, ...]} from CMU dict file."""
    entries: dict[str, list[str]] = {}
    with open(path, encoding="latin-1") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith(";;;"):
                continue
            # Strip inline comments (e.g. "# place, danish")
            if "#" in line:
                line = line[:line.index("#")].strip()
            parts = line.split()
            word = parts[0].lower()
            word = re.sub(r"\(\d+\)$", "", word)
            ipa = arpabet_to_ipa(parts[1:])
            entries.setdefault(word, []).append(ipa)
    return entries


def load_britfone(path: Path) -> dict[str, list[str]]:
    """Return {word: [ipa, ...]} from Britfone CSV."""
    entries: dict[str, list[str]] = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            comma = line.index(",")
            raw_word = line[:comma].strip()
            phones_str = line[comma + 1:].strip()
            word = re.sub(r"\(\d+\)$", "", raw_word).lower()
            ipa = phones_str.replace(" ", "")
            entries.setdefault(word, []).append(ipa)
    return entries


def write_tsv(
    ga: dict[str, list[str]],
    rp: dict[str, list[str]],
    out_path: Path,
    first_only: bool = False,
) -> None:
    shared = sorted(set(ga) & set(rp))

    with open(out_path, "w", encoding="utf-8") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow(["input", "target"])

        def dedup(ipas, first_only):
            return [ipas[0]] if first_only else list(dict.fromkeys(ipas))

        # G2P
        for word, ipas in sorted(ga.items()):
            for ipa in dedup(ipas, first_only):
                writer.writerow([f"g2p en ga: {word}", ipa])
        for word, ipas in sorted(rp.items()):
            for ipa in dedup(ipas, first_only):
                writer.writerow([f"g2p en rp: {word}", ipa])

        # P2G
        for word, ipas in sorted(ga.items()):
            for ipa in dedup(ipas, first_only):
                writer.writerow([f"p2g en ga: {ipa}", word])
        for word, ipas in sorted(rp.items()):
            for ipa in dedup(ipas, first_only):
                writer.writerow([f"p2g en rp: {ipa}", word])

        # Cross-dialect transduction and dialect identification (shared vocab only)
        for word in shared:
            ga_ipas = dedup(ga[word], first_only)
            rp_ipas = dedup(rp[word], first_only)

            for ga_ipa, rp_ipa in best_pairs(ga_ipas, rp_ipas):
                # Skip pairs that differ only in secondary stress
                if ga_ipa.replace("ˌ", "") == rp_ipa.replace("ˌ", ""):
                    continue
                writer.writerow([f"transduce en ga to en rp: {ga_ipa}", rp_ipa])
                writer.writerow([f"transduce en rp to en ga: {rp_ipa}", ga_ipa])

            all_ipas = set(ga_ipas) | set(rp_ipas)
            for ipa in all_ipas:
                in_ga = ipa in ga_ipas
                in_rp = ipa in rp_ipas
                if in_ga and in_rp:
                    label = "en ga,rp"
                elif in_ga:
                    label = "en ga"
                else:
                    label = "en rp"
                writer.writerow([f"identify dialect: {ipa}", label])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cmudict", required=True, type=Path)
    parser.add_argument("--britfone", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--first-only", action="store_true",
                        help="Keep only the first pronunciation variant per word")
    args = parser.parse_args()

    print("Loading CMU dict...")
    ga = load_cmudict(args.cmudict)
    print(f"  {len(ga)} words")

    print("Loading Britfone...")
    rp = load_britfone(args.britfone)
    print(f"  {len(rp)} words")

    shared = set(ga) & set(rp)
    print(f"  {len(shared)} words in common (used for transduction + dialect ID)")

    print(f"Writing {args.out}...")
    write_tsv(ga, rp, args.out, first_only=args.first_only)
    print("Done.")


if __name__ == "__main__":
    main()
