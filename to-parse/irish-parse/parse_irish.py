#!/usr/bin/env python3
"""Parse pre-standard Irish with Stanza, cross-checked against UDPipe.

Pipeline per input sentence:
  1. Standardize the pre-standard text via the intergaelic API (-> text_standard).
  2. Parse the standardized text with Stanza (primary).
  3. Feed Stanza's tokens to UDPipe (shared tokenization) for a second opinion.
  4. Rebuild the parse over the original tokens: FORM is the pre-standard
     surface exactly as written; standardization splits become multiword
     tokens; deleted originals get Skip=Standard placeholder rows.
  5. Emit CoNLL-U from each parser plus a Markdown report of the differences a
     human needs to resolve.

Input: one sentence per non-blank line (plain text), or a CoNLL-U file whose
``# text`` comments supply the sentences (--from-conllu).

Outputs (given --out PREFIX):
  PREFIX.conllu          primary parse (Stanza), original forms, text_standard
  PREFIX.udpipe.conllu   UDPipe parse of the same tokens
  PREFIX.diff.md         differences to resolve, in Markdown
"""
from __future__ import annotations

import argparse
import os
import sys
from typing import List, Optional, Tuple

from irish_parse import build, compare, conllu, modernize, prestandard


def read_sentences(path: str, from_conllu: bool) -> List[str]:
    with open(path, encoding="utf-8") as fh:
        text = fh.read()
    if from_conllu:
        out = []
        for sent in conllu.parse(text):
            t = sent.meta_get("text")
            if t:
                out.append(t)
        return out
    return [ln.strip() for ln in text.splitlines() if ln.strip()]


def process(
    sentences: List[str], offline: bool, rules=None
) -> Tuple[List[conllu.Sentence], List[conllu.Sentence], List[compare.SentenceDiff]]:
    from irish_parse import stanza_parser, udpipe_parser

    primary: List[conllu.Sentence] = []
    udpipe_out: List[conllu.Sentence] = []
    diffs: List[compare.SentenceDiff] = []

    for i, original in enumerate(sentences, start=1):
        sid = str(i)
        standard, pairs = modernize.standardize(original, offline=offline)
        if pairs is not None and rules:
            pairs = prestandard.apply(pairs, rules)
            standard = " ".join(s for _, s in pairs if s.strip())
        if pairs is None:
            print(
                f"[sent {sid}] standardization unavailable; parsing text as-is",
                file=sys.stderr,
            )

        # parse the standardized text; both parsers share this tokenization
        st = stanza_parser.parse_sentence(standard)
        forms = [t.form for t in st.tokens]
        ud = udpipe_parser.parse_tokens(forms)

        display_standard = modernize.detokenize(standard)
        st.meta_set("sent_id", sid)
        st.meta_set("text", original)
        st.meta_set("text_standard", display_standard)

        if pairs is not None:
            alignment = modernize.align(pairs, forms)
            primary_sent = build.build_primary(st, pairs, alignment)
        else:
            # no standardization available: keep the text as parsed, flag it
            alignment = None
            primary_sent = st
            for tok in primary_sent.tokens:
                tok.add_misc("Align", "NoStandard")

        ud.metadata = [("sent_id", sid), ("text_standard", display_standard)]

        primary.append(primary_sent)
        udpipe_out.append(ud)
        # compare over the shared standard tokenization (st, not the rebuilt one)
        diffs.append(compare.compare(sid, original, st, ud, alignment))

    return primary, udpipe_out, diffs


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("input", help="input text file (one sentence per line)")
    ap.add_argument(
        "--out",
        required=True,
        help="output prefix; writes PREFIX.conllu, PREFIX.udpipe.conllu, PREFIX.diff.md",
    )
    ap.add_argument(
        "--from-conllu",
        action="store_true",
        help="read sentences from the '# text' comments of a CoNLL-U file",
    )
    ap.add_argument(
        "--offline",
        action="store_true",
        help="never call the intergaelic API; use cache only",
    )
    ap.add_argument(
        "--pre-standard",
        metavar="FILE",
        help="TSV of text-specific standardisation overrides "
        "(original<TAB>standard, multi-word allowed), applied on top of "
        "the intergaelic output",
    )
    args = ap.parse_args(argv)

    sentences = read_sentences(args.input, args.from_conllu)
    if not sentences:
        print("No input sentences found.", file=sys.stderr)
        return 1

    rules = prestandard.load_rules(args.pre_standard) if args.pre_standard else None

    out_dir = os.path.dirname(args.out)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    primary, udpipe_out, diffs = process(sentences, offline=args.offline, rules=rules)

    with open(f"{args.out}.conllu", "w", encoding="utf-8") as fh:
        fh.write(conllu.dump(primary))
    with open(f"{args.out}.udpipe.conllu", "w", encoding="utf-8") as fh:
        fh.write(conllu.dump(udpipe_out))
    with open(f"{args.out}.diff.md", "w", encoding="utf-8") as fh:
        fh.write(compare.render_markdown(diffs))

    affected = sum(d.n_tokens_affected for d in diffs)
    flagged = sum(1 for d in diffs if d.diffs or d.misaligned)
    print(
        f"Parsed {len(sentences)} sentence(s). "
        f"{flagged} flagged, {affected} token(s) with disagreements.\n"
        f"  {args.out}.conllu\n  {args.out}.udpipe.conllu\n  {args.out}.diff.md"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
