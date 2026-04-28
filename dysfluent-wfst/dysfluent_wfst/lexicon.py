"""TSV lexicon loading and lexicon FST construction."""

from __future__ import annotations

from typing import Optional

import pynini


def load_lexicon(path: str) -> list[tuple[str, str]]:
    """Load a TSV lexicon file.

    Format: ``word<TAB>p1 p2 p3`` (one entry per line).
    Lines starting with ``#`` are comments. Blank lines are skipped.
    Multiple lines for the same word give multiple pronunciations.

    Returns a list of (word, pronunciation_string) tuples where the
    pronunciation string is space-separated phonemes.
    """
    entries: list[tuple[str, str]] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split("\t", maxsplit=1)
            if len(parts) != 2:
                continue
            word, pron = parts
            entries.append((word.strip(), pron.strip()))
    return entries


def build_lexicon_fst(
    entries: list[tuple[str, str]],
    input_token_type: str = "utf8",
    output_syms: Optional[pynini.SymbolTable] = None,
) -> pynini.Fst:
    """Build a lexicon transducer from (word, pronunciation) entries.

    Uses ``pynini.string_map`` to create a transducer mapping
    orthographic forms to phoneme sequences.

    Args:
        entries: List of (word, pronunciation_string) tuples.
        input_token_type: Token type for the input (word) side.
        output_syms: SymbolTable for the output (phoneme) side.
            If None, output is treated as utf8.
    """
    output_type = output_syms if output_syms is not None else "utf8"
    fst = pynini.string_map(
        entries,
        input_token_type=input_token_type,
        output_token_type=output_type,
    )
    return fst.optimize()


def lookup_word(word: str, lexicon_fst: pynini.Fst) -> pynini.Fst:
    """Look up a single word in the lexicon FST.

    Returns a transducer/acceptor containing all pronunciations for
    the word. Raises ``pynini.FstArgError`` if the word is not found.
    """
    word_fst = pynini.escape(word)
    return pynini.compose(word_fst, lexicon_fst)


def build_utterance_fst(
    words: list[str],
    lexicon_fst: pynini.Fst,
    rules_fst: Optional[pynini.Fst] = None,
) -> pynini.Fst:
    """Build an utterance-level pronunciation FST by concatenating word lookups.

    For each word, looks it up in the lexicon, optionally composes with
    the phonetic rules transducer, projects to the output side, and
    concatenates into a single utterance FST.

    Args:
        words: Sequence of words in the utterance.
        lexicon_fst: Compiled lexicon transducer.
        rules_fst: Optional cdrewrite rules transducer. If provided,
            each word's pronunciation is composed with it to expand
            the pronunciation lattice with phonetic variants.

    Returns:
        An FST over the phoneme symbol table containing all
        pronunciation paths for the utterance.
    """
    if not words:
        raise ValueError("words must be non-empty")

    word_fsts = []
    for word in words:
        word_pron = lookup_word(word, lexicon_fst)
        if rules_fst is not None:
            word_pron = pynini.compose(word_pron, rules_fst)
        word_fsts.append(word_pron.project("output"))

    utt_fst = word_fsts[0]
    for wf in word_fsts[1:]:
        utt_fst = pynini.concat(utt_fst, wf)
    return utt_fst.optimize()
