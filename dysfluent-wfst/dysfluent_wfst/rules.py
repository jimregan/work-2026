"""YAML-based phonetic rule loading and compilation to pynini cdrewrite.

Rules use the MFA (Montreal Forced Aligner) YAML format::

    rules:
      - segment: d
        replacement: ''
        preceding_context: 'n'
        following_context: ''
      - segment: ə ɫ
        replacement: ɫ̩
        preceding_context: ''
        following_context: '[^ʊɔɝaɛeoæɐɪəɚɑʉɒi].*'

Fields:
  - ``segment``: phoneme sequence to match (space-separated).
      May contain regex-like character classes (``[abc]``, ``[^abc]``)
      and ``?`` for optionality.
  - ``replacement``: replacement phoneme sequence (space-separated).
      Empty string means deletion.
  - ``preceding_context``: left context (regex-like, or empty).
  - ``following_context``: right context (regex-like, ``$`` for
      word/utterance boundary, or empty).

Each rule is compiled to ``pynini.cdrewrite(..., mode="opt")`` so both
the citation form and the variant survive in the output lattice.
Rules are composed into a single cascade.
"""

from __future__ import annotations

import re
from typing import Optional

import pynini


def _symbol_acceptor(sym: str, syms: pynini.SymbolTable) -> pynini.Fst:
    """Acceptor for a single phoneme symbol in the SymbolTable label space."""
    return pynini.accep(sym, token_type=syms)


def build_sigma_star(syms: pynini.SymbolTable) -> pynini.Fst:
    """Build sigma_star: the closure over all symbols in the table.

    Arcs are labelled with the SymbolTable's integer ids (via
    ``token_type=syms``) so the rules share a label space with the
    lexicon and reference FSTs. Using byte/utf8 escaping here instead
    would silently produce empty compositions.
    """
    symbol_fsts = []
    for idx in range(syms.num_symbols()):
        sym = syms.find(idx)
        if sym == "<eps>" or sym == "":
            continue
        symbol_fsts.append(_symbol_acceptor(sym, syms))

    sigma = pynini.union(*symbol_fsts)
    return sigma.closure().optimize()


def _all_symbols(syms: pynini.SymbolTable) -> set[str]:
    """Return the set of all non-epsilon symbols in the table."""
    result = set()
    for idx in range(syms.num_symbols()):
        sym = syms.find(idx)
        if sym and sym != "<eps>":
            result.add(sym)
    return result


# Regex to tokenise an MFA-style pattern string into chunks:
#   [^...] or [...] or X? or .* or single-char-phoneme
_PATTERN_TOKEN_RE = re.compile(
    r"""
    \[\^[^\]]+\]  # negated character class  [^abc]
    | \[[^\]]+\]  # character class          [abc]
    | \.\*        # wildcard                 .*
    | \$          # boundary                 $
    | .           # single character (phoneme)
    """,
    re.VERBOSE,
)


def _segment_symbols(text: str, all_syms: set[str]) -> list[str]:
    """Greedily segment a codepoint run into known SymbolTable symbols.

    Longest-match first, so multi-codepoint symbols (e.g. ``ʉː``) are
    preferred over their constituent codepoints. Codepoints that match
    no symbol are skipped.
    """
    by_len = sorted((s for s in all_syms if s), key=len, reverse=True)
    result: list[str] = []
    i = 0
    while i < len(text):
        for sym in by_len:
            if text.startswith(sym, i):
                result.append(sym)
                i += len(sym)
                break
        else:
            i += 1
    return result


def _parse_pattern(
    pattern: str,
    syms: pynini.SymbolTable,
    sigma_star: pynini.Fst,
) -> pynini.Fst:
    """Parse an MFA-style regex-like pattern into a pynini acceptor.

    All sub-expressions are built in the SymbolTable label space
    (``token_type=syms``) so they compose with the lexicon and reference
    FSTs. Literal runs are segmented into known symbols (longest-match).

    Supported syntax:

    - Literal phoneme symbols are concatenated.
    - ``[abc]`` — union of the listed symbols.
    - ``[^abc]`` — union of all symbols *except* the listed ones.
    - ``?`` after any element — makes that element optional.
    - ``.*`` — sigma_star (match anything).
    - ``$`` — word/utterance boundary (treated as epsilon, since our
      FSTs operate on isolated utterance chunks).
    """
    all_syms = _all_symbols(syms)
    eps = pynini.accep("", token_type=syms)

    tokens = _PATTERN_TOKEN_RE.findall(pattern)
    if not tokens:
        return eps

    atoms: list[pynini.Fst] = []
    lit = ""

    def flush_lit() -> None:
        nonlocal lit
        for sym in _segment_symbols(lit, all_syms):
            atoms.append(_symbol_acceptor(sym, syms))
        lit = ""

    for tok in tokens:
        if tok == ".*":
            flush_lit()
            atoms.append(sigma_star.copy())
        elif tok == "$":
            flush_lit()  # boundary → epsilon
        elif tok.startswith("[^") and tok.endswith("]"):
            flush_lit()
            excluded = set(_segment_symbols(tok[2:-1], all_syms))
            included = sorted(all_syms - excluded)
            if not included:
                raise ValueError(f"Negated class {tok} excludes all symbols")
            atoms.append(
                pynini.union(*[_symbol_acceptor(s, syms) for s in included])
            )
        elif tok.startswith("[") and tok.endswith("]"):
            flush_lit()
            members = _segment_symbols(tok[1:-1], all_syms)
            atoms.append(
                pynini.union(*[_symbol_acceptor(s, syms) for s in members])
            )
        elif tok == "?":
            flush_lit()
            if atoms:
                atoms.append(pynini.union(atoms.pop(), eps))
        else:
            lit += tok
    flush_lit()

    if not atoms:
        return eps

    result = atoms[0]
    for a in atoms[1:]:
        result = pynini.concat(result, a)
    return result.optimize()


def _compile_element(
    field_value: str,
    syms: pynini.SymbolTable,
    sigma_star: pynini.Fst,
) -> pynini.Fst:
    """Compile a segment/replacement/context field value to a pynini FST.

    If the value contains spaces, each space-separated token is treated
    as a phoneme symbol and they are concatenated. Regex-like patterns
    (``[...]``, ``?``, ``.*``, ``$``) within tokens are parsed by
    ``_parse_pattern``.

    If the value contains no spaces but has regex metacharacters, the
    whole string is parsed as a pattern.

    All acceptors use ``token_type=syms`` so they share a label space
    with the lexicon FST. An empty string produces epsilon.
    """
    if not field_value:
        return pynini.accep("", token_type=syms)

    # Check if it looks like it uses regex features
    has_regex = bool(re.search(r"[\[\]?*$^]", field_value))

    if " " in field_value:
        # Space-separated phoneme tokens
        tokens = field_value.split()
        parts = []
        for tok in tokens:
            if re.search(r"[\[\]?*$^]", tok):
                parts.append(_parse_pattern(tok, syms, sigma_star))
            else:
                parts.append(_symbol_acceptor(tok, syms))
        result = parts[0]
        for p in parts[1:]:
            result = pynini.concat(result, p)
        return result.optimize()

    if has_regex:
        return _parse_pattern(field_value, syms, sigma_star)

    # Plain single phoneme symbol
    return _symbol_acceptor(field_value, syms)


def _compile_one_rule(
    rule: dict,
    syms: pynini.SymbolTable,
    sigma_star: pynini.Fst,
) -> pynini.Fst:
    """Compile a single MFA rule dict to an optional cdrewrite transducer.

    Args:
        rule: Dict with keys ``segment``, ``replacement``,
            ``preceding_context``, ``following_context``.
        syms: Symbol table for the phoneme inventory.
        sigma_star: Closure over all symbols.

    Returns:
        A pynini cdrewrite transducer with ``mode="opt"``.
    """
    segment_str = str(rule.get("segment", ""))
    replacement_str = str(rule.get("replacement", ""))
    preceding_str = str(rule.get("preceding_context", ""))
    following_str = str(rule.get("following_context", ""))

    segment_fst = _compile_element(segment_str, syms, sigma_star)
    replacement_fst = _compile_element(replacement_str, syms, sigma_star)

    tau = pynini.cross(segment_fst, replacement_fst)

    lam = _compile_element(preceding_str, syms, sigma_star)
    rho = _compile_element(following_str, syms, sigma_star)

    return pynini.cdrewrite(
        tau, lam, rho, sigma_star, direction="ltr", mode="opt"
    )


def load_rules(path: str) -> list[dict]:
    """Load rules from an MFA-format YAML file.

    Args:
        path: Path to YAML file with a top-level ``rules`` key.

    Returns:
        List of rule dicts, each with keys ``segment``, ``replacement``,
        ``preceding_context``, ``following_context``.
    """
    import yaml

    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f)

    if not isinstance(data, dict) or "rules" not in data:
        raise ValueError(
            f"Rules YAML must have a top-level 'rules' key, got: "
            f"{list(data.keys()) if isinstance(data, dict) else type(data)}"
        )

    rules = data["rules"]
    if not isinstance(rules, list):
        raise ValueError(f"'rules' must be a list, got {type(rules)}")

    for i, rule in enumerate(rules):
        if "segment" not in rule:
            raise ValueError(f"Rule {i} is missing required key 'segment'")
        if "replacement" not in rule:
            raise ValueError(f"Rule {i} is missing required key 'replacement'")

    return rules


def compile_rules(
    rules_path: Optional[str],
    syms: pynini.SymbolTable,
) -> Optional[pynini.Fst]:
    """Compile phonetic rules from an MFA-format YAML file.

    Each rule is compiled to an optional cdrewrite transducer
    (``mode="opt"``), and all rules are composed into a single cascade.

    Args:
        rules_path: Path to YAML rules file.
            If None, returns None (no rules applied).
        syms: Symbol table for building sigma_star and compiling
            rule patterns.

    Returns:
        A composed rules transducer, or None if no rules_path given.
    """
    if rules_path is None:
        return None

    sigma_star = build_sigma_star(syms)
    rules = load_rules(rules_path)

    if not rules:
        return None

    rule_fsts = [_compile_one_rule(r, syms, sigma_star) for r in rules]

    composed = rule_fsts[0]
    for fst in rule_fsts[1:]:
        composed = pynini.compose(composed, fst)

    return composed.optimize()
