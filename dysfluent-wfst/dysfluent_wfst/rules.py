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


def build_sigma_star(syms: pynini.SymbolTable) -> pynini.Fst:
    """Build sigma_star: the closure over all symbols in the table."""
    symbol_fsts = []
    for idx in range(syms.num_symbols()):
        sym = syms.find(idx)
        if sym == "<eps>" or sym == "":
            continue
        symbol_fsts.append(pynini.escape(sym))

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


def _parse_pattern(
    pattern: str,
    syms: pynini.SymbolTable,
    sigma_star: pynini.Fst,
) -> pynini.Fst:
    """Parse an MFA-style regex-like pattern into a pynini acceptor.

    Supported syntax (operating on individual characters within the
    pattern string, since MFA compiles to Python regexes):

    - Literal characters are concatenated as pynini symbols.
    - ``[abc]`` — union of characters a, b, c.
    - ``[^abc]`` — union of all symbols *except* a, b, c.
    - ``?`` after any element — makes it optional.
    - ``.*`` — sigma_star (match anything).
    - ``$`` — word/utterance boundary (accepted as empty string /
      epsilon, since our FSTs operate on isolated utterance chunks).

    Space-separated tokens in the *original* YAML field are handled
    by the caller (``_compile_element``); this function handles the
    regex-like sub-expressions within a single token or within a
    non-space-separated pattern string.
    """
    all_syms = _all_symbols(syms)

    tokens = _PATTERN_TOKEN_RE.findall(pattern)
    if not tokens:
        # Empty pattern → epsilon
        return pynini.accep("", token_type="utf8")

    parts: list[pynini.Fst] = []
    i = 0
    while i < len(tokens):
        tok = tokens[i]

        if tok == ".*":
            parts.append(sigma_star.copy())
            i += 1

        elif tok == "$":
            # Boundary — treat as epsilon (we operate on utterance chunks)
            i += 1

        elif tok.startswith("[^") and tok.endswith("]"):
            # Negated character class
            excluded = set(tok[2:-1])
            included = all_syms - excluded
            if not included:
                raise ValueError(
                    f"Negated class {tok} excludes all symbols"
                )
            fst = pynini.union(
                *[pynini.escape(s) for s in sorted(included)]
            )
            # Check for trailing ?
            if i + 1 < len(tokens) and tokens[i + 1] == "?":
                fst = pynini.union(fst, pynini.accep("", token_type="utf8"))
                i += 2
            else:
                i += 1
            parts.append(fst)

        elif tok.startswith("[") and tok.endswith("]"):
            # Character class
            chars = list(tok[1:-1])
            fst = pynini.union(*[pynini.escape(c) for c in chars])
            if i + 1 < len(tokens) and tokens[i + 1] == "?":
                fst = pynini.union(fst, pynini.accep("", token_type="utf8"))
                i += 2
            else:
                i += 1
            parts.append(fst)

        elif tok == "?":
            # Stray ? without a preceding group — skip
            i += 1

        else:
            # Literal character
            fst = pynini.escape(tok)
            if i + 1 < len(tokens) and tokens[i + 1] == "?":
                fst = pynini.union(fst, pynini.accep("", token_type="utf8"))
                i += 2
            else:
                i += 1
            parts.append(fst)

    if not parts:
        return pynini.accep("", token_type="utf8")

    result = parts[0]
    for p in parts[1:]:
        result = pynini.concat(result, p)
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

    An empty string produces epsilon (empty acceptor).
    """
    if not field_value:
        return pynini.accep("", token_type="utf8")

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
                parts.append(pynini.escape(tok))
        result = parts[0]
        for p in parts[1:]:
            result = pynini.concat(result, p)
        return result.optimize()

    if has_regex:
        return _parse_pattern(field_value, syms, sigma_star)

    # Plain single phoneme symbol
    return pynini.escape(field_value)


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

    required_keys = {"segment", "replacement", "preceding_context", "following_context"}
    for i, rule in enumerate(rules):
        missing = required_keys - set(rule.keys())
        if missing:
            raise ValueError(
                f"Rule {i} is missing keys: {missing}. "
                f"Required: {required_keys}"
            )

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
