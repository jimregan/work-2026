from __future__ import annotations

import copy
import random
from typing import Any

from timit_utils import TIMIT_61_PHONES, SILENCE_PHONES

# Phones eligible for insertion: exclude silence and closure bursts
_INSERT_INVENTORY: list[str] = sorted(TIMIT_61_PHONES - SILENCE_PHONES - {
    "bcl", "dcl", "gcl", "pcl", "tcl", "kcl",
})


def insert_phone(phones: list[str], rng: random.Random) -> list[str]:
    phone = rng.choice(_INSERT_INVENTORY)
    pos = rng.randint(0, len(phones))
    result = phones[:]
    result.insert(pos, phone)
    return result


def delete_phone(phones: list[str], rng: random.Random) -> list[str]:
    if len(phones) <= 1:
        return phones[:]
    pos = rng.randrange(len(phones))
    result = phones[:]
    del result[pos]
    return result


def swap_phones(phones: list[str], rng: random.Random) -> list[str]:
    if len(phones) < 2:
        return phones[:]
    pos = rng.randrange(len(phones) - 1)
    result = phones[:]
    result[pos], result[pos + 1] = result[pos + 1], result[pos]
    return result


EDIT_OPS: list[tuple[str, Any]] = [
    ("insert", insert_phone),
    ("delete", delete_phone),
    ("swap", swap_phones),
]


def apply_edits(
    lexicon: dict[str, list[list[str]]],
    n_edits: int,
    rng: random.Random,
) -> tuple[dict[str, list[list[str]]], list[dict]]:
    """
    Apply exactly n_edits random edits to the lexicon.

    Each edit:
      1. Choose a word type uniformly at random from the lexicon keys.
      2. Choose the first pronunciation (index 0) as the edit target.
      3. Choose an op uniformly at random and apply it.

    Returns (modified_lexicon, edit_log) where each edit_log entry is:
      {"word": str, "op": str, "before": [...], "after": [...]}
    """
    modified = copy.deepcopy(lexicon)
    words = list(modified.keys())
    log: list[dict] = []

    for _ in range(n_edits):
        word = rng.choice(words)
        op_name, op_fn = rng.choice(EDIT_OPS)
        before = modified[word][0][:]
        after = op_fn(before, rng)
        modified[word][0] = after
        log.append({"word": word, "op": op_name, "before": before, "after": after})

    return modified, log
