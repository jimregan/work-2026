# Plan: Add exhaustive possessive-suffix forms for lemmatisation

## Goal

Add a separate function that generates all combinations of mutation × emphatic suffix for a noun, for use in building a lemmatisation lexicon. This complements the existing `get_possessive_forms()` which produces only the linguistically correct 7-person forms.

## Rationale

After numbers other than "dhá", the mutation on the noun may not match the standard possessive person rules. A lemmatiser needs to recognise any mutated+suffixed form it encounters, including forms with incorrect or missing hyphens.

## Design

### New function: `get_all_emphatic_forms()` on `Noun`

For each form in `self.sg_nom`, generate every combination of:
- **4 mutations**: `NoMut`, `Len1`, `PrefH`, `Ecl1`
- **6 suffixes**: `sa`, `se`, `san`, `sean`, `na`, `ne`

That's 24 combinations per base form.

**Hyphen tolerance**: for each combination where `_emph_suffix()` would insert a hyphen (e.g. "phas-sa"), also emit the non-hyphenated variant ("phassa"). And vice versa: where no hyphen is inserted, also emit the hyphenated variant if the join point is s+s or n+n. This handles common errors in either direction.

Returns a list of `(form_string, lemma)` tuples — the lemma is `self.get_lemma()`, so the output is directly usable as a lookup table mapping surface form → lemma.

### Files to modify

1. **`pygramadan/noun.py`** — add `get_all_emphatic_forms()` method
2. **`tests/test_noun.py`** — add tests for the new method

### No changes to existing code

`get_possessive_forms()`, `get_all_forms()`, and `_emph_suffix()` remain as-is.

## Implementation detail

```python
_EMPH_SUFFIXES = ['sa', 'se', 'san', 'sean', 'na', 'ne']
_EMPH_MUTATIONS = [Mutation.NoMut, Mutation.Len1, Mutation.PrefH, Mutation.Ecl1]

def get_all_emphatic_forms(self):
    """Generate all mutation × suffix combinations for lemmatisation.

    Returns list of (surface_form, lemma) tuples, including
    hyphen-tolerant variants.
    """
    lemma = self.get_lemma()
    forms = set()
    for nom in self.sg_nom:
        for mut in _EMPH_MUTATIONS:
            mutated = mutate(mut, nom.value)
            for suffix in _EMPH_SUFFIXES:
                # Canonical form (with correct hyphenation)
                canonical = _emph_suffix(mutated, suffix)
                forms.add((canonical, lemma))
                # Hyphen-tolerant variants
                if '-' in canonical:
                    # Strip the suffix hyphen (not eclipsis n-)
                    # Find last hyphen that's at the suffix join
                    no_hyphen = _remove_suffix_hyphen(canonical, suffix)
                    forms.add((no_hyphen, lemma))
                else:
                    # Add hyphenated variant if join point matches
                    with_hyphen = _add_suffix_hyphen(mutated, suffix)
                    if with_hyphen != canonical:
                        forms.add((with_hyphen, lemma))
    return sorted(forms)
```

The helpers `_remove_suffix_hyphen` and `_add_suffix_hyphen` handle the hyphen tolerance:

- `_remove_suffix_hyphen(form, suffix)`: if form ends with `-suffix`, remove that hyphen (but preserve any eclipsis `n-` prefix)
- `_add_suffix_hyphen(mutated, suffix)`: insert a hyphen between mutated form and suffix if last char of mutated matches first char of suffix (s/s or n/n)

## Tests

- `test_all_emphatic_forms_count()` — verify a simple noun produces expected number of forms (24 canonical + hyphen variants per base form)
- `test_all_emphatic_forms_contains_canonical()` — spot-check that known correct possessive forms appear
- `test_all_emphatic_forms_hyphen_tolerance()` — verify both "phas-sa" and "phassa" appear; both "n-arán-na" and "n-aránna" appear
- `test_all_emphatic_forms_returns_lemma()` — verify all tuples have the correct lemma

## Verification

```bash
pytest tests/test_noun.py -v
pytest
flake8 . --count --select=E9,F63,F7,F82 --show-source --statistics
```
