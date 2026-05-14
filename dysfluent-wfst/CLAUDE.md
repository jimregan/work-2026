# Detecting Phonetic Variation with WFSTs (OpenFST/Pynini + k2)

This document describes how to build a system for finding phonetic variations
in speech -- due to speaking rate, dialectal differences, or other sources --
using **pynini** (the Python wrapper for OpenFst) for graph construction, and
then importing the text-format FSTs into **k2** for the dense-lattice
intersection step (which has no OpenFst equivalent because it operates on
neural-network posteriors).

The approach is inspired by the zero-shot dysfluency detection system in
<https://arxiv.org/abs/2505.16351> (Interspeech 2025), but adapted:
the "error" paths in the WFST are reinterpreted as **variation paths** that
capture legitimate phonetic differences rather than disfluencies.

The system serves three goals:
1. **Lexicon validation**: check whether a phonetic lexicon's transcriptions
   match what speakers actually produce, flagging entries where the acoustic
   evidence consistently disagrees with the citation form.
2. **Rule validation**: test whether phonetic rules (implemented via
   cdrewrite) accurately predict observed variation -- rules whose predicted
   variants are never selected by the decoder may be wrong or inapplicable;
   variation that the decoder finds but no rule predicts reveals missing rules.
3. **Variation analysis**: characterise phonetic variation in speech corpora
   by speaking rate, dialect, or speaker.

Two feature sources are used:
- **Wav2Vec2 CTC** log-probabilities (frame-level phoneme posteriors)
- **Praat/Parselmouth** acoustic features (pitch, formants, intensity, etc.)

---

## High-level workflow

```
Audio ──► Wav2Vec2 CTC ──► log-probabilities (T×C tensor)
  │                                 │
  ├──► Parselmouth ──► Praat features (pitch, formants, ...)
  │                         │
  │                    [optional: combine / augment posteriors]
  │                         │
Reference text ──► G2P ──► phoneme IDs
                                    │
                        ┌───────────┘
                        ▼
              Build reference FST        (pynini)
                with variation paths
              Build CTC topology         (pynini)
              Compose them               (pynini)
              Export to text format
                        │
                        ▼
              Import into k2             (k2.Fsa.from_str)
              Intersect with dense       (k2.intersect_dense)
              Shortest path / N-best     (k2.shortest_path)
                        │
                        ▼
              Analyse variation from state trajectory
              Cross-reference with Praat features
```

## Dependencies

- `pynini` (and therefore `openfst`) -- for FST construction and composition
- `k2` -- for dense-lattice intersection with CTC posteriors
- `torch`, `torchaudio` -- audio I/O and Wav2Vec2
- `transformers` -- Wav2Vec2ForCTC model
- `parselmouth` -- Python wrapper for Praat; acoustic feature extraction
- `cmudict` (or other G2P) -- grapheme-to-phoneme
- `numpy` -- phoneme similarity matrix and feature arrays

---

## 1. Symbol tables

### 1.1 Phoneme inventory (input symbols)

The input symbol table must match the vocabulary of the wav2vec2 CTC model
being used. Extract it from the model's tokenizer/processor:

```python
from transformers import AutoProcessor
processor = AutoProcessor.from_pretrained("your-wav2vec2-model")
vocab = processor.tokenizer.get_vocab()
# vocab is a dict like {"<pad>": 0, "a": 1, "b": 2, ...}
```

Build a pynini `SymbolTable` from the model's vocabulary. Index 0 is
typically `<pad>` and doubles as the CTC blank. The indices **must**
match the model's output indices exactly:

```python
import pynini

input_syms = pynini.SymbolTable()
input_syms.add_symbol("<eps>", 0)  # OpenFst convention: 0 = epsilon

for symbol, idx in sorted(vocab.items(), key=lambda x: x[1]):
    if idx == 0:
        continue  # already have epsilon / blank at 0
    input_syms.add_symbol(symbol, idx)
```

The original repo uses a 277-symbol IPA inventory from
`config/lexicon.json`, but any CTC phoneme or character model will work
as long as the symbol table matches.

### 1.2 Output symbols

The output symbol table starts as a copy of the input table, but is
extended at runtime with transition-marker symbols of the form
`{i}<trans>{j}` that encode source and destination states. These markers
are how the decoder tracks state jumps for dysfluency detection later.
In pynini, add them to the output `SymbolTable` as you build arcs
(see Section 3).

### 1.3 Phoneme lexicons

There are two lexicons available:

#### (a) Citation-form lexicon

A Swedish phonetic lexicon (IPA-based) giving canonical/citation-form
pronunciations. Phonetic variation is modelled on top of this by
applying phonetic rules (see below).

#### (b) Acoustically validated lexicon

A second lexicon derived from a CTC-based pipeline: an orthographic
CTC model and a phonetic CTC model (same base model, different
fine-tuning) were run over a speech corpus, and their outputs were
aligned by timestamp. Where the orthographic model's word-level
timestamps matched the phonetic model's output, the phonetic
transcription was taken as the pronunciation for that word. This gives
pronunciations that reflect **actual observed realisations** rather than
citation forms.

This validated lexicon can serve as:
- **Ground truth for lexicon validation** (Section 8.1): compare the
  citation-form lexicon against it to find entries that disagree
- **An alternative reference for WFST decoding**: using validated
  pronunciations as the reference means the decoder is aligning
  against what speakers actually produce, so any remaining variation
  the decoder finds is genuinely unexpected
- **Training data for rule induction**: differences between citation
  and validated forms are examples of phonetic processes that the
  rules should capture

#### Building lexicon FSTs

Compile either lexicon into a string-map transducer:

```python
# lexicon_entries: list of (orthographic, phonemic) tuples
# e.g. [("huset", "h ʉː s ə t"), ("gata", "ɡ ɑː t a"), ...]
citation_fst = pynini.string_map(
    citation_entries,
    input_token_type="utf8",
    output_token_type=phone_syms,  # SymbolTable of phonemes
)

validated_fst = pynini.string_map(
    validated_entries,
    input_token_type="utf8",
    output_token_type=phone_syms,
)
```

Or, if the lexicon is large, build it from a text file using
`pynini.string_file`.

Words may have multiple validated pronunciations (from different
utterances). These naturally become multiple paths in the FST,
which is desirable -- the decoder picks the best-matching one.

#### Choosing which lexicon to use

| Task | Lexicon |
|------|---------|
| Validate citation-form entries | Decode with citation, compare against validated |
| Validate phonetic rules | Decode with citation + rules, compare against validated |
| Find unexpected variation | Decode with validated as reference |
| Build pronunciation lattice with rules | Citation + cdrewrite (Section 1.4) |
| Build pronunciation lattice without rules | Validated (already reflects real speech) |

### 1.4 Phonetic rules with cdrewrite

Each phonetic rule is a context-dependent rewrite rule compiled with
`pynini.cdrewrite`. The signature is:

```python
rule = pynini.cdrewrite(
    tau,      # the transduction: what changes (e.g. "ɡ" -> "ɣ")
    lam,      # left context
    rho,      # right context
    sigma_star,  # closure over the alphabet
    direction="ltr",   # left-to-right (default) or "rtl" or "sim"
    mode="obl",        # obligatory (default) or "opt" for optional
)
```

**Use `mode="opt"` for variation modelling**: this creates a transducer
that non-deterministically either applies or skips the rule, so both
the citation form and the variant survive in the output lattice.

Example Swedish phonetic rules:

```python
sigma_star = pynini.union(
    *[pynini.escape(s) for s in all_phonemes + ["[BOS]", "[EOS]"]]
).closure().optimize()

vowel = pynini.union(*[pynini.escape(v) for v in vowels])
voiced = pynini.union(*[pynini.escape(c) for c in voiced_consonants])

# Retroflexion: /r/ + dental -> retroflex
#   /r n/ -> /ɳ/, /r d/ -> /ɖ/, /r t/ -> /ʈ/, /r s/ -> /ʂ/, /r l/ -> /ɭ/
retroflex_n = pynini.cdrewrite(
    pynini.cross("r n", "ɳ"), "", "", sigma_star, mode="opt"
)
retroflex_d = pynini.cdrewrite(
    pynini.cross("r d", "ɖ"), "", "", sigma_star, mode="opt"
)
retroflex_t = pynini.cdrewrite(
    pynini.cross("r t", "ʈ"), "", "", sigma_star, mode="opt"
)
retroflex_s = pynini.cdrewrite(
    pynini.cross("r s", "ʂ"), "", "", sigma_star, mode="opt"
)
retroflex_l = pynini.cdrewrite(
    pynini.cross("r l", "ɭ"), "", "", sigma_star, mode="opt"
)

# Voiced stop lenition: /ɡ/ -> [ɣ] between vowels
g_lenition = pynini.cdrewrite(
    pynini.cross("ɡ", "ɣ"), vowel, vowel, sigma_star, mode="opt"
)

# Schwa deletion in unstressed syllables (fast speech)
schwa_del = pynini.cdrewrite(
    pynini.cross("ə", ""), "", "", sigma_star, mode="opt"
)

# Compose rules into a single cascade
rules = pynini.compose(
    retroflex_n,
    pynini.compose(retroflex_d,
    pynini.compose(retroflex_t,
    pynini.compose(retroflex_s,
    pynini.compose(retroflex_l,
    pynini.compose(g_lenition,
                   schwa_del)))))
).optimize()
```

#### Applying rules to the lexicon

Compose the lexicon with the rule cascade to get a transducer that maps
orthographic forms to all phonetically plausible surface realisations:

```python
varied_lexicon = pynini.compose(lexicon_fst, rules).optimize()
```

For a single utterance, look up the word sequence and compose with rules:

```python
def get_pronunciation_lattice(words, lexicon_fst, rules, phone_syms):
    """Return an FSA over phone_syms containing citation and variant forms."""
    word_fsts = []
    for word in words:
        # Look up word in lexicon
        word_input = pynini.escape(word)
        word_pron = pynini.compose(word_input, lexicon_fst)
        # Apply optional phonetic rules
        word_varied = pynini.compose(word_pron, rules)
        word_fsts.append(word_varied.project("output"))
    # Concatenate word pronunciations
    utt_fst = word_fsts[0]
    for wf in word_fsts[1:]:
        utt_fst = pynini.concat(utt_fst, wf)
    return utt_fst.optimize()
```

This pronunciation lattice replaces the linear reference FSA from the
original system. It already encodes variation, so the reference FST
(Section 3) can be built from **paths through this lattice** rather
than from a single phoneme sequence. Alternatively, compose this
lattice directly into the decoding pipeline in place of the linear
reference.

#### Interaction with the reference FST variation arcs

There are two complementary sources of variation:

1. **Rule-based** (cdrewrite): known, linguistically motivated phonetic
   processes (retroflexion, lenition, elision, etc.)
2. **Data-driven** (substitution/skip/back arcs in the reference FST):
   catches variation not covered by explicit rules, based on acoustic
   similarity

These can be combined: apply the rule cascade first to expand the
pronunciation lattice, then build the reference FST with additional
substitution/skip arcs on top. This way, known rules get preferred
paths (lower cost) while the similarity-based arcs serve as a fallback.

### 1.5 Phoneme similarity matrix for substitution arcs

The substitution arcs need a phoneme similarity matrix. The original
uses a 41x41 ARPAbet matrix (`utils/rule_sim_matrix.npy`) indexed by:

```
"|":0  OW:1  UW:2  EY:3  AW:4  AH:5  AO:6  AY:7  EH:8  K:9
NG:10  F:11  JH:12 M:13  CH:14 IH:15 UH:16 HH:17 L:18  AA:19
R:20   TH:21 AE:22 D:23  Z:24  OY:25 DH:26 IY:27 B:28  W:29
S:30   T:31  SH:32 ZH:33 ER:34 V:35  Y:36  N:37  G:38  P:39
"-":40
```

If the new model uses a different phoneme set, you will need to either:
- Map model phonemes to ARPAbet to index into this matrix, or
- Build a new similarity matrix for the model's phoneme set (e.g. based
  on articulatory features or acoustic confusability)

---

## 2. CTC topology (pynini)

The CTC topology is a single-state-per-token FSA that allows:
- Self-loops for frame repetitions of any label (including blank)
- Transitions from blank to any label and vice versa
- No direct label-to-different-label transitions (must go through blank)

In the original code this is `k2.ctc_topo(max_token=N, modified=False)`.

Rebuild it in pynini as a transducer with `N+1` states (one per token
plus a final state). For standard (non-modified) CTC:

```python
def build_ctc_topo(num_tokens, syms):
    """Build standard CTC topology.

    State 0 = blank state.
    States 1..num_tokens-1 = one per non-blank token.

    Arcs:
      - From state 0: self-loop on blank (0:0);
        arc to state t on input t / output t, for each t in 1..num_tokens-1
      - From state t: self-loop on token t (t:t);
        arc to state 0 on blank (0:0);
        arc to state t' on input t' / output t', for each t' != t
    """
    compiler = pynini.Compiler()
    blank = 0
    # state 0: blank state
    compiler.add_arc(0, pynini.Arc(blank, blank, 0, 0))  # self-loop
    for t in range(1, num_tokens):
        compiler.add_arc(0, pynini.Arc(t, t, 0, t))      # blank -> token
    # states 1..N-1: token states
    for t in range(1, num_tokens):
        compiler.add_arc(t, pynini.Arc(t, t, 0, t))      # self-loop
        compiler.add_arc(t, pynini.Arc(blank, blank, 0, 0))  # -> blank
        for t2 in range(1, num_tokens):
            if t2 != t:
                compiler.add_arc(t, pynini.Arc(t2, t2, 0, t2))
    # all states are final
    for s in range(num_tokens):
        compiler.set_final(s, 0)
    fst = compiler.compile()
    fst.set_input_symbols(syms)
    fst.set_output_symbols(syms)
    return fst
```

> **Note:** The full CTC topology above has O(N^2) arcs. For large
> vocabularies, factor it as two smaller transducers or use the
> k2 version directly. For the 277-token inventory here it is fine.

---

## 3. Reference FSA graph (pynini)

This is the core graph that encodes the expected (citation-form) phoneme
sequence plus all allowed **variation paths**. In the original dysfluency
system these model repetitions, deletions, etc.; here they model
legitimate phonetic variation:

- **Substitution arcs** = phoneme realisations that differ from citation
  form (e.g. dialect-specific vowel shifts, flapping, lenition)
- **Skip/deletion arcs** = phoneme elision due to fast speech or dialect
  (e.g. consonant cluster simplification, schwa deletion)
- **Back/repetition arcs** = can still model hesitation repetitions if
  desired, but may also capture gemination or lengthening effects

Given a phoneme-ID sequence `phones = [p0, p1, ..., pL-1]` of length L:

### States

- States `0` through `L`: state `i` means "have consumed up to phoneme i"
- State `L+1`: superfinal

### Arc types

For each state `i` (0-indexed phoneme position), for each state `j`:

#### (a) Correct transition: `i -> i+1`
- Input label: `phones[i]`, output label: `phones[i]`
- Weight: `alpha = 1 - 10^(-beta)` (default beta=5, so alpha ~ 0.99999)

#### (b) Substitution arcs: `i -> i+1`  (if `sub=True`)
- For each of the top-2 most similar phonemes to `phones[i]`
  (looked up via the 41x41 similarity matrix after converting to CMU):
  - Input label: `0` (epsilon/blank)
  - Output label: a dynamic `<trans>` marker `f"{i}<trans>{i+1}"`
  - Weight: `error_score / 10000` where `error_score = 1 - alpha`

#### (c) Skip/deletion arcs: `i -> j` where `j > i+1` and `j - i <= 3`  (if `skip=True`)
- Input label: `0` (epsilon)
- Output label: a dynamic `<trans>` marker `f"{i}<trans>{j}"`
- Weight: `error_score * exp(-(i-j)^2 / 2)`

#### (d) Back/repetition arcs: `i -> j` where `j < i` and `i - j <= 2`  (if `back=True`)
- Input label: `0` (epsilon)
- Output label: a dynamic `<trans>` marker `f"{i}<trans>{j}"`
- Weight: `error_score * exp(-(i-j)^2 / 2)`

#### (e) Final arc: `L -> L+1`
- Input label: `-1`, output label: `-1`, weight: `0`
- (This is k2's convention for the final arc; in OpenFst, just mark
  state `L` as final with weight 0 and skip the superfinal state.)

### Building in pynini

```python
import math

def build_ref_fst(phoneme_ids, beta, input_syms, output_syms,
                  similarity_matrix, phn2idx, lexicon,
                  skip=False, back=True, sub=True, is_ipa=False):
    alpha = 1 - 10**(-beta)
    error_score = 1 - alpha

    compiler = pynini.Compiler(
        isymbols=input_syms,
        osymbols=output_syms,
    )

    L = len(phoneme_ids)
    next_osym_id = output_syms.num_symbols()

    for i, phone in enumerate(phoneme_ids):
        for j in range(L + 1):
            if i == j:
                continue

            if j == i + 1:
                # (a) Correct transition
                w = -math.log(alpha)  # OpenFst uses tropical weights by default
                compiler.add_arc(i, pynini.Arc(phone, phone, w, j))

                if skip:
                    # Skip arc on correct transition
                    marker = f"{i}<trans>{j}"
                    mid = output_syms.add_symbol(marker, next_osym_id)
                    next_osym_id += 1
                    w_err = -math.log(error_score * math.exp(-(i-j)**2/2))
                    compiler.add_arc(i, pynini.Arc(0, mid, w_err, j))
                    continue

                if sub:
                    # (b) Substitution arcs
                    phone_text = lexicon[phone]
                    cmu_phone = ipa_to_cmu(phone_text) if is_ipa else phone_text
                    if cmu_phone in phn2idx:
                        sim_idx = phn2idx[cmu_phone]
                        top2 = torch.topk(similarity_matrix[sim_idx], 2).indices
                        for sid in top2:
                            if sid == sim_idx:
                                continue
                            marker = f"{i}<trans>{j}"
                            mid = output_syms.add_symbol(marker, next_osym_id)
                            next_osym_id += 1
                            w_sub = -math.log(error_score / 10000)
                            compiler.add_arc(i, pynini.Arc(0, mid, w_sub, j))
            else:
                if alpha == 1:
                    continue
                if j > i and skip and j - i <= 3:
                    # (c) Skip / deletion
                    marker = f"{i}<trans>{j}"
                    mid = output_syms.add_symbol(marker, next_osym_id)
                    next_osym_id += 1
                    w_skip = -math.log(error_score * math.exp(-(i-j)**2/2))
                    compiler.add_arc(i, pynini.Arc(0, mid, w_skip, j))
                elif j < i and back and i - j <= 2:
                    # (d) Back / repetition
                    marker = f"{i}<trans>{j}"
                    mid = output_syms.add_symbol(marker, next_osym_id)
                    next_osym_id += 1
                    w_back = -math.log(error_score * math.exp(-(i-j)**2/2))
                    compiler.add_arc(i, pynini.Arc(0, mid, w_back, j))

    # Final state
    compiler.set_final(L, 0)

    fst = compiler.compile()
    return fst
```

### Weight convention difference

**Important:** The original code stores raw probabilities as arc weights
and relies on k2's log-semiring operations. OpenFst's tropical semiring
uses negative log probabilities. When building in pynini, convert:
`w_openfst = -log(w_original)`. When exporting to k2 text format, convert
back if needed, or keep the log-semiring weights and specify the semiring
when loading in k2.

---

## 4. Composition

In pynini:

```python
ctc_topo = build_ctc_topo(num_tokens, syms)
ref_fst = build_ref_fst(phoneme_ids, beta=5, ...)

# Sort for composition
ctc_topo.arcsort("olabel")
ref_fst.arcsort("ilabel")

composed = pynini.compose(ctc_topo, ref_fst)
```

This replaces the `k2.compose(ctc_fsa, ref_fsa)` call.

---

## 5. Export to k2 text format

k2's `Fsa.from_str()` reads a simple text format where each line is:

```
src_state dest_state input_label output_label weight
```

and the last line is just the final state number. Use OpenFst's text
printer or iterate over arcs:

```python
def fst_to_k2_str(fst):
    lines = []
    for state in fst.states():
        for arc in fst.arcs(state):
            lines.append(
                f"{state} {arc.nextstate} {arc.ilabel} {arc.olabel} {arc.weight}"
            )
        if fst.final(state) != pynini.Weight.zero("tropical"):
            # mark as final: add arc to superfinal with label -1
            superfinal = fst.num_states()
            lines.append(f"{state} {superfinal} -1 -1 {fst.final(state)}")
    lines.append(str(fst.num_states()))  # superfinal state
    return "\n".join(lines)
```

Then load in k2:

```python
import k2
composed_k2 = k2.Fsa.from_str(fst_to_k2_str(composed), acceptor=False)
composed_k2 = k2.arc_sort(composed_k2).to(device)
```

---

## 6. Dense intersection and decoding (k2 only)

This part must stay in k2 because it intersects a symbolic FSA with a
dense matrix of neural-network log-probabilities:

```python
# Dense FSA from CTC logits
log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
supervision_segments = torch.tensor([[0, 0, T]], dtype=torch.int32)
dense_fsa = k2.DenseFsaVec(log_probs, supervision_segments)

# Intersect
lattice = k2.intersect_dense(composed_k2, dense_fsa, output_beam=25)

# Best path
shortest = k2.shortest_path(lattice, use_double_scores=True)
output_labels = shortest[0].aux_labels[:-1].tolist()
```

---

## 7. Variation detection from state trajectory

After decoding, the output label sequence contains a mix of phoneme IDs
and `<trans>` markers. The detection algorithm:

1. **Deduplicate** consecutive identical labels, filter out special tokens
   (`<pad>`, `<s>`, `</s>`, `<unk>`, `|`, `-`, `SIL`, `SPN`).

2. **Merge consecutive `<trans>` markers**: if two `<trans>` tokens appear
   in a row, merge them: keep the source of the first and the destination
   of the last. E.g. `3<trans>5` followed by `5<trans>7` becomes `3<trans>7`.

3. **Build a state trajectory**: walk through the merged sequence. Each
   `<trans>` token updates the current state to its destination. Each
   phoneme token occupies `(current_state, current_state+1)`.

4. **Classify each phoneme** (reinterpreted for variation analysis):
   - **Substitution**: the decoded phoneme differs from the citation-form
     phoneme at this position -- indicates a dialectal or rate-related
     variant (e.g. vowel shift, flapping, lenition)
   - **Deletion/elision**: `start_state > prev_end + 1` (skipped states) --
     phonemes dropped due to fast speech or dialect (consonant cluster
     simplification, unstressed vowel deletion, etc.)
   - **Repetition/lengthening**: `start_state` already appeared in state
     history -- may indicate gemination, emphasis, or hesitation
   - **Insertion**: `start_state < min(state_history)` -- epenthetic
     segments (e.g. intrusive /r/, vowel epenthesis)
   - **Normal**: canonical realisation matching citation form

---

## 8. Lexicon and rule validation

The WFST decoding pipeline can be used to validate both the phonetic
lexicon and the phonetic rules by comparing what the lexicon/rules
predict against what the acoustic model actually hears.

### 8.1 Lexicon validation

**Goal**: find lexicon entries whose citation-form transcription does not
match the acoustic evidence, suggesting the transcription is wrong.

#### Method: decode with citation form only (no variation arcs)

Run the decoder with `sub=False, skip=False, back=False` so only the
exact citation-form path is available. Compare the lattice score (total
path cost) against a threshold:

```python
# Decode with no variation allowed -- forces citation form
lattice_strict, ref_phones = build_lattice(
    emission, length, ref_text,
    beta=5, back=False, skip=False, sub=False, num_beam=25
)
strict_score = lattice_strict.get_tot_scores(
    log_semiring=True, use_double_scores=True
)

# Decode with full variation -- lets the decoder find the best match
lattice_free, _ = build_lattice(
    emission, length, ref_text,
    beta=2, back=True, skip=True, sub=True, num_beam=25
)
free_score = lattice_free.get_tot_scores(
    log_semiring=True, use_double_scores=True
)

# Large gap => citation form is a poor match; lexicon entry may be wrong
score_gap = free_score - strict_score
```

#### Method: comparison against the validated lexicon

Since an acoustically validated lexicon already exists (Section 1.3b),
a direct comparison is possible without re-decoding. For each word
present in both lexicons, align the citation-form and validated
transcriptions (e.g. via edit distance) and flag positions that differ:

```python
def compare_lexicons(citation_lex, validated_lex):
    """Find words where citation and validated pronunciations disagree."""
    mismatches = []
    for word in citation_lex:
        if word not in validated_lex:
            continue
        cit = citation_lex[word]       # e.g. ["h", "ʉː", "s", "ə", "t"]
        val = validated_lex[word]      # may be a list of variant prons
        for val_pron in val:
            ops = edit_operations(cit, val_pron)
            if any(op != "match" for op in ops):
                mismatches.append({
                    "word": word,
                    "citation": cit,
                    "validated": val_pron,
                    "operations": ops,
                })
    return mismatches
```

Where the validated lexicon consistently disagrees at the same position
across multiple speakers/utterances, the citation-form entry is suspect.
Where only some speakers show the difference, it is likely genuine
variation rather than a lexicon error.

#### Method: forced alignment comparison

Alternatively, compare two forced alignments:
1. Force-align against the **citation-form lexicon transcription**
2. Force-align against the **validated lexicon transcription**
   (or an unconstrained CTC decoding)

Where the two consistently disagree at the same position across multiple
utterances of the same word, the citation-form entry is suspect.

#### Aggregation across utterances

For each lexicon entry, collect results across all utterances containing
that word. Flag entries where:
- The citation-form score is consistently poor (high cost)
- The same substitution is selected by the decoder in a majority of
  occurrences (suggests a systematic transcription error, not variation)
- The decoded phoneme at a given position never matches the lexicon
  phoneme (even across different speakers/rates)

```python
from collections import Counter

# Per-word, per-position: what does the decoder choose?
word_position_counts = defaultdict(lambda: defaultdict(Counter))

for result in all_results:
    word = result["word"]
    for item in result["dys_detect"]:
        pos = item["start_state"]
        decoded = item["phoneme"]
        expected = ref_phones[pos]
        word_position_counts[word][pos][decoded] += 1

# Flag positions where the most common decoded phoneme != expected
for word, positions in word_position_counts.items():
    for pos, counts in positions.items():
        most_common = counts.most_common(1)[0]
        if most_common[0] != expected_phones[word][pos]:
            print(f"Suspect: {word} pos {pos}: lexicon has "
                  f"{expected_phones[word][pos]}, "
                  f"decoder prefers {most_common[0]} "
                  f"({most_common[1]} times)")
```

### 8.2 Rule validation

**Goal**: test whether each cdrewrite rule correctly predicts observed
variation, and find variation that no rule covers.

#### Offline validation against the validated lexicon

Since the validated lexicon (Section 1.3b) already contains observed
pronunciations, rules can be tested without any audio decoding at all:
apply the rule cascade to citation-form entries and check whether the
validated pronunciation appears among the outputs.

```python
def validate_rules_offline(citation_lex, validated_lex, rules, phone_syms):
    """Check which validated pronunciations are reachable via the rules."""
    reachable = 0
    unreachable = []
    for word in citation_lex:
        if word not in validated_lex:
            continue
        # Apply rules to citation form -> lattice of possible variants
        cit_fst = pynini.accep(
            citation_lex[word], token_type=phone_syms
        )
        varied = pynini.compose(cit_fst, rules).optimize()
        # Check each validated pronunciation
        for val_pron in validated_lex[word]:
            val_fst = pynini.accep(val_pron, token_type=phone_syms)
            intersection = pynini.intersect(varied, val_fst)
            if intersection.start() != pynini.NO_STATE_ID:
                reachable += 1
            else:
                unreachable.append({
                    "word": word,
                    "citation": citation_lex[word],
                    "validated": val_pron,
                })
    return reachable, unreachable
```

Unreachable validated pronunciations indicate either missing rules or
rules with overly restrictive contexts.

#### Testing individual rules with audio

For each rule, build two decoder configurations:
1. **With the rule** applied to the pronunciation lattice
2. **Without the rule** (citation form only, or all rules except this one)

Compare the lattice scores:

```python
def test_rule(utterances, lexicon_fst, all_rules, rule_to_test):
    """Compare decoding with and without a specific rule."""
    # All rules except the one under test
    other_rules = compose_all_except(all_rules, rule_to_test)

    results = {"with_rule": [], "without_rule": []}
    for utt in utterances:
        # Pronunciation lattice WITH the rule
        pron_with = pynini.compose(
            lexicon_fst, pynini.compose(other_rules, rule_to_test)
        )
        score_with = decode_and_score(utt, pron_with)

        # Pronunciation lattice WITHOUT the rule
        pron_without = pynini.compose(lexicon_fst, other_rules)
        score_without = decode_and_score(utt, pron_without)

        results["with_rule"].append(score_with)
        results["without_rule"].append(score_without)

    return results
```

#### Rule validation outcomes

| Outcome | Interpretation |
|---------|---------------|
| Rule path selected, score improves | Rule correctly predicts observed variation |
| Rule path available but never selected | Rule may be wrong, or context doesn't arise in the data |
| No rule covers observed variation | Missing rule -- decoder uses fallback substitution arcs instead |
| Rule path selected but score worsens | Rule applies in wrong context -- check left/right context constraints |

#### Detecting missing rules

When the decoder consistently selects a **fallback substitution arc**
(from the similarity-based arcs in Section 3) at the same phoneme
position, this suggests a systematic phonetic process not captured by
any existing rule:

```python
# Collect fallback substitutions (not covered by cdrewrite rules)
missing_patterns = defaultdict(Counter)

for result in all_results:
    for item in result["dys_detect"]:
        if item["dysfluency_type"] == "substitution":
            pos = item["start_state"]
            expected = ref_phones[pos]
            actual = item["phoneme"]
            # Check if any rule predicts this substitution
            if not any_rule_predicts(expected, actual, context):
                missing_patterns[(expected, actual)][context] += 1

# Frequent uncovered patterns suggest new rules to write
for (src, tgt), contexts in missing_patterns.items():
    total = sum(contexts.values())
    if total >= threshold:
        print(f"Possible missing rule: {src} -> {tgt} "
              f"({total} occurrences)")
        for ctx, count in contexts.most_common(3):
            print(f"  context: {ctx} ({count}x)")
```

### 8.3 Joint validation workflow

```
                    ┌──────────────────┐
                    │  Speech corpus   │
                    └────────┬─────────┘
                             │
              ┌──────────────┼──────────────┐
              ▼              ▼              ▼
        Decode with    Decode with    Decode with
        citation only  rules applied  rules + fallback
              │              │              │
              ▼              ▼              ▼
        Strict scores  Rule scores    Free scores
              │              │              │
              └──────┬───────┘              │
                     ▼                      │
              Compare: does the      Compare: does the
              rule improve fit?      fallback find more?
                     │                      │
                     ▼                      ▼
              Rule validation        Missing rule detection
                     │
              ┌──────┴──────┐
              ▼              ▼
        Lexicon entries    Rules that
        never match        never fire
        (fix lexicon)      (fix rules)
```

---

## 9. Praat/Parselmouth feature integration

Parselmouth provides a second source of acoustic evidence that can
complement the wav2vec2 posteriors and help disambiguate variation types.

**The Praat analysis is designed to run independently of the WFST
decoding pipeline.** The two stages communicate through serialised
intermediate results (alignment files), so they can run at different
times, on different machines, or be iterated independently.

### 9.1 Decoupled architecture

```
 Stage 1: WFST decoding (can run first, produces alignment)
 ──────────────────────────────────────────────────────────
  Audio + Text ──► WFST pipeline ──► alignment JSON
                                      (per-phoneme: label, start/end
                                       frame, variation type, scores)
                         │
                         ▼
                    Save to disk
                    (one JSON per utterance, or a single JSONL/CSV)


 Stage 2: Praat analysis (can run later, reads alignment)
 ──────────────────────────────────────────────────────────
  Audio + alignment JSON ──► Parselmouth ──► enriched results
                                              (alignment + acoustic
                                               measurements per segment)
```

### 9.2 Alignment output format

The WFST decoder should serialise enough information for Praat to
work without re-running decoding. At minimum, per utterance:

```json
{
  "utterance_id": "spk01_utt042",
  "audio_path": "data/audio/spk01_utt042.wav",
  "ref_text": "huset är stort",
  "sample_rate": 16000,
  "frame_shift_ms": 20.0,
  "segments": [
    {
      "phoneme": "h",
      "ref_phoneme": "h",
      "start_frame": 5,
      "end_frame": 9,
      "start_time_s": 0.10,
      "end_time_s": 0.18,
      "variation_type": "normal",
      "ref_state": 0,
      "lattice_score": -1.23
    },
    {
      "phoneme": "ʉː",
      "ref_phoneme": "ʉː",
      "start_frame": 9,
      "end_frame": 15,
      "start_time_s": 0.18,
      "end_time_s": 0.30,
      "variation_type": "normal",
      "ref_state": 1,
      "lattice_score": -0.87
    }
  ]
}
```

Frame-to-time conversion: `time_s = start_frame * frame_shift_ms / 1000`.
The frame shift depends on the wav2vec2 model (typically 20 ms).

### 9.3 Extracting Praat features per segment

```python
import parselmouth
import json
import numpy as np

def enrich_with_praat(alignment_path):
    """Load a saved alignment and add Praat measurements to each segment."""
    with open(alignment_path) as f:
        alignment = json.load(f)

    snd = parselmouth.Sound(alignment["audio_path"])
    pitch = snd.to_pitch()
    formants = snd.to_formant_burg()
    intensity = snd.to_intensity()

    for seg in alignment["segments"]:
        t0 = seg["start_time_s"]
        t1 = seg["end_time_s"]
        tmid = (t0 + t1) / 2

        # Duration
        seg["duration_s"] = t1 - t0

        # Pitch (F0) -- mean and range over the segment
        f0_vals = [pitch.get_value_at_time(t)
                   for t in np.arange(t0, t1, 0.005)
                   if pitch.get_value_at_time(t)]
        f0_vals = [v for v in f0_vals if v and v > 0]
        seg["f0_mean"] = float(np.mean(f0_vals)) if f0_vals else None
        seg["f0_range"] = float(np.ptp(f0_vals)) if f0_vals else None

        # Formants at midpoint (most relevant for vowels)
        seg["f1"] = formants.get_value_at_time(1, tmid)
        seg["f2"] = formants.get_value_at_time(2, tmid)
        seg["f3"] = formants.get_value_at_time(3, tmid)

        # Intensity
        seg["intensity_mean"] = intensity.get_average(
            from_time=t0, to_time=t1
        )

    return alignment
```

### 9.4 Uses of Praat features in validation and analysis

Once the Praat-enriched results are available, they support several
analyses without re-running the WFST:

- **Vowel quality validation**: for segments labelled as a specific
  vowel, check whether F1/F2 fall in the expected region. A lexicon
  entry transcribed as /ʉː/ but whose F1/F2 consistently lands in
  /uː/ territory is suspect.
- **Duration-based rate estimation**: mean segment duration gives a
  per-utterance speaking rate estimate. Correlate with variation
  frequency (more elisions at higher rate = expected).
- **Formant-based substitution confirmation**: when the WFST decoder
  selects a vowel substitution, check if the formants agree with the
  substituted vowel rather than the citation vowel.
- **Rule context acoustics**: for each rule application, collect the
  acoustic profile of the context (e.g. for lenition: is the preceding
  vowel long? is intensity low?) to refine rule contexts.

### 9.5 Integration strategies (optional, for tighter coupling)

If at some point you want to feed Praat features back into the WFST
pipeline rather than just using them post-hoc:

#### (a) Augmenting the dense FSA scores
Combine wav2vec2 log-probabilities with Praat-derived scores before
building the `DenseFsaVec`. For example, adjust the posteriors of
vowel classes based on formant evidence:

```python
# Resample Praat features to match wav2vec2 frame rate
# Then modify log_probs before creating DenseFsaVec:
log_probs[:, :, vowel_indices] += formant_based_adjustment
```

#### (b) Praat-informed variation weights
Use Praat features to dynamically adjust the weights on variation arcs
in the reference FST. For instance, if speech rate is high (short
syllable durations, compressed F0 range), lower the cost of deletion
arcs; if formant measurements suggest a particular vowel shift, lower
the cost of the corresponding substitution arc.

#### (c) Separate feature streams
Build a second dense score matrix from Praat features (e.g. a simple
GMM or lookup-table mapping formant values to phoneme likelihoods)
and combine it with the wav2vec2 dense FSA via log-linear interpolation
before intersection.

---

## 10. Weighted Phoneme Error Rate (WPER)

(May be useful as an evaluation metric for how far a realisation
deviates from citation form.)

Standard edit-distance but with a weighted substitution cost:

```
sub_cost(a, b) = 1 - similarity_matrix[phn2idx[a], phn2idx[b]]
insertion_cost = 1
deletion_cost  = 1
WPER = dp[N][M] / N
```

This operates on CMU (ARPAbet) phoneme strings using the 41x41 matrix.

---

## 11. Key parameters

| Parameter  | Default | Effect |
|------------|---------|--------|
| `beta`     | 5       | Variation tolerance. `alpha = 1 - 10^(-beta)`. Lower values allow more variation from citation form. For phonetic variation work, values of 2-4 may be more appropriate than the original default of 5. |
| `num_beam` | 25      | Beam width for `k2.intersect_dense`. |
| `back`     | True    | Allow backward arcs (repetition/gemination). Max 2-state jump back. |
| `skip`     | False   | Allow skip arcs (elision/deletion). Max 3-phoneme skip. **Consider enabling for fast-speech analysis.** |
| `sub`      | True    | Allow substitution arcs using top-N similar phonemes. Consider increasing from top-2 to top-3 or more for dialect work where larger phonemic shifts are expected. |

---

## 12. Acoustic model

The system uses a Wav2Vec2-based CTC model. Audio is resampled to 16 kHz
before inference. The model produces a `(1, T, C)` tensor where T is the
number of frames (roughly 50 per second of audio) and C is the vocabulary
size.

The choice of wav2vec2 checkpoint determines:
- **C** (vocabulary size) and therefore the symbol table dimensions
- Whether the output labels are IPA, ARPAbet, characters, or something else
- How reference text must be converted to match the output vocabulary
- Whether the existing similarity matrix applies or a new one is needed

Extract the vocab from the model's processor (see Section 1.1) and use it
to drive all downstream symbol table construction.

---

## Summary of what pynini replaces vs. what stays in k2

| Operation | Original (k2) | Reimplementation |
|-----------|---------------|-----------------|
| CTC topology | `k2.ctc_topo()` | `pynini` (build manually) |
| Reference FST | `k2.Fsa.from_str()` | `pynini.Compiler` |
| Composition | `k2.compose()` | `pynini.compose()` |
| Arc sorting | `k2.arc_sort()` | `fst.arcsort()` |
| Dense intersection | `k2.intersect_dense()` | **k2** (no pynini equivalent) |
| Shortest path | `k2.shortest_path()` | **k2** (operates on dense lattice) |
| Variation detection | Python post-processing | Python post-processing (adapted) |
| Praat features | N/A | `parselmouth` (new) |
