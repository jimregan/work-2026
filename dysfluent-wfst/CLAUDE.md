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

### 1.3 Phoneme mapping for reference text

The reference text must be converted to the same phoneme set as the
model's vocabulary. How this is done depends on the model:

- **IPA model**: Use CMUdict to get ARPAbet, then convert to IPA using a
  mapping like `config/ipa2cmu.json` (which maps IPA -> CMU; invert it).
- **ARPAbet model**: Use CMUdict directly, strip stress markers (digits).
- **Character model**: No phoneme conversion needed; use characters directly.
- **Other phoneme set**: Write or find an appropriate mapping.

The key requirement is that the phoneme IDs used to build the reference
FST are indices into the model's output vocabulary.

### 1.4 Phoneme similarity matrix for substitution arcs

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

## 8. Praat/Parselmouth feature integration

Parselmouth provides a second source of acoustic evidence that can
complement the wav2vec2 posteriors and help disambiguate variation types.

### 8.1 Extracting features

```python
import parselmouth
from parselmouth.praat import call

snd = parselmouth.Sound("utterance.wav")

# Pitch (F0) -- speaking rate correlate, intonation
pitch = snd.to_pitch()
f0_values = pitch.selected_array["frequency"]  # per-frame F0

# Formants -- vowel quality / dialect variation
formants = snd.to_formant_burg()
# Extract F1-F3 at each time step:
times = [formants.get_time_from_frame_number(i+1)
         for i in range(formants.get_number_of_frames())]
f1 = [formants.get_value_at_time(1, t) for t in times]
f2 = [formants.get_value_at_time(2, t) for t in times]
f3 = [formants.get_value_at_time(3, t) for t in times]

# Intensity -- stress patterns, reduction
intensity = snd.to_intensity()

# Duration / speaking rate
# Can be derived from the WFST alignment: phoneme durations in frames
```

### 8.2 Integration strategies

There are several ways to combine Praat features with the WFST output:

#### (a) Post-hoc annotation
Run the WFST decoder first, then use the frame-level alignment to
extract Praat features for each decoded phoneme segment. This enriches
each phoneme with acoustic measurements (F0, formants, intensity,
duration) for downstream analysis. No changes to the WFST itself.

#### (b) Augmenting the dense FSA scores
Combine wav2vec2 log-probabilities with Praat-derived scores before
building the `DenseFsaVec`. For example, adjust the posteriors of
vowel classes based on formant evidence:

```python
# Resample Praat features to match wav2vec2 frame rate
# Then modify log_probs before creating DenseFsaVec:
log_probs[:, :, vowel_indices] += formant_based_adjustment
```

#### (c) Praat-informed variation weights
Use Praat features to dynamically adjust the weights on variation arcs
in the reference FST. For instance, if speech rate is high (short
syllable durations, compressed F0 range), lower the cost of deletion
arcs; if formant measurements suggest a particular vowel shift, lower
the cost of the corresponding substitution arc.

#### (d) Separate feature streams
Build a second dense score matrix from Praat features (e.g. a simple
GMM or lookup-table mapping formant values to phoneme likelihoods)
and combine it with the wav2vec2 dense FSA via log-linear interpolation
before intersection.

---

## 9. Weighted Phoneme Error Rate (WPER)

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

## 10. Key parameters

| Parameter  | Default | Effect |
|------------|---------|--------|
| `beta`     | 5       | Variation tolerance. `alpha = 1 - 10^(-beta)`. Lower values allow more variation from citation form. For phonetic variation work, values of 2-4 may be more appropriate than the original default of 5. |
| `num_beam` | 25      | Beam width for `k2.intersect_dense`. |
| `back`     | True    | Allow backward arcs (repetition/gemination). Max 2-state jump back. |
| `skip`     | False   | Allow skip arcs (elision/deletion). Max 3-phoneme skip. **Consider enabling for fast-speech analysis.** |
| `sub`      | True    | Allow substitution arcs using top-N similar phonemes. Consider increasing from top-2 to top-3 or more for dialect work where larger phonemic shifts are expected. |

---

## 11. Acoustic model

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
