# Recreating the Dysfluency Detection WFST with OpenFST/Pynini

This document describes how to reimplement the zero-shot speech dysfluency
detection system from this repository using **pynini** (the Python wrapper for
OpenFst) for graph construction, and then importing the text-format FSTs into
**k2** only for the dense-lattice intersection step (which has no OpenFst
equivalent because it operates on neural-network posteriors).

Reference paper: <https://arxiv.org/abs/2505.16351> (Interspeech 2025).

---

## High-level workflow

```
Audio ──► Wav2Vec2 CTC ──► log-probabilities (T×C tensor)
                                    │
Reference text ──► CMUdict ──► phoneme IDs
                                    │
                        ┌───────────┘
                        ▼
              Build reference FST   (pynini)
              Build CTC topology    (pynini)
              Compose them          (pynini)
              Export to text format
                        │
                        ▼
              Import into k2        (k2.Fsa.from_str)
              Intersect with dense  (k2.intersect_dense)
              Shortest path         (k2.shortest_path)
                        │
                        ▼
              Detect dysfluencies from state trajectory
```

## Dependencies

- `pynini` (and therefore `openfst`) -- for FST construction and composition
- `k2` -- for dense-lattice intersection with CTC posteriors
- `torch`, `torchaudio` -- audio I/O and Wav2Vec2
- `transformers` -- Wav2Vec2ForCTC model
- `cmudict` -- grapheme-to-phoneme
- `numpy` -- phoneme similarity matrix

---

## 1. Symbol tables

### 1.1 Phoneme inventory (input symbols)

The acoustic model uses a 277-symbol IPA-based phoneme inventory stored in
`config/lexicon.json`. Index 0 is `<pad>` and doubles as the CTC blank.
Build a pynini `SymbolTable` from this list:

```python
import pynini

input_syms = pynini.SymbolTable()
input_syms.add_symbol("<eps>", 0)  # OpenFst convention: 0 = epsilon

# Index the phoneme inventory starting at 1, but keep the original
# indices from lexicon.json because the CTC model emits those indices.
# Since lexicon[0] = "<pad>" which is CTC blank, map it to 0.
for idx, phone in enumerate(lexicon):
    if idx == 0:
        continue  # already have epsilon / blank at 0
    input_syms.add_symbol(phone, idx)
```

### 1.2 Output symbols

The output symbol table is the same as the input table, but extended at
runtime with transition-marker symbols of the form `{i}<trans>{j}` that
encode source and destination states. These markers are how the decoder
tracks state jumps for dysfluency detection later. In pynini, add them
to the output `SymbolTable` as you build arcs (see Section 3).

### 1.3 CMU-to-IPA mapping

`config/ipa2cmu.json` maps IPA symbols to ARPAbet (CMU) symbols.
The inverse mapping is needed to go from CMUdict output back to the
IPA labels used by the acoustic model. Some IPA symbols map to two
CMU symbols (e.g. `ʃ` -> `SH CH`); only the first is used.

### 1.4 CMU phoneme-to-index map (for similarity matrix)

A separate 41-symbol ARPAbet index is used to look up the 41x41
phoneme similarity matrix (`utils/rule_sim_matrix.npy`):

```
"|":0  OW:1  UW:2  EY:3  AW:4  AH:5  AO:6  AY:7  EH:8  K:9
NG:10  F:11  JH:12 M:13  CH:14 IH:15 UH:16 HH:17 L:18  AA:19
R:20   TH:21 AE:22 D:23  Z:24  OY:25 DH:26 IY:27 B:28  W:29
S:30   T:31  SH:32 ZH:33 ER:34 V:35  Y:36  N:37  G:38  P:39
"-":40
```

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

This is the core graph that encodes the expected phoneme sequence plus
all allowed disfluency paths. Given a phoneme-ID sequence
`phones = [p0, p1, ..., pL-1]` of length L:

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

## 7. Dysfluency detection from state trajectory

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

4. **Classify each phoneme**:
   - **Repetition**: `start_state` already appeared in state history
   - **Insertion**: `start_state < min(state_history)` (went before any
     previously seen state)
   - **Deletion**: `start_state > prev_end + 1` (skipped states); emit a
     `<del>` marker for the gap, then the phoneme as normal
   - **Normal**: otherwise

---

## 8. Weighted Phoneme Error Rate (WPER)

Standard edit-distance but with a weighted substitution cost:

```
sub_cost(a, b) = 1 - similarity_matrix[phn2idx[a], phn2idx[b]]
insertion_cost = 1
deletion_cost  = 1
WPER = dp[N][M] / N
```

This operates on CMU (ARPAbet) phoneme strings using the 41x41 matrix.

---

## 9. Key parameters

| Parameter  | Default | Effect |
|------------|---------|--------|
| `beta`     | 5       | Error tolerance. `alpha = 1 - 10^(-beta)`. Higher = stricter alignment. |
| `num_beam` | 25      | Beam width for `k2.intersect_dense`. |
| `back`     | True    | Allow backward arcs (repetition detection). Max 2-state jump back. |
| `skip`     | False   | Allow skip arcs (deletion detection). Max 3-phoneme skip. |
| `sub`      | True    | Allow substitution arcs using top-2 similar phonemes. |

---

## 10. Acoustic model

The system uses a Wav2Vec2-based CTC model that outputs logits over the
277-symbol IPA phoneme inventory. Audio is resampled to 16 kHz before
inference. The model produces a `(1, T, 272)` tensor where T is the
number of frames (roughly 50 per second of audio) and 272 is the
vocabulary size (277 symbols minus some unused slots, depending on the
specific model checkpoint).

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
| Dysfluency detection | Python post-processing | Python post-processing (unchanged) |
