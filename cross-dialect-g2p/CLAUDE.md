# cross-dialect-g2p

## Idea

Treat English G2P as multitask learning: train on General American (GA) and
Received Pronunciation (RP), then investigate whether other dialects can be
learned as a few-shot task.

## Data

- **GA**: CMU Pronouncing Dictionary (`/Users/joregan/Playing/cmudict/cmudict.dict`)
  - ARPABET format, converted to IPA via `convert.py`
- **RP**: Britfone 3.0.1 (`/Users/joregan/Playing/Britfone/britfone.main.3.0.1.csv`)
  - Already in IPA (space-separated phones, stress marks included)

## Model

ByT5-small (byte-level T5) — avoids tokenisation issues with IPA characters.
Task-prefix format: `g2p en ga: word`, `g2p en rp: word`, etc.

## Tasks trained

| Task | Input | Output |
|------|-------|--------|
| G2P | `g2p en ga: hello` | `həˈloʊ` |
| G2P | `g2p en rp: hello` | `həˈləʊ` |
| P2G | `p2g en ga: həˈloʊ` | `hello` |
| P2G | `p2g en rp: həˈləʊ` | `hello` |
| Transduction | `transduce en ga to en rp: həˈloʊ` | `həˈləʊ` |
| Transduction | `transduce en rp to en ga: həˈləʊ` | `həˈloʊ` |
| Dialect ID | `identify dialect: həˈloʊ` | `en ga` / `en rp` / `en ga,rp` |

## Key design decisions

- **All pronunciation variants kept** by default (`--first-only` flag to restrict)
- **Transduction pairs** use minimum edit distance matching (via rapidfuzz), not
  cartesian product, to avoid spurious cross-variant pairings
- **Transduction skipped** when GA and RP differ only in secondary stress (ˌ)
- **Dialect ID** emits `en ga,rp` when a pronunciation is shared by both dialects
- **AH → ə** in GA (all stress levels); ˈ/ˌ stress marks emitted before the vowel.
  Phonological justification: /ʌ/ and /ə/ are allophones of the same phoneme
  (stressed vs unstressed), so `ˈə` represents the STRUT vowel in GA.

## Few-shot dialect generalisation — current thinking

For a third dialect, the only satisfactory approach is:

**Train transduction pairs** (`transduce en rp to en X:`, `transduce en X to en rp:`)
for the new dialect during training. This anchors the dialect prefix and gives the
model a prior on the phonology before few-shot `g2p en X:` examples arrive at
inference time.

This requires word-pronunciation pairs for the third dialect that overlap with
Britfone vocabulary (to generate the RP↔X transduction pairs).

**Open question at time of writing**: how to source minimal data for a third dialect,
and whether dialect-level specification via Wells lexical sets (e.g. TRAP→æ,
BATH→ɑː, FACE→æɪ) could be used to synthesise training data. Britfone's TRAP/BATH
distinction means lexical set membership is derivable from the RP side, which
partially solves the TRAP/BATH ambiguity in GA.

## Files

- `convert.py` — ARPABET→IPA converter
- `prepare_data.py` — builds `data/g2p.tsv` from CMU dict + Britfone
- `train.py` — ByT5 fine-tuning
- `predict.py` — inference + word accuracy / CER evaluation
- `pyproject.toml` — dependencies (transformers, datasets, torch, rapidfuzz)

## Running data prep

```bash
python prepare_data.py \
  --cmudict /Users/joregan/Playing/cmudict/cmudict.dict \
  --britfone /Users/joregan/Playing/Britfone/britfone.main.3.0.1.csv \
  --out data/g2p.tsv
```
