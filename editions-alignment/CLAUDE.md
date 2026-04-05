# Claude Code Instructions: Literary Text Alignment Tool

## Project Overview

Build a command-line Python tool that aligns multiple editions of literary texts (novels, poems) at the sentence level. It uses **MosesSentenceSplitter** for sentence splitting, **Sentence Transformers** for semantic similarity scoring, and a **contiguity penalty** to discourage non-sequential matches. Output is exported as CSV, JSON, or XML.

---

## Tech Stack

- **Language**: Python 3.9+
- **Tokenizer**: `sacremoses` (MosesSentenceSplitter)
- **Embeddings**: `sentence-transformers` (e.g. `all-MiniLM-L6-v2`)
- **Alignment**: Dynamic programming over a semantic similarity matrix with a contiguity penalty
- **CLI**: `click`
- **Output**: `csv`, `json`, `xml.etree.ElementTree` (all stdlib)

---

## Project Structure

```
text-aligner/
├── aligner.py        # CLI entry point
├── tokenizer.py      # Sentence splitting via MosesSentenceSplitter
├── embedder.py       # Sentence Transformer embedding + similarity matrix
├── align.py          # DP alignment with contiguity penalty
├── exporters.py      # CSV / JSON / XML writers
├── requirements.txt
└── README.md
```

**`requirements.txt`:**
```
sacremoses>=0.0.53
sentence-transformers>=2.2.0
click>=8.0
numpy>=1.24
```

---

## Step-by-Step Build Instructions

---

### Step 1 — Sentence Tokenization (`tokenizer.py`)

Use `MosesSentenceSplitter` to split each edition into sentences.

```python
from sacremoses import MosesSentenceSplitter

def split_sentences(text: str, lang: str = "en") -> list[str]:
    splitter = MosesSentenceSplitter(lang)
    sentences = splitter.split(text.splitlines())
    return [s.strip() for s in sentences if s.strip()]
```

- Accept a `--lang` flag (default `en`) to support non-English literary works.
- Preserve original sentence order — this is critical for the alignment algorithm.

---

### Step 2 — Embedding & Similarity Matrix (`embedder.py`)

Embed all sentences from both editions using a Sentence Transformer model, then compute a full pairwise cosine similarity matrix.

```python
import numpy as np
from sentence_transformers import SentenceTransformer

def embed(sentences: list[str], model_name: str = "all-MiniLM-L6-v2") -> np.ndarray:
    model = SentenceTransformer(model_name)
    return model.encode(sentences, convert_to_numpy=True, normalize_embeddings=True)

def similarity_matrix(emb_a: np.ndarray, emb_b: np.ndarray) -> np.ndarray:
    """
    Returns an (M x N) cosine similarity matrix.
    Since embeddings are L2-normalised at encode time, this is a dot product.
    Values are in [-1, 1].
    """
    return emb_a @ emb_b.T
```

- Default model: `all-MiniLM-L6-v2` — fast and effective for literary prose.
- Expose `--model` as a CLI flag so users can swap in a larger model (e.g. `all-mpnet-base-v2`) for better accuracy.
- Always normalise embeddings at encode time so the dot product equals cosine similarity.

---

### Step 3 — DP Alignment with Contiguity Penalty (`align.py`)

This is the core of the tool. Run dynamic programming over the (M x N) similarity matrix. At each step, apply a penalty proportional to the number of sentences skipped since the last match — this strongly favours alignments that proceed in order through both texts.

#### 3a — Score Function

```python
def score(sim: float, gap: int, penalty_weight: float = 0.1) -> float:
    """
    sim:            cosine similarity in [-1, 1] for a candidate sentence pair
    gap:            number of sentences skipped beyond a contiguous step
                    (0 = the two matched sentences are immediately adjacent
                     to the previous matched pair; 1 = one sentence skipped, etc.)
    penalty_weight: cost per skipped sentence (tunable via --penalty CLI flag)

    The penalty is linear: skipping k sentences costs k * penalty_weight.
    At the default of 0.1, skipping 5 sentences costs 0.5 — roughly the
    difference between a strong and a weak semantic match.
    """
    return sim - (penalty_weight * gap)
```

#### 3b — DP Table

```python
import numpy as np

def align(
    sim_matrix: np.ndarray,
    penalty_weight: float = 0.1,
    null_threshold: float = 0.2,
    band: int = 10,
) -> list[tuple[int | None, int | None]]:
    """
    Align sentences from edition A (rows) to edition B (cols).

    Arguments:
        sim_matrix:     (M x N) cosine similarity matrix from embedder.py
        penalty_weight: cost per skipped sentence applied to the score function
        null_threshold: minimum cosine similarity required to form a match;
                        candidate pairs below this are never matched, even if
                        they are the best available option
        band:           DP band width — only sentences within ±band positions
                        of the diagonal are considered as match candidates;
                        reduces complexity from O(M²N²) to O(MN·band²)

    Returns:
        A list of (i, j) index pairs where:
          (i, j)     → sentence i in A aligned to sentence j in B
          (i, None)  → sentence i in A has no match in B
          (None, j)  → sentence j in B has no match in A
    """
    M, N = sim_matrix.shape
    dp = np.full((M + 1, N + 1), -np.inf)
    dp[0, 0] = 0.0
    back = {}

    for i in range(1, M + 1):
        for j in range(1, N + 1):
            best_score = -np.inf
            best_prev = None

            # Option 1: match sentence i-1 (A) with sentence j-1 (B).
            # Search only within the band for efficiency.
            pi_min = max(0, i - band - 1)
            pj_min = max(0, j - band - 1)
            for pi in range(pi_min, i):
                for pj in range(pj_min, j):
                    if dp[pi, pj] == -np.inf:
                        continue
                    # Gap = how many extra sentences were skipped beyond 1-step.
                    gap_a = (i - 1) - pi   # sentences skipped in A
                    gap_b = (j - 1) - pj   # sentences skipped in B
                    gap = max(gap_a, gap_b) - 1
                    gap = max(gap, 0)

                    sim = sim_matrix[i - 1, j - 1]
                    if sim < null_threshold:
                        continue  # never form a match below the threshold

                    candidate = dp[pi, pj] + score(sim, gap, penalty_weight)
                    if candidate > best_score:
                        best_score = candidate
                        best_prev = (pi, pj)

            # Option 2: skip sentence i-1 in A (unmatched sentence in A).
            if dp[i - 1, j] > best_score:
                best_score = dp[i - 1, j]
                best_prev = (i - 1, j)

            # Option 3: skip sentence j-1 in B (unmatched sentence in B).
            if dp[i, j - 1] > best_score:
                best_score = dp[i, j - 1]
                best_prev = (i, j - 1)

            dp[i, j] = best_score
            back[(i, j)] = best_prev

    # Traceback from (M, N) to (0, 0).
    alignment = []
    i, j = M, N
    while i > 0 or j > 0:
        prev = back.get((i, j))
        if prev is None:
            break
        pi, pj = prev
        if pi < i and pj < j:
            alignment.append((i - 1, j - 1))   # matched pair
        elif pi < i:
            alignment.append((i - 1, None))     # A sentence unmatched
        else:
            alignment.append((None, j - 1))     # B sentence unmatched
        i, j = pi, pj

    alignment.reverse()
    return alignment
```

---

### Step 4 — Exporters (`exporters.py`)

Each exporter takes:
- `sentences_a`, `sentences_b`: `list[str]`
- `alignment`: `list[tuple[int | None, int | None]]`
- `meta`: `dict` with edition names, model, penalty weight, threshold, etc.

#### CSV

```python
import csv

def to_csv(path, sentences_a, sentences_b, alignment, meta):
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["idx_a", "sentence_a", "idx_b", "sentence_b"])
        for (i, j) in alignment:
            writer.writerow([
                i if i is not None else "",
                sentences_a[i] if i is not None else "",
                j if j is not None else "",
                sentences_b[j] if j is not None else "",
            ])
```

#### JSON

```python
import json

def to_json(path, sentences_a, sentences_b, alignment, meta):
    rows = [
        {
            "edition_a": {"index": i, "sentence": sentences_a[i] if i is not None else None},
            "edition_b": {"index": j, "sentence": sentences_b[j] if j is not None else None},
        }
        for (i, j) in alignment
    ]
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"meta": meta, "alignment": rows}, f, ensure_ascii=False, indent=2)
```

#### XML

```python
import xml.etree.ElementTree as ET

def to_xml(path, sentences_a, sentences_b, alignment, meta):
    root = ET.Element("alignment", **{k: str(v) for k, v in meta.items()})
    for (i, j) in alignment:
        pair = ET.SubElement(root, "pair")
        a_el = ET.SubElement(pair, "edition_a", index=str(i) if i is not None else "")
        a_el.text = sentences_a[i] if i is not None else ""
        b_el = ET.SubElement(pair, "edition_b", index=str(j) if j is not None else "")
        b_el.text = sentences_b[j] if j is not None else ""
    ET.indent(root)
    ET.ElementTree(root).write(path, encoding="unicode", xml_declaration=True)
```

---

### Step 5 — CLI Entry Point (`aligner.py`)

```python
import click
from tokenizer import split_sentences
from embedder import embed, similarity_matrix
from align import align
from exporters import to_csv, to_json, to_xml

@click.command()
@click.argument("edition_a", type=click.Path(exists=True))
@click.argument("edition_b", type=click.Path(exists=True))
@click.option("--lang",      default="en",                  help="Language for MosesSentenceSplitter")
@click.option("--model",     default="all-MiniLM-L6-v2",    help="Sentence Transformers model name")
@click.option("--penalty",   default=0.1,  type=float,      help="Contiguity penalty weight per skipped sentence")
@click.option("--threshold", default=0.2,  type=float,      help="Minimum cosine similarity to form a match")
@click.option("--band",      default=10,   type=int,        help="DP band width (limits search radius for performance)")
@click.option("--output",    default="alignment.json",      help="Output file path")
@click.option("--format",    "fmt", default="json",         type=click.Choice(["csv", "json", "xml"]))
def main(edition_a, edition_b, lang, model, penalty, threshold, band, output, fmt):
    click.echo("Splitting sentences...")
    sents_a = split_sentences(open(edition_a, encoding="utf-8").read(), lang)
    sents_b = split_sentences(open(edition_b, encoding="utf-8").read(), lang)
    click.echo(f"  Edition A: {len(sents_a)} sentences")
    click.echo(f"  Edition B: {len(sents_b)} sentences")

    click.echo(f"Embedding with '{model}'...")
    emb_a = embed(sents_a, model)
    emb_b = embed(sents_b, model)
    sim = similarity_matrix(emb_a, emb_b)

    click.echo(f"Aligning (penalty={penalty}, threshold={threshold}, band={band})...")
    pairs = align(sim, penalty_weight=penalty, null_threshold=threshold, band=band)

    meta = {
        "edition_a": edition_a, "edition_b": edition_b,
        "model": model, "penalty_weight": penalty,
        "null_threshold": threshold, "band": band,
    }
    exporters = {"csv": to_csv, "json": to_json, "xml": to_xml}
    exporters[fmt](output, sents_a, sents_b, pairs, meta)
    click.echo(f"Done. {len(pairs)} rows written to '{output}'.")

if __name__ == "__main__":
    main()
```

---

## Key Design Decisions to Communicate to Claude Code

1. **Contiguity penalty is linear and tunable.** Each skipped sentence costs `penalty_weight` off the similarity score. At the default of `0.1`, skipping 5 sentences costs `0.5` — comparable to the difference between a strong and a mediocre semantic match. Users working with heavily revised texts should lower the penalty; users working with lightly edited texts can raise it.

2. **The penalty applies to the gap beyond a single step.** A perfectly contiguous match (each edition advances by exactly 1) has `gap = 0` and incurs no penalty. Only sentences skipped *beyond* the immediate next one are penalised.

3. **Null matches are first-class citizens.** The alignment explicitly represents unmatched sentences as `(i, None)` or `(None, j)`. They are never silently dropped. All exporters must handle `None` indices cleanly.

4. **The similarity threshold prevents forced bad matches.** A pair below `--threshold` is never formed even if it is the best available option; the algorithm will instead emit unmatched rows. This avoids pairing completely unrelated sentences just because nothing better was found.

5. **Band constraint makes long texts practical.** The default `--band 10` limits the DP search to sentences within ±10 positions of the diagonal, reducing complexity from O(M²N²) to O(MN·band²). Increase it when aligning texts with large structural reorderings (e.g. drastically revised chapters).

6. **The model is a runtime flag, not hardcoded.** For multilingual texts suggest `paraphrase-multilingual-MiniLM-L12-v2`; for maximum accuracy on English suggest `all-mpnet-base-v2`.

---

## Example Usage

```bash
# Basic run with JSON output
python aligner.py frankenstein_1818.txt frankenstein_1831.txt --output aligned.json

# Stricter contiguity, XML output
python aligner.py paradise_lost_1667.txt paradise_lost_1674.txt \
  --penalty 0.2 --threshold 0.3 --format xml --output aligned.xml

# Larger model, wider band for a heavily revised novel
python aligner.py ulysses_a.txt ulysses_b.txt \
  --model all-mpnet-base-v2 --band 20 --format csv --output ulysses_aligned.csv

# Multilingual alignment (e.g. two French editions)
python aligner.py hugo_ed1.txt hugo_ed2.txt \
  --lang fr --model paraphrase-multilingual-MiniLM-L12-v2 --output hugo_aligned.json
```

---

## Testing Guidance

Ask Claude Code to write tests (`pytest`) covering these cases:

| Test case | Expected behaviour |
|---|---|
| **Identical texts** | Every sentence aligns 1-1, similarity ≈ 1.0, no `None` entries |
| **One sentence inserted in B** | Inserted sentence appears as `(None, j)`; surrounding pairs unaffected |
| **One sentence deleted from B** | Deleted sentence appears as `(i, None)` |
| **High penalty, in-order vs shuffled** | In-order alignment scores higher than a shuffled one when similarities are similar |
| **threshold=1.0** | Almost nothing matches; nearly all rows are null pairs |
| **threshold=0.0** | Every sentence is matched to something |
| **Empty input** | Raises a clear `ValueError` with a descriptive message |
| **Single-sentence inputs** | Returns exactly one aligned pair |
