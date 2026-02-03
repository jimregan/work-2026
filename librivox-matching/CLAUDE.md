# LibriVox etext matching

## Goal

Align LibriVox audiobook recordings to their source etexts (typically from
Project Gutenberg) to produce word-level timestamps. This uses a two-stage
pipeline:

1. **Chunk-level alignment**: match VibeVoice speaker-diarised transcript
   chunks to the corresponding passages in the etext using n-gram matching
   (same approach as kblabb-notes).
2. **Word-level alignment**: within each matched chunk, use CTC segmentation
   with wav2vec2 to get per-word timestamps.

## Input formats

### VibeVoice output (JSONL-ish)

```json
{"Start":7.22,"End":10.48,"Speaker":0,"Content":"Yeah, you're, I mean, your place looks amazing."},
{"Start":10.48,"End":12.68,"Speaker":1,"Content":"Yeah, thank you."},
```

Each line has `Start`/`End` (seconds), `Speaker` (int), and `Content` (text).

### Etext

Plain text from Project Gutenberg (or similar). Includes Gutenberg/LibriVox
boilerplate at the start and end that should be preserved (tagged as
boilerplate, not discarded).

## Pipeline

### Stage 1: Chunk-to-etext matching

1. **Parse VibeVoice output** into a list of chunks with start/end times and
   text.
2. **Merge chunks** that would otherwise split a sentence. Use punctuation
   (`.`, `!`, `?`) as sentence boundaries — if a chunk does not end with
   sentence-ending punctuation, merge it with the following chunk.
3. **Normalise** both the merged chunk text and the etext for matching:
   - Lowercase, remove punctuation, NFKC normalise, collapse whitespace.
   - For English, number verbalisation is less critical than for Swedish
     (the kblabb case), but handle common cases (chapter numbers, dates).
4. **N-gram contiguous matching** (from kblabb-notes approach):
   - Build n-grams (1..6) of both texts.
   - Boolean match vectors + weighted convolution scoring.
   - Find contiguous matching regions, filter short matches, bridge small
     gaps.
   - Run in both directions (chunk→etext and etext→chunk).
5. **Fuzzy matching** as fallback using `rapidfuzz.fuzz.partial_ratio_alignment`.
6. **Log mismatches**: where VibeVoice text differs from the etext (ASR
   errors, OCR errors in the etext, genuine textual variants), log the
   diff for later manual correction. Output should include:
   - The chunk timestamp range
   - The VibeVoice text
   - The matched etext passage
   - A word-level diff (e.g. from `difflib`)

### Stage 2: Word-level CTC segmentation

For each matched chunk:

1. **Extract audio** for the chunk's time range from the audiobook file.
2. **Run CTC segmentation** using `ctc_segmentation` with a wav2vec2 model
   (`jonatasgrosman/wav2vec2-large-xlsr-53-english` or similar).
3. **Use the etext words** (not the VibeVoice ASR output) as the ground
   truth transcript for alignment — the etext is the authoritative text.
   Where the etext and VibeVoice text diverge, fall back to the VibeVoice
   text for that segment (since it at least matches what was spoken).
4. **Output per-word timestamps** with confidence scores.

### Boilerplate handling

LibriVox recordings typically start and end with reader-spoken boilerplate
(title, author, chapter info, licence text). This text won't appear in the
Gutenberg etext. Strategy:

- Detect boilerplate chunks by failed etext matching (low match score).
- Keep them in the output, tagged as `"boilerplate": true`.
- Optionally match against a set of known LibriVox boilerplate templates.

## Output format

```json
{
  "source_etext": "pg12345.txt",
  "audio_file": "chapter01.mp3",
  "segments": [
    {
      "start": 7.22,
      "end": 10.48,
      "etext": "your place looks amazing",
      "vibevoice": "Yeah, you're, I mean, your place looks amazing.",
      "words": [
        {"text": "your", "start": 7.85, "end": 8.12, "conf": 0.95},
        {"text": "place", "start": 8.12, "end": 8.45, "conf": 0.91}
      ],
      "boilerplate": false
    }
  ],
  "mismatches": [
    {
      "chunk_start": 7.22,
      "chunk_end": 10.48,
      "vibevoice": "Yeah, you're, I mean, your place looks amazing.",
      "etext": "your place looks amazing",
      "diff": "- Yeah, you're, I mean, your place looks amazing.\n+ your place looks amazing"
    }
  ]
}
```

## Dependencies

- `ctc_segmentation`
- `transformers` (wav2vec2)
- `torch`
- `numpy`
- `rapidfuzz`
- `nltk`
- `soundfile` or `librosa` (audio loading)

## Notes

- The n-gram matching logic is shared with the kblabb-notes project. Consider
  extracting it into a shared module if both projects end up in active use.
- The kblabb approach was designed for Swedish parliamentary transcripts. For
  English audiobooks the text normalisation is simpler but the texts are much
  longer, so performance of the matching step may matter.
- CTC segmentation works best with clean audio and accurate transcripts. For
  segments where the etext diverges significantly from the spoken audio (e.g.
  the reader ad-libs or skips text), confidence scores will be low — use these
  to flag segments for review.
- The `align_with_transcript` function takes the etext words as ground truth;
  `get_word_timestamps` uses the model's own predictions. Prefer
  `align_with_transcript` with etext words for accuracy, falling back to
  `get_word_timestamps` for boilerplate/unmatched segments.
