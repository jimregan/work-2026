	# Index

Top-level map of active work. Directories are relative to `work-2026/`.

---

## Writing — active papers

### Derived Text Formats (workshop paper)
`derived-text/paper.qmd`
Highest-priority paper. Method is largely complete; focus is on schema spec, pipeline scripts, Docker, and small sample dataset.
See `derived-text/TODO.md` for phased plan (Phase 0–7).
Key decision: UD format chosen partly for robustness under upstream HTML change (not yet written up).

### Phonetic Corpus — Swedish parliament (journal)
`pronunciation-data/` · template: Interspeech 2024
Previously rejected from journals. Missing: evaluation section, results table, manual validation study.
Needs reframing from "project report" to "validated resource paper."
Paper structure: `political/paper.qmd`

### Multilingual G2P
`multilingual-g2p/paper.qmd`
**Blocked**: accent/stress markers were stripped; need to restore and re-run full pipeline before results are publishable. Accent-stripped results should be kept as a comparison point.
Kaggle notebooks and local notebooks documented in `multilingual-g2p/TODO.md`.

### Spoken Sentence Transformers — Odyssey 2026
`spoken-sentence-transformers-odyssey/od2026_latex_template/`
LaTeX template is in place. Eval plan in `eval-plan.md`, training plan in `training-plan.md`.
`utterance-map.json` in `librivox-multispeaker/` still needs to be committed to the dataset repo.

---

## Projects with CLAUDE/AGENTS instructions

These have active agent instructions — pick up and continue:

| Dir | Notes |
|-----|-------|
| `dysfluent-wfst/` | CLAUDE.md present |
| `editions-alignment/` | CLAUDE.md present |
| `image-region-ocr/` | CLAUDE.md + README |
| `irish-lemmatisation/` | AGENTS.md |
| `kelly-hu/` | AGENTS.md |
| `librivox-matching/` | CLAUDE.md |
| `librivox-multispeaker/` | AGENTS.md |
| `dieck22/` | AGENT.md |

---

## Data / corpus directories (not papers yet)

| Dir | Contents |
|-----|----------|
| `arctic/` | CMU Arctic speaker directories |
| `etymwn/` | etymwn.tsv (etymology wordnet) |
| `etexts/` | Poetry text files |
| `hungarian/` | ChatGPT reel notes (Jan 2026) |
| `mo-sceal-fein/` | w2v.json alignment files |
| `an_tiriseoir/` | OCR'd An tIriseoir PDF + XML |
| `open_speech/` | OSR JSONL files |
| `mmconv/` | vibevoice subdir |
| `pronunciation-data/` | IEEEtran + checklist files |
| `phonetic-corpus/` | NOTES.md, eval checklist |
| `accents-gmu-native/` | GMU accent data |
| `editions-alignment/` | Align scripts |
| `ocr-alignment/` | OCR alignment scripts |
| `timit-phonetic-words/` | Notebooks + patches |
| `wolne-lektury/` | Polish texts notebooks |

---

## Loose files to sort

- `areas.txt`, `msf-omni.txt` — unknown provenance
- `r1`, `r2`, `raw` — Halloween Bangs transcript data (to revisit)
- `dubliners-03.json`, `english_native.json`, `halloween_bangs_si_128kb.json` — audio/transcript data
- `0001-*.patch` — patches for unidentified repos (check if still needed)
- `autopst-codebase-notes.md`, `nettt.md`, `multi-source-scorer-instructions.md` — notes to sort
- `to-sort/` — misc links and qmd files
