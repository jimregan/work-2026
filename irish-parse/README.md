# irish-parse

Parse **pre-standard Irish** with [Stanza](https://stanfordnlp.github.io/stanza/)
as the primary parser, cross-checked against [UDPipe](https://ufal.mff.cuni.cz/udpipe),
and get a Markdown report of every disagreement a human needs to resolve.

Runs fully offline inside a devcontainer: both the Stanza `ga` model and the
UDPipe Irish-IDT model are baked into the image.

## Why standardize first?

Stanza and UDPipe are trained on the modern **Irish-IDT** treebank, so they parse
Caighdeán (standard) Irish well but stumble on pre-standard spelling. This tool
therefore standardizes each sentence first via the
[Cadhán intergaelic API](https://cadhan.com) (the same service the
`intergaelic-modernize` skill uses), parses the standardized text, and maps the
result back onto the original surface forms.

For each sentence the output carries:

- `# text` — your original pre-standard sentence
- `# text_standard` — the intergaelic (Caighdeán) form. This is what the
  parsers actually analyze, and the **only** place the standardized form appears
- **FORM** — the original surface form, exactly as written in `# text`
- **LEMMA** (and UPOS/feats/head/deprel) — the parser's output for the
  standardized text
- an original token that standardization split into several words becomes a
  **multiword token**: the range line (`2-4  d'ith`) carries the original
  form; the word rows carry the standard words with their analyses
- an original token **deleted** by standardization gets a placeholder row with
  MISC `Skip=Standard` and an empty analysis (the parsers never saw it) —
  fill it in by hand
- MISC `Align=Check` — the original↔standard alignment was a guess; verify

The alignment is a proper sequence alignment (not a greedy walk), so a single
divergence never desynchronises the rest of the sentence.

> `# text_modern` (a dialectal-but-modern spelling) is **not** generated — that
> is a human editorial decision, added during correction.

## Usage

Inside the devcontainer:

```bash
python parse_irish.py examples/quiggin_sample.txt --out out/quiggin
```

This writes three files:

| File | Contents |
|------|----------|
| `out/quiggin.conllu` | primary parse (Stanza), original forms, `text_standard` |
| `out/quiggin.udpipe.conllu` | UDPipe parse of the **same** tokens |
| `out/quiggin.diff.md` | differences to resolve, as a Markdown report |

Both parsers are fed an identical tokenization (Stanza's), so the diff is a
straight position-by-position comparison of UPOS, lemma, head, and deprel.

### Input formats

- Default: one sentence per non-blank line (plain text). Manual sentence
  splits are respected verbatim — nothing is re-split or merged (Stanza's
  sentence splitter is disabled).
- `--from-conllu`: read the sentences from the `# text` comments of an existing
  CoNLL-U file (e.g. to re-parse a file you are correcting).

### Options

- `--offline` — never call the intergaelic API; use only the on-disk cache
  (`~/.cache/intergaelic`). Sentences with no cached standardization are parsed
  as-is and their tokens marked `Align=NoStandard`.

## Devcontainer

Open the folder in VS Code and "Reopen in Container" (or `devcontainer up`). The
image install pulls:

- Stanza + the `ga` model
- `ufal.udpipe` + `irish-idt-ud-2.5-191206.udpipe` → `/models/irish-idt.udpipe`

Override the UDPipe model with the `UDPIPE_MODEL` environment variable.

## Development

The CoNLL-U handling, alignment, and comparison logic depend only on the Python
standard library and are unit-tested without the parser stacks:

```bash
python -m pytest tests/ -q
```

## Layout

```
parse_irish.py            CLI orchestrator
irish_parse/
  conllu.py               CoNLL-U read/write (stdlib only)
  modernize.py            intergaelic standardization + original alignment
  build.py                rebuild the parse over the original tokens
  stanza_parser.py        Stanza wrapper (primary)
  udpipe_parser.py        UDPipe wrapper (cross-check)
  compare.py              diffing + Markdown report
tests/                    pure-logic unit tests
examples/                 sample pre-standard input
.devcontainer/            Dockerfile + devcontainer.json (offline models)
```
