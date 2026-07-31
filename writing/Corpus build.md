# corpus-build: status report via a Riksdag toy corpus

## What this is

`corpus-build` is a library for modelling a speech corpus as a **graph of
versioned, content-addressed artifacts** connected by registered,
versioned transformations — rather than a directory tree or a fixed linear
pipeline. The point of the graph shape is that multiple, differently
reliable interpretations of the same source material can coexist (two
ASR passes, a corrected version and an uncorrected one, an official
transcript and what was actually said) without the library forcing a
single "correct" answer. Reproducibility is closer in spirit to a build
system (Make/Bazel) than to a one-shot processing script: every artifact
knows exactly what produced it and from what inputs.

We're validating this design against a real, concrete case — Riksdag
(Swedish parliament) session recordings and their official API — rather
than building the whole thing speculatively and hoping it holds up later.

## How to read this report

For each stage of the toy pipeline below, I've tagged what exists as one
of:

- **Core mechanism, built & tested** — general library machinery, would
  apply to any corpus, has real tests (often against real Riksdag data,
  not synthetic fixtures).
- **Corpus-specific, built** — Riksdag/Swedish-specific logic that plugs
  into the core mechanism; not the library's job to generalize.
- **Designed, not built** — a decision is written down (usually as an
  ADR), no code exists yet.
- **Not yet built or designed** — an acknowledged gap.

The distinction that matters for your report: gaps in the first category
are things *the library itself* is missing and would block any corpus, not
just this one. Gaps in the second category are Riksdag/Swedish-specific
work that was always going to be needed regardless of how good the
library gets.

## Walking through the toy pipeline

### Stage 0 — Importing already-collected API dumps

Given a directory of previously-downloaded Riksdag API JSON responses
(one file per debate), each file becomes a specification artifact
(the reconstructed original API request) plus a response artifact (the
JSON body), linked by a `batch_import` execution rather than a `fetch`
execution — so provenance alone tells you "this was imported," no new
field needed. Acquisition time comes from the file's own mtime or a
batch-level override, never from the moment the import runs.

- **Core mechanism, built & tested**: `batch_import` reuses the same
  specification/response artifact shape a live fetch would produce.
  Tested against real archived files, including a case that caught a
  real bug (acquisition time silently not being set when given a bare
  date instead of a datetime).
- **Corpus-specific, built**: the actual API endpoint template
  (`https://data.riksdagen.se/api/mhs-vodapi?<dokid>`) and how a
  filename maps to a `dokid`. Deliberately kept as a caller-supplied
  function, not hardcoded into the library.

### Stage 1 — Extracting the video link and re-fetching

Reads an already-obtained API response, pulls out the video's download
URL, and feeds it through the ordinary fetch machinery — unmodified.

- **Core mechanism, built & tested**: the pattern itself (read an
  artifact, derive a new specification, feed it through existing fetch)
  reuses `fetch()`/`create_source_ref()` without any changes.
- **Corpus-specific, built**: the JSON field path
  (`videodata[0].streams.files[0].downloadfileurl`) is specific to this
  version of the Riksdag API and lives in its own small module, not
  mixed into the generic fetch code.
- Verified for real, not just in tests: imported one archived document,
  extracted its video URL, and fetched the actual video from Riksdagen's
  live CDN (2.5MB, real network round trip).

### Stage 2 — CTM alignment as a black-box transformation

Aligns a word-timed ASR hypothesis against reference text using a
Smith-Waterman aligner, producing a correspondence artifact.

- **Core mechanism, built & tested**: the aligner itself is generic (it
  aligns any two word sequences, no Riksdag knowledge). Reproduces a
  known reference implementation's test case exactly, and reproduces the
  same edit classification as a real worked example from this corpus.
  The wrapping pattern — alignment output becomes a correspondence
  artifact whose metadata records *how* the match was made, so different
  consumers can filter by trust level later — is itself general.
- **Corpus-specific, not yet built**: everything downstream that decides
  what an "error" actually means — a droppable discourse particle ("ju"),
  a list read as "A och B och C" instead of "A, B och C", a genuine
  mistranscription, a disfluency the official transcript never recorded.
  These are Swedish-language, Riksdag-specific judgment calls and are
  explicitly scoped as separate, later transformations, not part of the
  aligner itself.

### Stage 3 — Registering these as real, tracked transformations

This is the biggest gap right now, and worth flagging clearly.

- **Core mechanism, built & tested**: the registry can register a
  transformation *without* a container image ("code-versioned" — identity
  rests on the git commit instead) — added specifically because text
  transformations like the three above have no environment-sensitive
  runtime worth Docker-pinning, unlike acoustic/ML transformations.
- **Not yet built**: nobody has actually called this registration for
  `batch_import`/extraction/the aligner, and no `Execution` record has
  ever been created linking real artifacts. Every result described above
  came from calling plain Python functions directly and inspecting the
  return values — **nothing has been written to a real storage backend
  for this corpus yet.** There's no queryable, persisted Riksdag corpus
  on disk at this point, only proven, tested building blocks.

### Stage 4 — Content storage layout

- **Designed, not built**: agreed convention is
  `corpus/<acquisition-date>/<hash-shard>/<original-filename>`, shard
  depth configurable for corpus size, deliberately kept compatible with
  tools like `git-lfs` without needing to know about them. Never
  implemented — imported content currently just points back at the
  original archive files in place.
- This is core-mechanism work — any corpus benefits from a real content
  layout — even though Riksdag's scale (thousands of files) is what
  surfaced the need for it.

### Stage 5 — Speech segmentation (deferred on purpose)

- **Designed, not built**: needs two forms — a collection-level artifact
  (the whole ordered set of speeches in a debate) and individual
  per-speech artifacts, not a choice between them. A real, working prior
  implementation of this exists in an older, separate codebase
  (`sync_asr`), not yet ported.
- The general shape (a segmentation is a first-class artifact; multiple
  partitions of the same source can coexist) is already core to the
  model. The "speakers array" specifically is a Riksdag API concept.

### Stage 6 — Two ASR passes, reconciled before judgment

- **Designed, not built, and blocked on real tooling**: word-level ASR
  and phonetic-level ASR need to be reconciled against each other by time
  overlap *before* either is compared to the reference — a worked example
  showed a case where the word-level pass alone would have misdiagnosed
  a disfluency as a transcription error. No ASR has actually been run
  inside corpus-build; this needs real model/Docker decisions (ASR is
  exactly the acoustic case that *does* need image-level pinning, unlike
  everything built so far).

### Stage 7 — Downstream judgment passes

- **Designed, not built**: paraphrase-acceptance, correction, and
  disfluency-splitting as three separate, independently versioned
  transformations, each recording *how* it decided something in the
  correspondence artifact's metadata rather than silently picking a
  winner. The mechanism is core; the actual rules (a correction lexicon,
  the "ju" heuristic) are entirely Riksdag/Swedish-specific.

### Stage 8 — Staleness / rebuild detection

- **Not started, intentionally.** Its own prerequisite — whether a
  collection is a first-class artifact or something reconstructed by
  grouping — is still an open question, and staleness work is explicitly
  blocked on settling that first.

## Summary table

| Stage | Core mechanism | Corpus-specific work |
|---|---|---|
| 0. Import | Built & tested | Built (API template) |
| 1. Extract + fetch | Built & tested | Built (field path) |
| 2. CTM alignment | Built & tested | Not built (judgment rules) |
| 3. Registration/persistence | Built, **not applied** | — |
| 4. Content layout | Designed only | — |
| 5. Speech segmentation | Model supports it | Designed, not ported |
| 6. Two-pass ASR | Designed only | Not started (needs tooling) |
| 7. Judgment passes | Designed only | Not started |
| 8. Staleness | Not started | Blocked on prerequisite |

## What's genuinely missing from the library (not Riksdag's fault)

- Nothing has been persisted to a real storage backend for this corpus —
  the single biggest gap between "the mechanism works" and "there's an
  actual corpus."
- Content storage layout — designed, not implemented.
- A prior design (transformation validity windows tied to date ranges,
  for handling API versions changing over time) is written down as an
  ADR but its fields were never added to the model.
- No non-Docker execution path exists yet for code-versioned
  transformations — running one still means calling the plain function
  and building the `Execution` record by hand.
- No way to query artifacts/correspondences by their reliability metadata
  yet (the query surface only walks provenance ancestry today) — this was
  flagged as expected future work, not an oversight.
- Collection modelling is unresolved, which blocks all of staleness
  detection.

## What's corpus-specific to Riksdag (would be needed regardless of how complete the library gets)

- The API endpoint template and how it maps to identifiers.
- JSON field-path parsing for this specific API version.
- Speaker/session parsing, a correction lexicon, paraphrase-acceptance
  heuristics — all Swedish-language judgment calls.
- Real ASR models and however they end up packaged.
- Speaker/committee metadata enrichment (party, term of office, etc.).

## Decision log (for traceability)

1. Per-layer typed identity — no global "same thing" relation across layers.
2. Metadata-primary storage — content is files, never database blobs.
3. Content-addressed identity, with an exception for pre-fetch intake nodes.
4. Split `Transformation` (identity) from `Execution` (provenance edge).
5. Image resolution + Docker execution + non-determinism policy.
6. Acquisition transformations (fetch/classify/WARC) are plain, tested
   functions, not yet packaged into containers.
7. Transformation versioning via git ancestry + date-bounded validity —
   **designed, not implemented.**
8. Correspondence reliability is queryable, never resolved by the
   pipeline — required no code change, just made an implicit principle
   explicit.
9. Transformations can be code-versioned (no container) as well as
   containerized — implemented; this is what stages 0–2 above actually
   run on.