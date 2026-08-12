# Milestone 4 — Concrete ingress and transform pipelines

Status: current.

Read `AGENTS.md` first. Everything there still applies. This file adds the
scope and decisions specific to this milestone.

## Goal

Prove the model against real material before milestone 3 (staleness) is
attempted. `03-staleness.md` names two unresolved prerequisites — collection
modelling, and non-determinism handling — and both are the kind of thing
easy to get wrong in the abstract and obvious once real data is flowing
through the machinery milestone 2 already built. This milestone builds two
small, concrete pipelines against real Riksdag (Swedish parliament) material
to surface what's actually missing, rather than speculating further.

## Concrete material

- `~/Playing/rdapi/api_output/` — ~9,986 previously-downloaded JSON
  documents from an old version of the Riksdag video API, one file per
  debate (`dokid`, e.g. `H001AU10`), no file extension. Confirmed structure
  includes `videodata[0].streams.files[0].downloadfileurl` (the video file
  URL), `videodata[0].debatedate`, and per-speaker segmentation
  (`speakers[].start`/`duration`/`text`/`party`) — useful downstream as
  correspondence/segmentation material later, out of scope for this
  milestone itself.
- A small, currently-unspecified set of live debates for the incremental
  test, fetched from whatever the current Riksdag API looks like today —
  this may differ in shape from the archived dump above, which is exactly
  what makes it a useful probe.

## Scope

### Batch import

- A `batch_import` transformation, registered like any other, siblings
  `fetch` rather than replacing it: given a directory of pre-collected JSON
  responses, it produces a specification artifact and a response artifact
  per document, linked by a `batch_import` execution rather than a `fetch`
  execution. Provenance alone then distinguishes "live-fetched" from
  "imported" — no new model field needed (see prior conversation; this is
  the reasoning already settled, not reopened here).
- Acquisition time on each imported response is sourced from the file's
  mtime, or a single batch-level override date, never from "now" (the
  `batch_import` execution's own processing time is separately "now," as
  usual).
- **Open: what is the specification's content for an imported document?**
  A live fetch's specification records a URL independently of its artifact
  UUID. An imported document has no URL being fetched right now — the natural
  candidate is
  the *original* API request that would have produced this `dokid`, if
  recoverable, so the specification stays a genuine "what was asked for"
  artifact rather than a re-labelled response. If that isn't recoverable
  for this dataset, the fallback is unresolved and should be decided before
  building rather than silently defaulting to something. Surface this
  rather than picking.

### Extraction and re-fetch

- A separate, registered transformation reads an imported (or live) API
  response and produces a new specification artifact whose content is the
  extracted video download URL. This is deliberately not classification —
  classification judges a response's outcome (obtained/gone/retry);
  extraction manufactures a follow-on specification for a different fetch.
  Confirm this split matches intent before building; it's a new
  transformation shape not explicitly discussed yet.
- That specification is then fed through the **existing, unmodified**
  `fetch` transformation from milestone 2 to retrieve video bytes. No
  changes to `fetch` itself are expected — if this milestone finds it needs
  any, that's a signal worth stopping on, not routing around.

### Transformation versioning (proving ADR 0007)

- Register the batch-import + extraction pipeline as a versioned family per
  [ADR 0007](docs/decisions/0007-transformation-versioning-and-date-windows.md):
  `module_path`, acquisition-time `valid_from`/`valid_until` on
  `Transformation`, registration-time supersession via git ancestry. This
  milestone is where 0007 gets implemented, not just designed.
- If the current live API turns out to have a different shape than the
  archived dump (expected, given it prompted this whole design thread), the
  live-fetch pipeline is a second family member, and registering it is the
  first real exercise of the supersession/window-closing behaviour.

### Content storage layout

- Implement the filesystem content layout agreed conversationally (not an
  ADR): `corpus/<acquisition-date>/<hash-shard>/<original-filename>`, shard
  depth configurable. This is the `content_locator` convention referenced
  by artifacts, independent of the metadata-sidecar layout already built in
  `storage/filesystem.py`. Nothing about `StorageBackend` changes; this is
  purely how content bytes get laid out by whatever writes them.

### Correspondence

- At least one concrete correspondence artifact asserting equivalence
  between an old-API-shaped document and its current-API-shaped
  counterpart (if the same debate is reachable both ways), exercising the
  asserted-correspondence mechanism from `AGENTS.md` on a real pair rather
  than a hypothetical one.

## Deliberately excluded

Do not build these, and do not add abstractions in anticipation of them:

- **Bulk-processing all ~9,986 archived documents.** This milestone proves
  the pipeline on a small representative sample. Running it over the whole
  archive is an operational task for later, not a design-validation task.
- **Any ASR, segmentation, or transcription.** Acquisition and registration
  plumbing only.
- **Automatic transformation dispatch by date.** ADR 0007 explicitly defers
  this; validity windows are a run-time guard rail, not a scheduler.
- **Retroactive rebuild-scoping** (patch a transformation, narrow the
  rebuild by document time). Parked for milestone 3 in ADR 0007; a
  different mechanism despite similar vocabulary.
- **Solving collection modelling.** The incremental test is meant to
  surface whether the existing model is sufficient, not to design a
  `Collection` artifact type pre-emptively.
- **Git-hook-triggered auto-registration.** Worth doing eventually,
  workflow tooling rather than model work; a stretch goal for this
  milestone, not a requirement of it.

## Tests

- A `batch_import` execution produces a specification/response pair per
  imported document; acquisition time comes from file mtime or the
  batch-level override, never from execution processing time.
- The extraction transformation produces a new UUID-identified specification
  artifact whose content is the extracted URL, distinct from the API
  response artifact it was read from.
- That specification round-trips through the **unmodified** `fetch`
  transformation from milestone 2.
- Registering a second family member under ADR 0007's mechanism correctly
  closes the first member's `valid_until` at the new registration date,
  unless already set explicitly.
- Content bytes land at `corpus/<acquisition-date>/<hash-shard>/<original-filename>`
  and are readable by ordinary tools (no library round-trip required to
  read them).
- At least one correspondence artifact round-trips through storage linking
  an old- and new-shaped document.

## Forward notes (not this milestone, recorded so they aren't lost)

Speech segmentation, deferred from this milestone's scope (see "Deliberately
excluded"), has more shape already decided than "later" implies:

- It needs **two forms**: a collection-level artifact (the whole ordered
  set of speeches for a debate) and individual per-speech artifacts. Not a
  choice between the two — both are needed.
- An individual speech, once split out, feeds existing Kaldi-derived CTM
  alignment code against a word-timed ASR result, to align the two.
- Empirically (every example observed so far), the API's start/duration
  window for a speech **entirely encloses** the actual speech — the
  boundary is conservative, not exact. Alignment has to account for that,
  not assume the stated window is the speech's true extent.
- `~/Playing/sync_asr` (a separate, older repo, same author) has real,
  working prior implementations of most of this: `riksdag_align.py`
  (collection+per-speech CTM partitioning, and driving the Smith-Waterman
  aligner against Riksdag reference text), Kaldi's own
  `segment_ctm_edits[_mild].py` (the stage after alignment that derives
  segment boundaries), and the shared `elements.py`/`ctm.py`/`ctm_edit.py`
  data model those depend on. Reuse candidates, not yet ported.
  `~/Playing/sync_asr/wav2vec2-riksdag-api-alignments/` is a published
  dump of ~750 prior ctmedit alignment files — a candidate reference/
  validation dataset, but not yet vetted for quality; pending manual
  check before relying on it.
- **The judgment layer is three inputs, not two, and at least three
  distinct passes.** Worked example: word-level ASR produced one garbled
  token (`tanledningen`) against reference `anledningen`, which looks like
  an obvious substitution error — but a second, phonetic-level ASR pass
  shows two separate chunks in that time span (`atəm`, then a clean
  `anleːnɪŋən`), meaning the word-level pass actually merged a disfluency
  with a correctly-spoken real word into one bad token. The reference
  transcript, being cleaned, never included the disfluency at all. This
  could not be diagnosed from the word-level CTM and reference alone; it
  needed the phonetic pass's finer time resolution.
  - Reconciling the two ASR passes against each other (by exact anchor and
    time-overlap — `time_aligner.py` already does this, independent of
    Riksdag/CTM specifics) has to happen **before** either is compared to
    the reference.
  - At least three separate, versioned reclassification passes sit after
    that: **paraphrase-acceptance** (trust hyp over ref — droppable
    particles like "ju", list-conjunction variants), **correction** (trust
    ref over hyp — genuine mistranscriptions, what `corrections.py`'s
    static lexicon already does, but as a registered transformation rather
    than a flat table), and **disfluency-splitting** (neither hyp nor ref
    is simply "right" — the token needs to be split before either
    judgment applies).
  - This is the concrete validation that the model's "provenance-
    preserving, separable transformations, access to earlier stages"
    thesis holds up against a genuinely hard real case — confirmed as
    exactly the kind of thing milestone 4 was meant to surface.

**KBLab is a real integration partner, not a hypothetical one.** They have
already packaged a subset of Riksdag speeches with mapping to known
metadata, and have already run Whisper over their segments. Re-running
Whisper on that subset would waste resources we don't need to spend —
their ASR output should be ingestable as its own artifact with its own
provenance (an externally-sourced ASR pass, same shape as `batch_import`
already distinguishes "imported" from "fetched" — no new model concept),
and reconciled against our own passes exactly per ADR 0008's correspondence
mechanism rather than either being treated as authoritative. Not scoped or
built; recorded so the integration isn't designed from scratch later
without this constraint in view.

**Known implementation gap, not a design gap.** `Transformation`'s
docstring and `02-transformation-registry.md` both describe recomposing
the stored config hierarchy and checking it reproduces the stored
composed config, surfacing a mismatch as a reportable inconsistency. No
code implements this check anywhere. The decision was always made; it was
just never built.

**Topology variety validated against a second, non-Riksdag source.** A
one-URL "corpus" (a single Wikimedia/Wiktionary pronunciation recording,
plus a `dummy_asr` stub transformation standing in for a real ASR pass)
was registered and run through the exact same mechanism as Riksdag —
`register_transformation`, `FunctionRunner`, a real `FilesystemBackend` —
and persisted into the *same* store, alongside Riksdag's artifacts. No new
orchestration code was needed. Concretely proved:
- The `fetch` transformation registered for this source resolved to the
  *identical* `TransformationId` already registered for Riksdag (same
  module, same config) — "one transformation, many executions across
  sources" is real, not just a claim.
- A hand-given URL seed needs no registered "search" transformation — a
  bare `SourceRef` artifact with `produced_by_execution=None` is fully
  representable, same as any other unproduced root artifact. This is a
  working default for the "basic" seed case from earlier design
  discussion, not a resolution of it — a real search adapter (open
  decision 5 below) is still undesigned.
- A stub transformation (`dummy_asr`) is a real, versioned
  `Transformation` like any other, not a shortcut — its output is
  independently identified and self-describing (`{"stub": true, "asr_engine":
  "dummy", "word_timed": false}`), so nothing downstream could mistake it
  for a real ASR pass without reading its own metadata.
- `query.py`'s `provenance_of` walked the new source's graph correctly
  with zero changes — confirmed the generic query surface, not just the
  write path, needed no per-source special-casing.

Real, unresolved finding from this: Wikimedia's edge rejected the fetch
with no `User-Agent` *and* with a properly-identifying custom one
(`corpus-build-test/0.1 (contact: ...)`), both 403, but accepted a
generic browser-shaped one (`Mozilla/5.0`) — confirmed independently with
`curl`. Worked around for this demo only via `fetch()`'s existing
`client=` parameter; `fetch()` itself was not changed. See open decision 6.

## Open decisions

Surface these; do not settle them unilaterally.

1. What content an imported document's specification artifact records if the
   original API request isn't recoverable for this dataset. ADR 0012 removes
   this as an identity question but not as an evidence-modelling question.
2. Whether extraction is its own transformation kind (as scoped above) or
   should be folded into classification after all.
3. Which live debates to use for the incremental test, and what the
   current Riksdag API's shape actually looks like — unknown until checked.
4. Hash-shard depth/algorithm for the content storage layout — a default
   should be proposed when building, not fixed here.
5. What a real search adapter looks like beyond a hand-given URL list —
   the "basic" seed case is validated (a bare, execution-less `SourceRef`),
   the "typical" case (a parameterized adapter, e.g. a YouTube channel
   scraper, concrete-vs-spec per the `CodeVersionSource`/`StorageBackend`
   pattern) is still undesigned.

Resolved:

- ~~Whether `fetch()` should carry a default `User-Agent`~~ — no default,
  configurable instead: `fetch()`/`walk_redirect_chain()` now take an
  explicit `headers` parameter (this is a scraper-based system; different
  sources will need different headers, and picking one default UA behaviour
  for all of them was never going to be right). Applied only when the call
  builds its own client — a caller supplying `client=` sets headers on it
  directly. Where a specific transformation's config maps to headers is
  still the calling adapter's job, not wired automatically.
