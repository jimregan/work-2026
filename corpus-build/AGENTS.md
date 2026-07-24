# AGENTS.md

Guidance for coding agents working on this repository.

This file holds the parts that do not change between milestones. The current
milestone — scope, out-of-scope, and its own decisions — lives in
`docs/milestones/`. **Read both before starting work.**

Current milestone: `docs/milestones/02-transformation-registry.md`
Completed: `docs/milestones/01-data-model.md`
Planned, do not implement: `docs/milestones/03-staleness.md`

## What this is

A Python library and CLI for modelling a speech corpus as a **graph of versioned
artifacts** connected by registered transformations, rather than as a directory
tree or a linear pipeline.

One-sentence thesis:

> A corpus is a graph of versioned artifacts with layer-specific identities,
> provenance-preserving transformations, and task-dependent views, allowing
> multiple valid interpretations of the same source to coexist while supporting
> incremental, reproducible processing.

If a proposed change makes that sentence less true, it is the wrong change.

## Design constraints

These come from the architecture notes and are not negotiable without an
explicit decision recorded in `docs/decisions/`.

**Identity is layer-specific.** There is no global "same thing" relation. Two
radio captures of one broadcast are distinct at the acquisition layer and may be
one object at the ASR layer. Do not add a global canonical ID or a
`deduplicate()` that collapses across layers.

**Artifacts are generic.** Recordings, transcripts, alignments, segmentations,
metadata, correspondence mappings, and temporary representations are all
artifacts. Do not privilege audio in the core model. Do not hardcode a
processing unit.

**Provenance is a DAG.** Artifacts have zero or more parents. Multi-parent is
the normal case (validated speech from transcript boundaries plus acoustic
boundaries), not an edge case.

**Transformations are objects, not functions.** A transformation record carries
implementation reference, container image (immutable digest preferred over tag),
parameters, and execution metadata. Git identifies source; the image identifies
executable behaviour. The model must be able to represent a transformation that
has not been run.

**Persistence is a property, not a position.** Durable versus ephemeral depends
on value and cost of reconstruction, not on where the artifact sits in a
pipeline. A 16 kHz WAV materialization is ephemeral even though it is
"downstream". Do not infer persistence from graph depth.

**Three distinct times.** Acquisition time, document time, processing time. Keep
them separate fields. Late-arriving material is processed according to document
time. Never collapse these into a single `timestamp`.

**Correspondence is an artifact.** The mapping between two segmentations, or
between two recordings of one broadcast, is itself a first-class artifact with
its own provenance and its own version history. It can improve independently of
either side.

**Multiple partitions coexist.** Transcript-based, acoustic, and schedule-based
boundaries over one source are all valid simultaneously. No partition replaces
another.

**Identifiers are content hashes.** Every artifact is addressed by a hash of its
own content. There is no separate ID space, no autoincrement, no UUID. Two
transformations that happen to produce identical bytes converge on one node with
two provenance parents — this is dedup working as intended, not a collision to
disambiguate.

**A fetch specification is an artifact whose content is the URL.** Hashing it is
content addressing applied to a specification rather than to bytes, so the
constraint above holds without exception. Acquisition is not a special case in
the identity scheme.

**Fetching is specification → response → classification.** A fetch transformation
takes a specification and produces a *response* artifact. A response is whatever
came back: bytes, a 404, a 503 with retry-after, a redirect. Failure is an
ordinary outcome of a network transaction, not an absence of one, and it
produces an artifact like any other. A separate classification transformation
then reads the response and decides what kind of outcome it was.

Consequences that must not be optimised away:

- Classification policy is where judgement lives and where it will be wrong
  first. It is a registered, versioned transformation, so a response can be
  reclassified later without refetching. Responses are therefore durable at
  least until classification is stable.
- The state of a specification — pending, obtained, gone — is *derived* by
  looking at the classifications that reference it. Do not add a mutable status
  field to a specification; it would break content addressing.
- A referral classifies to a new specification, giving
  specification → response → classification → specification'. Redirect
  convergence, where several specifications reach one target, is ordinary
  multi-parent structure in the DAG. It does not assert that the originating
  specifications were ever the same object — unrelated URLs share canonical
  landing pages and error pages routinely.

**Asserted correspondence is distinct from established identity.** Following a
redirect establishes that a target was reached. Claiming that an artifact
obtained from an old URL is the same object as one at a new URL — as when a
whole collection relocates and members would otherwise be refetched blind — is a
judgement, usually human. Represent it as a correspondence artifact with its own
provenance recording it as asserted, so it is versioned and can be wrong. Never
infer it silently to avoid work.

## Storage backends

Pluggable from day one. The core model must not import any specific backend.

- Define the backend contract as an explicit protocol/ABC.
- Implement at least two backends before declaring the interface stable. A
  filesystem-conventions backend and a SQLite backend are the expected pair.
- Any behaviour that only works in one backend is a leak in the interface —
  report it rather than special-casing.
- Backend-specific tests must run against the shared conformance suite.

## Working style

- Minimal, targeted changes. Do not refactor adjacent code that was not part of
  the task. Do not rewrite a module when an edit will do.
- Present options rather than picking silently when there is a real design fork.
  State the trade-off in one or two sentences and let the choice be made.
- Do not assert anything about files, configs, or environments you have not
  read. Read first.
- When a requirement is ambiguous, ask before implementing. A wrong
  implementation is more expensive than a question.
- Prefer explicit over clever. This is research infrastructure that other people
  will need to reason about.

## Code conventions

- Python, type-annotated, `mypy --strict` clean in the core model.
- Core model has no dependencies beyond the standard library. Backends and
  execution machinery may have their own.
- Dataclasses (frozen where the object is a value) for model types.
- Tests with pytest. Every provenance invariant gets a test: acyclicity,
  multi-parent, layer-specific identity, ephemeral regeneration.
- Property-based tests (hypothesis) for graph invariants are welcome.

## Anti-patterns

Concrete things that have been considered and rejected:

- A single canonical ID per "real world thing"
- A linear `Pipeline` or `Stage` class as the primary abstraction
- Persistence flags derived from position in the graph
- A single `timestamp` field
- Correspondence stored as a plain dict or as a field on one of the two sides
- Backend logic imported into the core model
- Containers that talk to a storage backend directly
- Speculative abstractions for out-of-scope items in the current milestone
