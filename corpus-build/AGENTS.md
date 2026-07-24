# AGENTS.md

Guidance for coding agents working on this repository.

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

## Current milestone

**Artifact and provenance data model only.**

In scope:

- Artifact identity, typing, and content addressing
- Provenance edges (DAG, multi-parent)
- Transformation records as first-class objects
- Storage backend interface (at least two implementations to prove it is real)
- Query surface sufficient to answer "how was this artifact produced"

Out of scope until the model is stable — do not build these yet, and do not
add abstractions in anticipation of them:

- Actual execution of transformations (containers, scheduling, retries)
- Ingest of real RixVox or broadcast data
- View and partition machinery beyond the minimum needed to test the model
- Timeline mapping and coordinate systems
- Any web UI

If a task seems to require out-of-scope work, say so and propose the smallest
in-scope subset instead of expanding the milestone.

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
- Core model has no dependencies beyond the standard library. Backends may have
  their own.
- Dataclasses (frozen where the object is a value) for model types.
- Tests with pytest. Every provenance invariant gets a test: acyclicity,
  multi-parent, layer-specific identity, ephemeral regeneration.
- Property-based tests (hypothesis) for graph invariants are welcome.
- No block separator comments (e.g., ######### Variables ######)

## Anti-patterns

Concrete things that have been considered and rejected:

- A single canonical ID per "real world thing"
- A linear `Pipeline` or `Stage` class as the primary abstraction
- Persistence flags derived from position in the graph
- A single `timestamp` field
- Correspondence stored as a plain dict or as a field on one of the two sides
- Backend logic imported into the core model
- Speculative abstractions for the out-of-scope items listed above

