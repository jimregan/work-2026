# 3. Content-addressed identity, with an intake exception

## Status

Superseded by [ADR 0012](0012-artifact-and-batch-identity.md). Retained as the
historical decision that the implementation may still reflect.

## Context

`03-staleness.md` (milestone 3, planning-only) assumes "Identifiers are
content hashes... so an input cannot change under a stable ID." Decision
[[0001-per-layer-typed-identity]] settled the type shape of identity but
left `local_id` an arbitrary string, which does not give that guarantee.
Ingest is expected to be mostly web sources: a fetch has no content until
after it runs, and the same URL fetched on different days can produce
different content — so "hash the content" cannot be the whole rule.

## Decision

`LayerId.local_id` should be content-addressed wherever possible, via two
convention helpers in `corpus_build.model.content_addressing` (not enforced
by the type system — the core model has no notion of "current content" to
hash from; callers invoke the right one):

- `content_address(data: bytes)` — SHA-256 of materialized bytes. The
  default for any artifact with actual content, interior or terminal.
- `locator_address(origin: str, at: date | datetime)` — SHA-256 of
  `date + origin`, for an intake/specification node declared *before* a
  fetch has happened. The date is part of the hash: refetching one URL on a
  different day is a different intake node, not a mutation of the same ID.

An intake node and the response it eventually produces are two separate
artifacts linked by a fetch transformation, not one artifact whose id
changes when content arrives.

Naming an actual "intake" / "specification" `LayerId` subclass is
deliberately **not** done here — `02-transformation-registry.md` lists that
naming as an open decision for milestone 2, and inventing one now would be
exactly the "speculative abstraction for an out-of-scope item" AGENTS.md
warns against. This decision settles the hashing mechanism only.

### Deferred to milestone 2 (recorded so it isn't lost)

- **Final-node filenames.** A terminal/deliverable artifact's `content_hash`
  and `id.local_id` are still ordinary content hashes — nothing changes
  about identity. But its `content_locator` needs a human-readable filename
  (mirroring the original filename when a whole file was processed, or a
  specified naming convention otherwise), unlike interior artifacts which
  can live at hash-named paths. This is a naming rule applied when a
  transformation writes output, i.e. execution machinery — nothing to build
  in the data model for it.
- **WARC as an acquisition datasource.** Fetching should have an option to
  write WARC; intermediate records should be writable as WARC-internal
  records (`WARC-Type: resource`). Merging WARC records is explicitly out
  of scope for this library. Relevant to the acquisition transformation
  design in milestone 2, not to the storage backend contract.

## Consequences

- `Artifact.content_hash` and `Artifact.id.local_id` coincide for a
  content-addressed artifact. They are not merged into one field:
  `content_hash` is the hash of whatever is currently materialized at
  `content_locator` and is independently re-derivable as an integrity
  check, while `id.local_id` is fixed at creation and may be a locator hash
  when there is no content yet.
- Milestone 3's staleness argument ("a stable ID implies stable content")
  now holds for content-addressed artifacts by construction. It does not
  extend to intake nodes, which is why acquisition needs its own staleness
  rule (`03-staleness.md` already treats this as a separate case).
