# 12. UUID artifact and batch identity, separate from content identity

## Status

Accepted and implemented. Supersedes the artifact-identity decision in ADR
0003.

## Context

ADR 0003 made an artifact's local identifier the hash of its content wherever
content was available. That joined two concepts which need different equality
rules: a graph node and the bytes to which that node refers.

A transformation does not necessarily change bytes. Staging content in a
tool-specific layout, attaching a new interpretation, or recording another
provenance path may legitimately produce a new artifact whose content is
byte-identical to its input. Content-hash node identity collapses those nodes.
It consequently loses graph structure and forces metadata and provenance to
share the equality rules of content storage.

Metadata for a corpus is intended to live in Git. A Git commit hash is not a
suitable replacement for an artifact or batch identifier, however. It does not
exist until the commit is made, changes when history is amended or rebased, and
identifies the complete repository snapshot rather than a domain transaction.
Embedding the containing commit hash in that commit's metadata would also be
circular.

## Decision

- Every artifact node has a generated UUID, represented as the `local_id` of
  its layer-typed `LayerId`. Layer typing remains: a UUID does not create a
  global "same real-world thing" relation.
- Every logical batch transaction has its own generated UUID. A batch UUID is
  allocated before its metadata is committed and remains stable if the Git
  history containing it is rewritten.
- `Artifact.content_hash` identifies and verifies materialized content. It is
  independent of the artifact UUID. Several artifact nodes may refer to the
  same content hash and locator.
- Git commits identify versions of the metadata graph. A commit may record
  which artifact and batch UUIDs it introduced, but its hash is not their
  identity and is not embedded into the metadata it contains.
- UUID version 4 is the default. No ordering or domain meaning is encoded in an
  identifier; acquisition, document, and processing order continue to come
  from their explicit time fields.

The resulting identities have separate responsibilities:

| Identifier | Identifies |
|---|---|
| Artifact UUID, qualified by layer | A node in the artifact graph |
| Batch UUID | A logical batch transaction |
| Content hash | Immutable materialized bytes |
| Git commit hash | A versioned snapshot of corpus metadata |

## Consequences

- A no-op or byte-preserving transformation produces a new artifact UUID and a
  new provenance edge while reusing the same content hash.
- Identical bytes may be deduplicated in content storage without deduplicating
  artifacts or their histories.
- Fetch specifications are ordinary UUID-identified artifacts. Their URL is
  content or metadata according to the specification representation; hashing
  the URL is not required to manufacture node identity.
- An artifact record immutably binds its UUID to its content reference and
  metadata. UUID generation does not itself supply that immutability;
  persistence must reject attempts to reuse one UUID for a different record.
  Content hashes remain the independent integrity check for materialized bytes.
- The `content_address` helper remains useful for content verification and
  placement. `locator_address` is no longer an artifact-identity mechanism.
- New artifact-creation paths use UUIDs. Explicit `LayerId` construction and
  deserialization remain able to read pre-ADR records during the design phase;
  no migration framework is introduced for data that is not yet finalized.
