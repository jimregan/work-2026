# 2. Metadata-primary storage, content as files not blobs

## Status

Accepted.

## Context

`StorageBackend` needs to decide whether it owns artifact content bytes
directly, or only structured records that point at content stored
elsewhere. Some artifacts have no content yet (a planned output of a
transformation that hasn't run); some have content that matters (audio,
transcripts); persistence is independent of position in the graph, so
"does this artifact currently have materialized content" cannot be assumed
either way.

## Decision

The backend contract stores `Artifact` and `Transformation` records only.
Content is referenced by `content_locator`, a plain string that is
conventionally a filesystem path, plus an independent `content_hash` for
verification. Both are optional and independent of each other.

When a backend does write content (as opposed to just recording where it
is), it must write a real file, never a database BLOB column — content
must be readable by ordinary software (an audio player, `less`, ffmpeg)
without going through this library.

## Consequences

- The SQLite backend has no BLOB column and no code path that could grow
  one; `content_locator` is TEXT.
- The filesystem backend never moves or copies bytes on `put_artifact`; it
  only records whatever locator string it's given.
- Nothing in the core model or either backend can materialize content from
  metadata alone — reconstructing an ephemeral artifact's bytes is a
  transformation-execution concern, explicitly out of scope for this
  milestone (see AGENTS.md).
