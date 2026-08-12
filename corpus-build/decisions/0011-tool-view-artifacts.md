# 11. ToolView: materializing content into a tool's hardcoded layout

## Status

Accepted. Not implemented — design only, same status as ADR 0007 and 0010.

## Context

Many real tools and scripts hardcode a directory layout and file-naming
convention (Montreal Forced Aligner's speaker-directory, matching-basename
convention is the case that prompted this). corpus-build's own content layout
does not generally match those conventions; rewriting every such tool to
tolerate that is not viable, and was never the plan — `AGENTS.md`'s one-sentence
thesis has said "task-dependent views" from the start, without the phrase
ever being given concrete shape until now.

The fix is a pre-transform: materialize the content a tool needs under the
name and path it expects, as its own step, distinct from the tool's actual
work. Depending on the storage backend this might be a hard link (cheap,
same filesystem) or a real copy (a different backend, or no hard-link
support) — an implementation choice per backend, not a single universal
rule.

The output of that materialization is, in an important sense, ordinary:
"effectively a transform, but (typically) without content changes" — the
bytes are usually identical to the input, only the location/name differs.
ADR 0012 resolves the identity problem directly: the materialized view has a
new artifact UUID even when it reuses its input's content hash. The `ToolView`
layer still describes the artifact's role, but a separate layer is no longer a
workaround required merely to prevent byte-identical nodes from colliding.

A related scope question came up: once a tool has been pointed at a
materialized view, is parsing *that tool's own log output* in scope?
No — that's tool-specific and left to whoever integrates that tool. But a
structural fact is in scope: whether a materialized view has a
*continuation* in the graph (a downstream execution and output artifact)
is a plain provenance query, not log parsing, and "nodes without a
continuation need to be rerun, or routed through a different
transformation" is a real, useful thing to be able to ask — see
Consequences.

## Decision

- New layer: `ToolViewId` (`layer = "tool_view"`), joining the existing set
  in `model/identity.py`. This is what gives "task-dependent views" from
  the thesis actual shape.
- A `ToolView` artifact is produced by an ordinary transformation/
  execution — typically the function-wrapper tier (ADR 0010), since
  staging is usually just a filesystem operation — whose input is the
  real corpus artifact and whose output is the `ToolView`. Nothing new is
  needed on `Execution` or `Transformation`; this is a normal provenance
  edge with `output_ids` pointing at a `ToolViewId`.
- The `ToolView`'s `metadata` records the external filename/path it was
  staged under (and whatever else makes it resolvable — the target tool
  or convention, at minimum). `content_locator` points at the actual hard
  link or copy; `content_hash` is still the genuine hash of the bytes, as
  for any artifact — materialization does not get a special exemption
  from integrity checking just because it usually preserves bytes exactly.
- Hard link vs. copy is a `StorageBackend`-dependent implementation
  choice, made when materialization is actually built — not decided here.
- **Scope boundary**: corpus-build maintains the `ToolView` artifact and
  its filename mapping so it is queryable. Parsing a specific external
  tool's log format to make its messages human-readable again is
  explicitly out of scope — an exercise for whoever integrates that tool.

## Consequences

- No change to `Artifact`, `Execution`, or `Transformation` — `ToolView`
  is a new layer using the existing generic shapes, same as every other
  layer.
- `model/identity.py` gains one new registered `LayerId` subclass.
- Forward note for milestone 3: a `ToolView` with no downstream execution
  or output artifact is a structural, graph-queryable fact ("no
  continuation"), not tool-log parsing — and identifying these for rerun
  is a concrete instance of two things already sitting open: milestone
  2's never-resolved question about a transformation that exits non-zero
  having written partial output, and milestone 3's rebuild-planning scope
  (itself still blocked on the collection-modelling prerequisite). This
  decision does not attempt to solve either; it only names where staging
  artifacts intersect them.
