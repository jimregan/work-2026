# Milestone 3 — Staleness detection and rebuild planning

Status: **planning only**. Do not implement any of this. Do not add
abstractions in milestone 2 in anticipation of it. This file exists so the
thinking is recorded, not so it can be built early.

Expect this milestone to be wrong the first time. That expectation is the reason
it is deferred rather than folded into milestone 2: it needs the model and the
execution machinery to be settled before it can fail informatively.

## The question

"Is this artifact still the correct output of this transformation on these
inputs?"

## What makes it answerable

Artifact UUIDs and content hashes are independent (ADR 0012). Persistence must
guarantee that an artifact UUID is bound to one immutable artifact record;
under that invariant, comparing an execution's recorded input UUIDs remains a
sound graph-version check. The content hash separately verifies that bytes at a
content locator still match the content referenced by that record. Staleness
must not treat UUID syntax itself as an integrity guarantee.

Milestone 2 execution records already carry what is needed: transformation
identity (digest + instantiated config), input artifact IDs, output artifact
IDs, resolved digest, and pinned/unpinned status.

## The hard parts

**Ephemeral nodes have no stored bytes to compare.** A 16 kHz WAV
materialization may not exist when the question is asked. Staleness for
ephemeral artifacts is a claim about whether regenerating them *would* produce
the same result, which is a different question from whether stored bytes match.

**Unpinned executions.** A tag-resolved run may have executed code that no
longer exists at that tag. The recorded digest says what ran; nothing says
whether the current tag still points there. Detectable, but it means staleness
has an axis beyond inputs — the transformation itself may have moved under a
stable-looking reference.

**Non-determinism makes "different output" ambiguous.** A rerun producing
different bytes could mean the inputs changed, or that the transformation is not
deterministic. Without resolving open decision 3 in milestone 2, staleness
cannot distinguish stale from merely rerun.

**Acquisition has no upstream to compare against.** A fetch specification's
inputs never change — the URL is the content. Whether a refetch is warranted is
a policy question about the world, not a graph question, and it reads
classification state (retry-after, gone, obtained) rather than input hashes.
This is a genuinely different staleness rule for acquisition, not a special
case of the general one.

**Asserted correspondence must be honoured.** Where a collection has relocated
and a human asserted old-spec ↔ new-spec identity, the planner must read that
correspondence and skip the refetch. Ignoring it refetches the collection, which
is the whole reason the assertion exists.

**Redirect chains and collection relocation.** A wholesale relocation produces
many new specifications with no responses. Distinguishing "one member moved"
from "the collection moved" is an aggregate judgement across specifications
sharing an origin or prefix; a single fetch cannot see it. Whether collections
are first-class artifacts or reconstructed by grouping is unresolved and should
be settled before this milestone starts.

**Nodes without a continuation.** A `ToolView` (ADR 0011) or any other
artifact fed into an external tool either gets a downstream execution and
output, or it doesn't — that's a structural, graph-queryable fact, not
tool-log parsing. A node with no continuation needs rerunning, or routing
through a different transformation; this milestone is where that detection
belongs. It intersects milestone 2's never-resolved question about a
transformation that exits non-zero having written partial output — both
are about what "didn't finish" actually means for a node in the graph.

## Cheapness constraint

The retry view reads classification outputs, which are small and already carry
retry-after — not raw responses. The expensive judgement happens once per
response at classification time, not once per response per scheduler pass. Any
design that re-examines response bytes on every planning pass has failed this
constraint.

## Prerequisite decisions

- Milestone 2 open decision 3 (non-determinism) must be settled first.
- Collection modelling — first-class artifact versus post-hoc grouping — must be
  settled first.
