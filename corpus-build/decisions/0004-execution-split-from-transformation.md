# 4. Execution split from Transformation

## Status

Accepted.

## Context

`02-transformation-registry.md` states transformation identity is "image
digest + instantiated config" and explicitly excludes input artifact IDs
from that identity: "the same transformation applied to different inputs
is one transformation and two executions." Milestone 1's `Transformation`
dataclass bundled `input_ids`/`output_ids` and a single `executed_at`
directly onto the same record as `implementation_ref`/`container_image`/
`parameters`, which cannot represent "one transformation, many executions"
— reusing a transformation over new inputs required a duplicate record.
This also blocks non-determinism detection (comparing multiple executions
of the same transformation over the same inputs), which milestone 2
requires by default (see decision below and `03-staleness.md`'s reliance on
stable transformation identity across reruns).

The graph shape itself — three nodes (`output → execution → transform`) vs.
two independent edges from output — was also an open question, settled in
conversation: three-node chain is the real structural graph; the output
artifact additionally carries a denormalized pointer to the transformation
for convenience, but the execution is the sole authoritative link to it.

## Decision

Split into two model types:

- `Transformation` (`corpus_build.model.transformation`): identity only —
  `id` (digest-or-tag + composed config), `implementation_ref` (git commit
  of the module), `container_image_ref`, `composed_config`. No inputs,
  outputs, or execution timing. `config_hierarchy` is optional — storing it
  alongside `composed_config` is a registrant's choice (readability/diffing
  only), not a requirement; `composed_config` is the one thing reproduction
  always uses.
- `Execution` (`corpus_build.model.execution`, new): `transformation_id`,
  `input_ids`/`output_ids` (the actual provenance edges), resolved image
  digest, `pinned: bool`, timing (including `processing_time`, moved off
  `Artifact`), stdout/stderr refs, and `execution_metadata` for
  transformation-specific structured data that doesn't belong on every
  execution (e.g. a fetch's chain-collapsed flag).

`Artifact` gains `produced_by_execution: ExecutionId | None` (primary) and
`produced_by_transformation: TransformationId | None` (denormalized copy of
`execution.transformation_id`, for a reader holding only the artifact
record). `Artifact.processing_time` is removed.

Provenance graph functions (`assert_acyclic`, `find_cycle`,
`ProvenanceGraph`) and the storage backend's producing/consuming queries
now operate on `Execution`, not `Transformation`. The backend contract
gains `put_execution`/`get_execution`/`all_executions`/
`executions_producing`/`executions_consuming`/`executions_of` alongside the
now inputs-free `put_transformation`/`get_transformation`.

## Consequences

- Milestone 1's "a transformation that has not been run must be
  representable" requirement is now automatic: an unexecuted transformation
  is simply one with zero `Execution` records, no `executed_at=None`
  sentinel needed.
- Non-determinism checking (decision recorded alongside this one) becomes
  possible: compare `output.content_hash` across `backend.executions_of(t)`
  entries sharing the same `input_ids`.
- This touched already-committed milestone-1 code (`3c23f9a`) and its
  tests; both backends and the full test suite were updated in the same
  pass, not left inconsistent.
