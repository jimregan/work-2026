# Implementation audit against the design documents

Audit date: 2026-08-02

## Scope and method

This audit predates ADR 0012. Its statements about content-addressed artifact
identity describe the implementation and design at the audit date, not the
current intended model. The subsequent implementation uses generated UUIDv4
artifact IDs, independent content hashes, UUID batch-import identity, and
distinct nodes for byte-preserving outputs.

This audit compares the Python implementation and tests with `AGENTS.md`, the
four milestone documents at repository root, and ADRs 0001–0010. It treats
accepted ADRs as authoritative, except where an ADR explicitly says that its
decision is not yet implemented. It treats milestone 4 as current.

Verification at the time of audit:

- `pytest -q`: 141 passed.
- `mypy --strict src`: no issues in 44 source files.

Passing tests establish consistency with the present test suite, not completion
of every documented milestone requirement. The principal mismatches below are
requirements for which code and tests do not yet exist.

## Material mismatches

### 1. ADR 0007 and the milestone-4 transformation windows are not implemented

Milestone 4 says that this milestone implements transformation families,
`module_path`, `valid_from`, `valid_until`, registration-time supersession via
git ancestry, and a run-time acquisition-window guard. None of those fields is
present on `Transformation`; neither backend serializes them; registration does
not inspect git ancestry or close a predecessor's window; and neither runner
checks input acquisition times.

This agrees with ADR 0007's own status, “Not yet implemented,” but conflicts
with milestone 4's statement that the work is in scope and with its associated
tests. This is the largest implementation gap because date-windowed retention
of old extractors is central to the continuous-corpus motivation.

### 2. The milestone-4 content layout is not implemented

The required layout is
`corpus/<acquisition-date>/<hash-shard>/<original-filename>`, with configurable
shard depth. `FilesystemBackend` deliberately writes metadata sidecars only,
which is consistent with the backend contract, but no separate content writer
implements the convention. `batch_import` leaves `content_locator` pointing to
the original imported file, and Docker output ingestion uses caller-selected
paths. There is no shard configuration or test for the promised layout.

The fix should remain separate from `StorageBackend`, as milestone 4 requires:
this is a missing content-placement mechanism, not a reason to expand the
metadata backend interface.

### 3. The concrete Riksdag proof is incomplete

The old-API batch importer and old-shape video-URL extractor exist as plain
functions and have focused tests. However, the repository contains no fixture
or recorded run demonstrating the representative archived sample, no live API
adapter or current-shape family member, and no old-shape/new-shape
correspondence artifact from the same debate. Consequently the following
milestone-4 claims remain unverified:

- a batch-import execution producing each specification/response pair;
- a registered extraction execution over real Riksdag material;
- reuse of the fetch transformation for the extracted real video URL;
- supersession between old and live API transformation family members;
- a persisted asserted correspondence between an old and current document.

`FunctionRunner` now supplies the general mechanism needed to create execution
records for code-versioned functions, but `import_directory`, extraction,
classification, and fetch are not exposed here as runner-compatible registered
transformation callables. The distinction matters: returning artifacts from a
plain function validates transformation logic, but it does not put that work in
the provenance graph.

### 4. Fetch and classification do not yet satisfy their full execution-record contract

The acquisition functions correctly model HTTP and transport failures as
ordinary `FetchResult` values, keep classification separate, retain redirect
information on response artifacts, and can write WARC. They are not themselves
wired through a runner as registered transformations. Therefore redirect
collapse and elided-hop information can be present on a response artifact but
is not automatically present on a corresponding `Execution`, despite milestone
2 requiring it in both places. Reclassification without refetching works at the
function/artifact level, but there is no end-to-end registered execution path
that demonstrates it.

This limitation is explicitly recorded in ADR 0006, so it is a known scope
deferral rather than an accidental code defect. It nevertheless contradicts
the stronger completion language in milestone 2 and the phrase “existing
fetch transformation” in milestone 4.

### 5. Config-hierarchy requirements and implementation have drifted

Milestone 2 requires storing both the composed configuration and hierarchy plus
overrides, and requires detection when recomposition drifts. ADR 0004 and the
`Transformation` docstring instead make hierarchy storage optional. The
registry defaults to storing a small descriptor containing `config_name` and
overrides, but no code recomposes it or compares the result with
`composed_config`. Milestone 4 already records the missing drift check.

This needs a documentation decision before implementation: either retain the
milestone-2 requirement that the hierarchy is always stored, or formally adopt
ADR 0004's optional hierarchy. In either case, when a hierarchy is present the
promised consistency check is still missing.

### 6. Code-versioned transformation identity differs from ADR 0009

ADR 0009 says identity is `implementation_ref + instantiated config`.
Registration actually uses
`<absolute module path>@<implementation_ref> + instantiated config`. The code
comment gives a sound reason: a git commit is repository-wide, so two modules at
one commit and with equal config must not collapse. Tests enforce the code's
behaviour.

The code is more defensible than the ADR wording, but an absolute path makes
identity machine-dependent. ADR 0007's intended persisted `module_path` could
resolve the ambiguity if it is defined as a stable repository-relative module
identifier and included in code-versioned identity. Until then, the ADR and
implementation disagree and identical registrations in two checkouts need not
converge.

### 7. ADR 0010's implementation status is stale

ADR 0010 says “Not implemented — design only.” The base
`TransformationRunner` and `FunctionRunner` are implemented, including role
checks, non-determinism checks, artifact persistence, and execution-record
creation. The Docker path has not been migrated to the hierarchy and the
shell-out runner remains absent. The accurate status is therefore “partially
implemented: base and function runner.”

## Documentation mismatches

- `AGENTS.md` says milestones live in `docs/milestones/`, but they are at the
  repository root.
- `02-transformation-registry.md` and `04-concrete-ingress-pipelines.md` both
  label themselves current. The former should be completed/superseded if
  milestone 4 is current.
- `AGENTS.md` lists milestones 1 and 2 as completed but milestone 2's own status
  is current.
- ADR 0006 still describes `batch_import` and extraction as prospective plain
  functions even though both now exist, while its broader “not registered and
  executed” limitation remains accurate.
- ADR 0007's status is accurate for the code but surprising beside milestone
  4's claim that implementation belongs to the current milestone.

These do not alter runtime behaviour, but they make it difficult to determine
which claims are safe to make in a paper.

## Areas that match well

The following architectural claims are supported by implementation and tests:

- Generic immutable `Artifact` records with independent persistence,
  acquisition time, and document time.
- Layer-typed identifiers and content-addressing helpers (the artifact-ID use
  described at the audit date was subsequently superseded and replaced by ADR
  0012).
- Transformation identity separated from executions over particular inputs.
- Multi-input and multi-output execution edges with DAG cycle rejection.
- Metadata-only storage behind one backend contract, demonstrated by filesystem
  and SQLite implementations sharing a conformance suite.
- Container image resolution for digest, tag, and combined references, with
  mismatch refusal and recorded pinning state.
- Bind-mount-only Docker execution with an empty environment and networking
  disabled by default.
- Noisy-by-default non-determinism detection for repeated runs over identical
  inputs.
- Fetch failures represented as results, separate response classification,
  redirect-chain handling, WARC output, and configurable request headers.
- Code-versioned function execution with persisted execution records.
- Correspondence artifacts carrying query-relevant alignment method and edit
  metadata, without selecting a canonical variant.

## Recommended order of correction

1. Resolve and implement ADR 0007, including a portable module-family
   identifier, backend serialization, supersession, guard checks, and tests.
2. Wrap the concrete acquisition functions for `FunctionRunner` and persist one
   small archived Riksdag demonstration end to end.
3. Add the separate content-placement helper and its layout tests without
   changing `StorageBackend`.
4. Select and capture a current API example, then create the required asserted
   correspondence artifact if a matching debate is available.
5. Decide the config-hierarchy rule and implement recomposition drift checking.
6. Update milestone and ADR statuses so the paper and repository tell the same
   implementation story.
