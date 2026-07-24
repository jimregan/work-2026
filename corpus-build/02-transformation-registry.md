# Milestone 2 — Transformation registry and execution

Status: current

Read `AGENTS.md` first. Everything there still applies. This file adds the
scope and the decisions specific to this milestone.

## Goal

Make transformations executable. A registered transformation, given input
artifacts, runs in a Docker container and produces output artifacts with
complete provenance. Transformation records were inert in milestone 1; they stay
representable-without-having-been-run, but now they can also run.

## Scope

- Transformation registration: module, config schema, image reference
- Image reference resolution and pinning policy (see below)
- Hydra config composition and storage
- Docker execution via bind mounts
- Execution records linking transformation, inputs, outputs, and run metadata
- CLI surface for register / inspect / run

## Deliberately excluded

Do not build these, and do not add abstractions in anticipation of them:

- **Staleness detection and rebuild planning.** Deferred to milestone 3; see
  `03-staleness.md`. Run what you are told to run. Do not add a
  dependency-walking scheduler, a `--force` flag implying its opposite, or
  cached-result short-circuiting. Planning for milestone 3 means prose in that
  file, not abstractions in this one — a `StalenessPolicy` interface with a
  single implementation returning `True` is the failure mode to avoid.
- **Retry scheduling.** Classification outputs carry retry-after; nothing in
  this milestone acts on it.
- Scheduling, queuing, retries, parallel execution
- Remote or cluster execution (SLURM, Kubernetes)
- GPU scheduling policy beyond passing device flags through
- Ingest of real corpus data
- Views, partitions, timeline mapping

If a task seems to require excluded work, say so and propose the smallest
in-scope subset instead of expanding the milestone.

## Transformations are module + config schema pairs

A transformation is a Python module paired with its structured Hydra config
schema. The schema is part of the module and travels with the image. The
registry validates instantiated configs against the schema rather than treating
config as opaque text — a config that does not typecheck against its schema is
rejected at registration, not at run time.

## Image reference policy

Digest is the intended path. Tags are permitted, with the consequences falling
on whoever chose them.

| Given | Behaviour |
|---|---|
| Digest only | Pinned at registration. Normal path. |
| Tag only | Resolved at execution. Execution record marked **unpinned**. |
| Both | Resolve the tag, compare to the digest, **hard fail on mismatch**. |

Rules:

- Never silently prefer one reference over the other when both are given, and
  never repair a mismatch. Fail loudly with both values in the error.
- An unpinned execution is recorded as unpinned so provenance queries can
  identify which results are not reproducible. Do not hide this, and do not
  refuse the run because of it.
- Record the resolved digest in every execution record regardless of which path
  was taken.

## Transformation identity

Transformation identity is **image digest + instantiated config**.

Input artifact IDs are *not* part of transformation identity — they belong to
the execution record. The same transformation applied to different inputs is one
transformation and two executions.

The transformation record additionally carries the **git commit hash of the
module file**, so the source that produced the behaviour is identifiable
independently of the image. This is recorded, not part of identity: two images
built from the same commit with different base layers are different
transformations.

## Config storage

Store **both** the composed config and the hierarchy plus overrides. Prefer the
hierarchy as the primary representation — it is the only sane way to work with
Hydra, and it stays readable and diffable. The composed result is stored
alongside it as the reproducibility record.

- On re-execution, reproduce from the composed config, not by recomposing the
  hierarchy. Recomposition can drift if the config tree changed.
- If recomposing the stored hierarchy does not reproduce the stored composed
  config, that is a reportable inconsistency — surface it rather than choosing
  a winner.

## Docker execution contract

Containers interact with the world **only through bind mounts**. This is not
negotiable: a container that knows about a storage backend breaks the pluggable
storage constraint in `AGENTS.md`.

- Inputs bind-mounted read-only at fixed, documented paths.
- Outputs written to a bind-mounted writable directory; the runner ingests them
  into the backend afterwards.
- Config passed in as a mounted file, not as an ever-growing pile of CLI flags
  or environment variables.
- The container receives no backend credentials, no backend URLs, and no
  network access it does not need.
- Path layout inside the container is part of the contract and must be
  documented in one place, not spread across call sites.

## Acquisition transformations

Fetch and classification are transformations like any other and are registered
through the same machinery. See the acquisition constraints in `AGENTS.md` for
the model; this section covers only what milestone 2 must implement.

- A fetch execution produces a response artifact for every outcome, including
  failures and redirects. An execution that produced no artifact is a bug in the
  fetcher, not a failed fetch.
- Classification is a separate registered transformation over a stored response.
  Reclassifying an old response is an ordinary execution, not a refetch path.
- Classification output points at both the response and the specification. It is
  small and structured, and it looks like the correspondence artifacts already
  in the model rather than a new artifact kind.

### Redirect following

Whether the HTTP client follows redirects itself is **configurable**. Following
is faster and matches client defaults; not following makes every hop a response
artifact and keeps the chain reconstructible.

When the client followed redirects, the response artifact and its execution
record must both carry a **chain-collapsed flag and the elided hop count**. Two
audiences need this: the person running the fetch, at runtime, and anything
reading provenance later, which otherwise cannot tell a direct response from a
three-hop one. Treat it exactly as the unpinned-execution flag is treated —
permitted, recorded, queryable, never hidden.

Whatever the default, a chain walker needs a depth bound. Redirect loops
revisit an existing specification node, so the artifact DAG stays acyclic and
the cycle is only visible in fetch history; a naive follower will not notice.

## Execution records

An execution record links to exactly one transformation and carries:

- input artifact IDs
- output artifact IDs
- resolved image digest, and whether the run was pinned or unpinned
- exit status, wall-clock duration, start time
- captured stdout/stderr reference
- the three times from `AGENTS.md` handled correctly — processing time belongs
  to the execution, document time belongs to the artifacts

## Tests

- Digest-only, tag-only, and both-given resolution paths, including the
  mismatch hard-fail
- Unpinned runs are marked unpinned in the record
- Config validated against schema; invalid config rejected at registration
- Composed and hierarchy configs both stored; recomposition drift detected
- A container cannot reach the backend — assert absence of credentials and
  backend paths in the container environment
- Transformation identity is stable across different inputs
- A transformation that has never executed is still representable and queryable
- A failed fetch produces a response artifact; a 404, a 503, and a redirect all
  round-trip through storage and reclassify without refetching
- Chain-collapsed flag and hop count present on both response and execution
  record when the client followed redirects
- A specification has no mutable status field; state is derived from
  classifications

## Open decisions

Surface these; do not settle them unilaterally.

1. Whether redirect-following defaults to on or off. Off is more faithful and
   costs round trips; on matches client defaults, and defaults tend to become
   the only setting anyone uses.
2. Naming for the bytes-free artifact kind. "Artifact" for a thing with no bytes
   surprises every new reader. A subtype name — `SourceRef`, `FetchSpec`,
   `Locator` — keeps the model uniform and is self-describing where it appears;
   renaming the general category to "node" or "entity" is more faithful but
   renames the central concept.
3. Non-determinism — a transformation rerun with identical inputs and config
   producing different bytes. Detect and flag, or accept silently?
4. Whether a dirty git working tree at registration is a warning or a refusal.

Note that "failed-execution outputs" is no longer open: under the response model
a failed fetch produces an artifact. The general case — a transformation that
exits non-zero having written partial output — is still open, and is not the
same question.
