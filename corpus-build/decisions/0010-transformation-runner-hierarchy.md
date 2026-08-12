# 10. A transformation-runner class hierarchy, generalizing beyond Docker

## Status

Accepted. Not implemented — design only, same status ADR 0007 had before
its fields existed.

## Context

ADR 0009 split `Transformation` into containerized and code-versioned
kinds at the *identity* level, correctly recognizing that text
transformations have no environment-sensitive runtime worth Docker-pinning.
But it left the *execution* side lopsided: `execution.orchestrator`
constructs and persists a real `Execution` record (start/end/exit status/
outputs, non-determinism check) for the containerized path, and nothing
does the equivalent for the code-versioned path. Running
`batch_import`/`extract`/the CTM aligner today means calling the plain
function and inspecting the return value by hand — no `Execution` record
is ever created, so none of that work is actually represented in the
provenance graph yet, per the practical-pipeline status report.

There is also a missing middle tier. Today's model only has two ways to
run something: a Python function call, or a full Docker container. Shelling
out to a local executable — a real, common case — fits neither: it isn't
a Python function, and it doesn't need a bespoke container image the way
an acoustic/ML transformation does.

## Decision

A base runner class owns the mechanics common to every kind of run:
composing/validating config, resolving identity (code-version or image),
constructing and persisting the `Execution` record, running the
non-determinism check — generalizing what `execution.orchestrator`
currently does only for the Docker case. `Transformation` and `Execution`
(the dataclasses) are unchanged; this is purely the runner/execution-side
abstraction that produces and persists them consistently, regardless of
how the underlying work actually happens.

Three concrete subclasses, forming a hierarchy a transformation can be
promoted along without changing its identity-level kind:

1. **Function wrapper** — calls a Python function directly. Formalizes
   what `batch_import`/`extract`/the aligner already do ad hoc today, now
   actually producing an `Execution` record.
2. **Shell-out wrapper** — invokes a local executable, not in a container.
   New; nothing today handles this case. **How this tier sources its own
   reproducibility guarantee is explicitly left open** — a pinned
   interpreter, a recorded environment snapshot, something else entirely —
   and is not decided by this ADR.
3. **Docker wrapper** — formalizes what `execution.orchestrator` +
   `execution.docker_runner` already do, reframed as one case of the base
   class rather than the only path. Distinguishes further between a
   *generic* container (a reusable, non-bespoke execution environment that
   just runs whatever script/config it's given — the "promoted" version of
   tiers 1 or 2) and a *custom* image purpose-built for one transformation
   (today's containerized path, for genuinely environment-sensitive tools).

Docker Compose's YAML vocabulary (services, command, environment, volumes)
is adopted as the shape guide for how a transformation's "how to run it"
configuration is expressed across all three tiers — not because Compose
itself is invoked, but so that promoting a transformation from one tier to
the next is filling in more of the same schema, not switching to an
unrelated one. Nothing in the codebase uses Docker Compose today; this is
new territory, not an extension of existing config-composition code (Hydra
composition is unrelated and keeps its own meaning).

## Consequences

- Existing code is unaffected until this is actually built:
  `batch_import`/`extract`/the aligner keep working as plain functions,
  `execution.orchestrator` keeps working for Docker. This ADR does not
  require migrating them, only describes where they'd land.
- Shell-out reproducibility sourcing is a real open question and should be
  surfaced again, not silently defaulted, when this is implemented.
- Whatever config shape is adopted needs to accommodate the generic-vs-
  custom container distinction inside the Docker tier, not just
  containerized-vs-not at the top level.
