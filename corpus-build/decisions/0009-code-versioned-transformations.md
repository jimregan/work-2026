# 9. Code-versioned transformations, alongside containerized ones

## Status

Accepted. Implemented (model, registry, both storage backends, CLI
`register`); `cli run` remains containerized-only, see Consequences.

## Context

`02-transformation-registry.md` and `AGENTS.md` treated every
transformation as containerized: identity is image digest + instantiated
config, `container_image_ref` is a required field, and packaging a
transformation into a Docker image was the only path — for `fetch`,
`classify`, `batch_import`, `extract`, and the CTM aligner, decision 0006
treated that packaging as *deferred*, not avoided.

That was too broad. Docker containerization exists to pin an
environment-sensitive runtime — CUDA build, model checkpoint, OS-level
codec library — where the same source code could genuinely behave
differently depending on what it's running against. Text transformations
(`batch_import`, `extract`, the CTM aligner wrapper) have no such surface:
they are stdlib-or-near-stdlib, and the git commit already backing
`implementation_ref` (checked clean at registration) is a complete
reproducibility guarantee on its own. Acoustic/ML transformations are the
ones that actually need image-level pinning; conflating the two meant
every text transformation was carrying a packaging obligation it never
needed to discharge.

## Decision

`Transformation` is one of two kinds, distinguished by whether
`container_image_ref` is present:

- **Containerized** (`container_image_ref` given): unchanged from
  `02-transformation-registry.md` — digest/tag/both resolution policy,
  identity rests on the image reference.
- **Code-versioned** (`container_image_ref` is `None`, not "not yet
  packaged"): identity rests on `implementation_ref` alone.

`implementation_ref` is populated the same way for both kinds — resolved
by a `registry.code_version.CodeVersionSource`, a protocol with one
concrete implementation (`GitCommitSource`, wrapping the existing
clean-tree-checked git commit lookup) chosen deliberately to be pluggable:
a future non-git versioning strategy (e.g. a dependency-lockfile hash)
can be added without changing `Transformation` or the registry again.

`register_transformation` takes `container_image_ref: str | None = None`
and `code_version_source: CodeVersionSource = GitCommitSource()`; which
kind results falls out of whether an image ref was passed, not a separate
flag. `transformation_identity`'s first parameter is renamed `version_ref`
(from `image_ref`) since it is no longer always an image.

`Execution.resolved_image_digest` becomes `str | None`; `pinned` stays
`bool` and is `True` for a code-versioned execution — the registered
`implementation_ref` is an exact commit, not a movable reference, so there
is no unpinned case the way an image tag has one.

## Consequences

- Breaking model change, made freely: no concrete corpus exists yet, so
  there is nothing to migrate. `Transformation` and `Execution` field
  order changed (optional fields moved after required ones); both storage
  backends' schemas changed (`container_image_ref`, `resolved_image_digest`
  now nullable columns).
- `cli register`'s `--image` is now optional. `cli run` is unchanged and
  stays containerized-only — it refuses with a clear error if pointed at a
  code-versioned transformation, rather than crashing on a missing image
  reference. Running a code-versioned transformation's plain function
  directly (as `fetch`/`classify`/`batch_import`/`extract`/the aligner
  already are, per decision 0006) and constructing its `Execution` record
  by hand is the path until/unless a non-Docker execution path is built —
  not designed here.
- `docs/02-transformation-registry.md`'s "Transformation identity" section
  is marked superseded-in-part rather than rewritten; it still accurately
  describes the containerized case.
