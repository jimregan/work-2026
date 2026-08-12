# 5. Execution machinery: image resolution, Docker, non-determinism

## Status

Accepted.

## Context

`02-transformation-registry.md` specifies image reference pinning policy,
a bind-mount-only Docker execution contract, and (per conversation) a
default-noisy non-determinism check suppressible only by an explicit config
flag. This covers the three pieces built to satisfy that: `execution/image_resolution.py`,
`execution/docker_runner.py`, `execution/orchestrator.py`.

## Decisions

**Image resolution** implements the digest/tag/both table exactly:
digest-only is pinned without a resolver call; tag-only resolves via an
injected `TagDigestResolver` and is marked unpinned; both given resolves
the tag and hard-fails on mismatch (`ImageDigestMismatchError`, carrying
both values, never repaired). The resolver is dependency-injected so the
policy logic is unit-testable without a registry; `docker_tag_digest_resolver`
is the real implementation, using `client.images.get_registry_data` to
query a digest without pulling.

**Docker execution** exposes exactly one mechanism: bind mounts at fixed,
documented paths (`/inputs/<layer>/<local_id>`, `/config/config.json`,
`/outputs/`). `docker_runner.run_container` never receives a
`StorageBackend` — it only sees artifacts already resolved to host paths —
and passes `environment={}` explicitly so no host/backend environment
leaks in. `network_disabled=True` by default; a transformation that
genuinely needs network (e.g. a future fetch transformation) opts in
per-call. Stdout/stderr are written to files and referenced by path, not
inlined, matching the "content is files, not blobs" storage decision
([[0002-metadata-primary-storage]]).

**Execution orchestration** (`execute`) resolves the image against the
*given* reference, then runs the container against the *resolved digest*
pinned reference (`repository@resolved_digest`) rather than re-trusting the
tag a second time — closes a TOCTOU gap between resolution and execution.
Output-file-to-`Artifact` mapping is left to a caller-supplied
`ingest_outputs` callback: which layer, which `artifact_type`, and where
content ultimately lives is policy this module doesn't own, same as
`docker_runner` leaving bind-mount content placement to its caller.

**Non-determinism** (`check_nondeterminism`) compares a new run's output
content hashes against every prior `Execution` of the same `Transformation`
over identical `input_ids`, raising `NonDeterminismError` on any content
hash or output-count mismatch. Suppressed only by
`transformation.composed_config["allow_nondeterministic"] = true` — read
"a flag in the input" from the conversation as a flag on the transformation's
config, not on the input artifacts, since it's a property of the
transformation's expected behaviour, not of any one input.

## Consequences

- A transformation registered with a tag only is executed against whatever
  digest that tag resolves to *at the moment of execution*, recorded on the
  `Execution` as unpinned — matches `03-staleness.md`'s framing of unpinned
  executions as a known, accepted reproducibility gap, not something this
  milestone tries to close.
- Testing this required a real Docker daemon and network access to pull
  `busybox`; both were available in the dev environment, so the test suite
  exercises real containers rather than mocking the Docker SDK. If this
  ever runs somewhere without Docker, `tests/test_docker_runner.py` and
  `tests/test_orchestrator.py`'s Docker-backed cases self-skip via a
  `client` fixture that pings the daemon first.
