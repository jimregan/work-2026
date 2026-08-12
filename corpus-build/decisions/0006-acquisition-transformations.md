# 6. Acquisition transformations: fetch, classify, WARC

## Status

Accepted.

## Context

`02-transformation-registry.md`'s "Acquisition transformations" section
plus conversation settled: redirects followed by default with both
locations recorded; `SourceRef` as the bytes-free intake layer name;
non-determinism noisy-by-default; a dirty git tree refuses registration.
WARC was raised as a future datasource and an optional fetch output.

## Decisions

**Scope boundary.** `corpus_build.acquisition` implements fetch,
redirect-chain walking, classification, and WARC writing as plain,
directly-testable Python functions — the logic a fetch transformation's
container would run. It is *not* wired into `execution.orchestrator` as an
actual registered-and-executed `Transformation`: doing so needs a real
container image (base image, dependency pinning, an entrypoint that reads
`/inputs`/`/config/config.json` and writes `/outputs`), which is a
packaging exercise with no specified base image, not a modelling question.
`execution.docker_runner`/`orchestrator` already proved the bind-mount
execution path end-to-end against a real container (`busybox`); a fetch
image built on that machinery would call exactly `fetch()`/`classify()`
here. Building that image is deferred.

**"Both locations recorded" is on the response, not the SourceRef.**
Artifacts are immutable — a `SourceRef` created before a fetch can't be
amended with a resolved location afterward. `create_response_artifact`
puts both `original_url` and `final_url` (plus `chain_collapsed` and
`elided_hop_count`) in the *response* artifact's metadata instead, since
that record is freshly created after the fetch completes and can legally
carry both. This is a reading of "recorded in the initial node" as "the
fetch's own record," not a literal mutation of the pre-fetch `SourceRef` —
flagging the interpretation in case a stricter reading (e.g. a
superseding `SourceRef` version) was intended.

**Redirect loops.** `walk_redirect_chain` tracks seen URLs and stops
(without raising) on a repeat, in addition to the required `max_depth`
bound — the doc requires only a depth bound and warns that "a naive
follower will not notice" a loop; tracking seen URLs makes the walker
non-naive by that definition. No formal DAG edge chains one hop's response
to the next hop's `SourceRef` — each hop is an independent fetch execution
— so a loop in fetch history never becomes an artifact-graph cycle; it's
only visible by reading the returned hops.

**Classification output** reuses the `CorrespondenceId` layer rather than
inventing a new artifact kind, exactly as the doc directs ("looks like the
correspondence artifacts already in the model"), with `response` and
`specification` pointers in its metadata.

**WARC** (`acquisition/warc.py`, via `warcio`) writes a single fetch or an
entire redirect chain to one WARC file — every hop but the last as
`WARC-Type: resource`, the last as `response`. Reading/ingesting from
existing WARC files as an acquisition datasource, and merging WARC files,
are both still out of scope; only writing is implemented.

## Consequences

- `fetch()` never raises for an HTTP or transport-level failure — every
  outcome is a `FetchResult`, matching "an execution that produced no
  artifact is a bug in the fetcher."
- Tests exercise a real local HTTP server (stdlib `http.server`), not
  mocks, covering 200/404/410/500/503-with-Retry-After, redirect-following,
  non-following, and a genuine two-hop redirect loop that terminates
  without hanging.
- `_parse_retry_after` only handles the delay-seconds form of
  `Retry-After`, not the HTTP-date form — narrower than a production
  fetcher would need, acceptable for this milestone's scope.
