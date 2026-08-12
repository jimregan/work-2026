# 7. Transformation versioning via git ancestry, and acquisition-time windows

## Status

Accepted. Not yet implemented — scoped for the concrete ingress/transform
pipeline work, not milestone 2.

## Context

Real source APIs change shape over time (the motivating case is the Riksdag
API). The processor for the old shape needs to keep running against
documents acquired under it, while a new processor handles documents
acquired after the change — two valid transformations, partitioned by when
their inputs were acquired, not a single transformation with internal
branching.

`Transformation` (decision [[0004-execution-split-from-transformation]])
currently carries `implementation_ref`, the git commit hash of the module's
repository *at registration time* — but not the module's path or name. Two
registrations of the same logical processor at different commits produce
two `Transformation` records with nothing recorded that links them. Git
ancestry between commits can establish "this is a later version of that,"
but only once something records which module a commit belongs to.

A related but distinct case surfaced in discussion: patching an existing
transformation (e.g. adding a missed text-replacement rule) and wanting to
scope the rebuild to the documents it actually affects. That is retroactive
— decided by rerunning and observing which outputs' hashes change, then
propagating — not a registration-time partition. It belongs to milestone 3
(staleness/rebuild propagation) and is deliberately not addressed by this
decision, despite both being describable as "a transformation and a date
range."

Ad hoc imports (a forgotten dataset, a collaborator's files) are a further
distinct case: they are not a new version of anything already registered,
and specify their own date window standalone, with no supersession logic
involved.

**Not API-specific.** Nothing about this mechanism is particular to APIs —
a scraper keyed to a website's layout supersedes the same way when the
layout changes, and the window is keyed on the *content's* acquisition
time regardless of who did the acquiring: a partner's crawl, interpretable
only by the scraper version current when they crawled it, selects the
right registered version the same way a live fetch of our own would,
because both are just "acquired at time T." See the continuous-corpus
motivation in `AGENTS.md`.

## Decision

- `Transformation` gains a `module_path` field (or equivalent module
  identifier) recorded at registration, alongside the existing
  `implementation_ref`. This is what makes two registrations recognizable
  as the same family — `implementation_ref` alone (a repo-wide commit hash)
  is not enough.
- `Transformation` gains `valid_from` / `valid_until` (acquisition-time
  bounds, open-ended by default) as core fields, not entries inside
  `composed_config`. They must be enforceable uniformly across every
  transformation regardless of that transformation's own config schema.
- Registration behaviour: when a new registration's `module_path` matches
  an existing `Transformation` and its `implementation_ref` is a git
  descendant of the existing one's, the existing record's `valid_until`
  defaults to the new registration's date — unless it was already set
  explicitly (never overwritten) or the registration call supplies an
  explicit boundary itself. Guessing a boundary the registrant didn't state
  is the registrant's problem, not something the tool infers further.
- Enforcement is a **guard rail at run time**, not automatic dispatch:
  `run` refuses to apply a transformation to inputs whose acquisition time
  falls outside its `[valid_from, valid_until)` window. Selecting which
  transformation to run for a given batch remains the caller's decision —
  "run what you are told to run" (milestone 2) still holds. Automatic
  selection by date is noted as a plausible future addition, not built now.
- The window is keyed on **acquisition time**, not document time — this is
  about which processor was in effect when the document was fetched, not
  what the document's content is about. (Document-time-scoped rebuild
  narrowing is the separate, parked, milestone-3 case above.)
- Ad hoc imports set `valid_from`/`valid_until` directly at registration
  with no `module_path` match to any prior transformation, and no
  supersession logic applies to them.
- Triggering re-registration automatically from a git hook
  (`post-commit`/`post-merge` invoking `corpus-build register` when a
  tracked module changes) is workflow tooling, not a model concept, and is
  not part of this decision.

## Consequences

- `Transformation` serialization (both backends) needs updating for the
  three new fields; existing records without them read as
  family-less/unbounded, not as an error.
- Registration needs a way to find existing transformations sharing a
  `module_path` — an addition to registration logic, not necessarily to the
  `StorageBackend` contract itself (filtering `all_transformations()` may
  be sufficient; a dedicated backend query is an implementation choice, not
  decided here).
- Milestone 3's staleness work still owns the retroactive-rescoping case;
  this decision does not attempt to solve it and should not be cited as
  though it does.
