# Milestone 1 — Artifact and provenance data model

Status: complete (superseded by milestone 2)

## Scope

- Artifact identity, typing, and content addressing
- Provenance edges (DAG, multi-parent)
- Transformation records as first-class objects
- Storage backend interface, with at least two implementations to prove it is
  real
- Query surface sufficient to answer "how was this artifact produced"

## Deliberately excluded

- Actual execution of transformations (containers, scheduling, retries)
- Ingest of real RixVox or broadcast data
- View and partition machinery beyond the minimum needed to test the model
- Timeline mapping and coordinate systems
- Any web UI

## Notes for later milestones

Transformation records exist in the model from this milestone onward but are
inert — the model must continue to represent a transformation that has never
been executed. Milestone 2 adds execution without changing that property.
