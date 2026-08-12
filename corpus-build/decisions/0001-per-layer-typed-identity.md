# 1. Per-layer typed identity

## Status

Accepted.

## Context

Identity is layer-specific (see AGENTS.md): there is no global "same thing"
relation. Two ways to encode that in the type system were considered:

- A plain composite key: `ArtifactId(layer: str, local_id: str)`, with
  distinctness enforced only by comparing the `layer` field at runtime.
- Per-layer typed IDs: a `LayerId` base class with one subclass per layer
  (`AcquisitionId`, `TranscriptId`, ...), so identifiers from different
  layers are different Python types.

## Decision

Per-layer typed IDs. `corpus_build.model.identity.LayerId` is the base
class; each layer subclasses it and registers itself with
`register_layer_id_class` so storage backends can deserialize records back
into the right type.

## Consequences

- Two artifacts from different layers can never compare equal or collide in
  a set/dict key, even accidentally, without a runtime string comparison
  bug being possible.
- Adding a new layer means adding a new `LayerId` subclass, not just a new
  string constant. This is deliberate friction: it keeps "what layers
  exist" visible in the type system rather than scattered as string
  literals.
- `Transformation.input_ids` / `output_ids` are typed as heterogeneous
  tuples of the `LayerId` base class, since one transformation routinely
  spans layers (see decision on multi-parent transformations in AGENTS.md).
