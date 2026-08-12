# 8. Correspondence reliability is queryable, not pipeline-resolved

## Status

Accepted.

## Context

`AGENTS.md` already treats correspondence as a first-class artifact, and
the graph-over-hierarchy decision underlying the whole model exists partly
so that multiple, differently-reliable annotations over the same material
can coexist rather than forcing one canonical interpretation. That
principle was never stated explicitly, and milestone 4's real material
makes the gap concrete.

The motivating case: Riksdag session recordings, individual speeches
within a session, aligned against word-level ASR output (wav2vec2, chosen
for timestamp reliability). The alignment between an ASR word and the
corresponding official-transcript word varies in how it was established —
sometimes a direct 1:1 match, sometimes reached via an inverted phonetic
dictionary, a named correction list, or text normalization. Sometimes the
matched span is narrower than a sentence, sometimes wider. Sometimes
meta-speech noise (cough, breath, hesitation) gets phonetically matched to
something, and whether to include it is itself a judgement call. And
sometimes (cf. the `rixvox-2` corpus) the official transcript is a trimmed
version of what was actually said — e.g. the ASR hears "tack fru talman"
where the transcript only records "fru talman" — and both wordings need to
stay accessible rather than one silently overwriting the other.

Different consumers of the same corpus want different things here: one
wants only "gold" annotations with a direct correspondence to the official
record, another is happy to include normalized or corrected matches.
Deciding that is not this library's job to pre-empt.

Separately: "the same essential operation, different details" (e.g.
find-and-replace with a different correction list) needs no new mechanism
— Transformation identity (image digest + instantiated config) already
distinguishes these, and
[[0007-transformation-versioning-and-date-windows]]'s `module_path` already
groups same-module-different-config instances into a family where that
grouping is useful.

The actual alignment mechanisms (Smith-Waterman CTM alignment, phonetic
matching, wav2vec2-specific handling) exist already in a separate,
older repository (`~/Playing/sync_asr`) and are explicitly treated as a
black box to be wrapped later, not redesigned as part of this decision.

## Decision

- A correspondence artifact's `metadata` is where match-method and
  reliability properties live: e.g. how the match was established (direct,
  phonetic-dictionary inversion, a named correction list, normalization),
  and any transformation-specific quality flags (meta-speech noise
  included, span granularity relative to a sentence). No new field on
  `Artifact` or new `LayerId` subclass is needed — `CorrespondenceId` and
  its open `metadata` mapping already fit.
- When a matching transformation produces more than one legitimate variant
  over the same underlying span, both are represented as distinct
  artifacts and, where applicable, distinct correspondences — never
  collapsed into one canonical choice by the transformation. Content
  addressing already makes this free: two different texts are two
  different artifacts by construction.
- Resolving which correspondences to trust or include is a **query-time**
  decision made by whoever is consuming the corpus, not a pipeline-time
  one. `query.py` is deliberately thin today (pure provenance-ancestry
  walking, no metadata filtering) — extending it to filter by these
  properties is anticipated future work, not designed here, and this
  decision must not be read as blocking on that extension existing yet.
- The alignment mechanisms themselves remain out of scope: they are
  wrapped as a transformation when that work starts, with match-method
  properties on the correspondence artifacts they produce being the
  integration point, not their internal implementation.

## Consequences

- No schema change now. Future alignment-transformation work is expected
  to populate correspondence metadata with match-method information rather
  than silently picking one interpretation and discarding the rest.
- Query-surface filtering by these properties is real, expected future
  work, tracked here so it isn't lost, but explicitly not scoped or
  designed by this decision.
