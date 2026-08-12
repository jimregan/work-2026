# Versioned artifact graphs for continuously growing web speech corpora

Draft for the Web as a Corpus workshop

## Abstract

Web speech corpora are commonly released as snapshots, although their sources
continue to change. Maintaining a growing corpus is not simply a matter of
fetching more documents: APIs are deprecated, page layouts change, redirects
collapse distinct locators, and material received from partners may only be
interpretable by an older extractor. Reprocessing also produces several valid
representations of the same source, including alternative segmentations,
transcripts, and uncertain correspondences. We present a corpus-building model
in which a corpus is a directed acyclic graph of UUID-identified artifacts
connected by versioned transformations and executions. Content hashes remain
independent integrity and deduplication keys. Identity is
defined per representational layer rather than globally; acquisition,
document, and processing times remain distinct; fetch specifications, network
responses, classifications, and correspondences are first-class artifacts.
This design preserves raw evidence and competing interpretations while making
reproducibility properties queryable. A Python prototype provides pluggable
filesystem and SQLite metadata stores, containerized and code-versioned
transformations, provenance queries, explicit acquisition outcomes, WARC
writing, and function-based execution records. Initial tests with archived
Swedish parliamentary API responses and a second single-URL source support the
generality of the graph model, while also exposing unfinished work in temporal
transformation selection and end-to-end live-source validation. We argue that
continuous web corpus construction is better treated as versioned evidence
management than as a linear processing pipeline.

## 1. Introduction

A web corpus is usually described as a collection obtained by crawling,
cleaning, and annotating web material. That description fits a release, but it
fits a continuously maintained corpus poorly. The web changes independently of
the corpus builder. An API may change schema or disappear; a scraper may fail
after a presentational redesign; a URL may redirect to a shared landing page;
and an archived response may arrive years after it was collected. Meanwhile,
speech processing creates multiple legitimate interpretations of one source:
schedule boundaries, acoustic boundaries, transcript-derived segments,
different ASR outputs, and alignments with different reliability thresholds.

A directory tree or linear pipeline makes these cases awkward. Both encourage
the assumption that an item occupies one place and has one current downstream
representation. Updating a stage then tends either to overwrite earlier work or
to produce an informal collection of versioned directories whose relationships
are external to the data.

We instead model a corpus as a graph of immutable, versioned artifacts linked
by recorded transformations and executions. The approach is motivated by a
speech corpus assembled from the live web, but the acquisition model applies to
other continuously collected web corpora. Our contributions are:

1. a layer-specific identity model that permits several valid representations
   of the same source without imposing a global canonical object;
2. an acquisition graph that separates request specification, observed
   response, and classification, retaining failures and redirects as evidence;
3. a provenance model that distinguishes a reusable transformation definition
   from each execution over concrete inputs;
4. explicit representation of temporal context and correspondence reliability;
5. a prototype evaluated through backend conformance, graph invariants,
   acquisition behaviour, and small heterogeneous ingress cases.

## 2. Why continuous corpora need a graph

### 2.1 Source evolution

Reproducibility is temporal for web data. The correct extractor for a response
is often the extractor that was valid when that response was acquired, not the
newest extractor available when it is processed. This is especially important
for delayed imports from collaborators or institutional archives. Replacing an
old extractor with a new one destroys the ability to interpret old material;
embedding all historical formats in one branching function obscures which
behaviour produced which result.

The model therefore treats transformation implementations as versioned records.
The intended temporal mechanism groups versions into module families and gives
each an acquisition-time validity interval. Selection remains explicit: the
interval is a guard against applying a known-inappropriate version rather than
an automatic scheduler. This portion of the prototype remains under
implementation and is not included in the experimental claims below.

### 2.2 Coexisting interpretations

Speech data has no single universally correct partition. A schedule partitions
a broadcast according to planned events; acoustic processing follows pauses or
speaker changes; a transcript follows textual units. Each is useful, and none
should overwrite another. The same is true of correspondence. A word alignment
may result from direct equality, text normalization, a correction list, or a
phonetic match. Consumers differ in which methods and quality flags they trust.

Consequently, correspondence is itself an artifact. Its metadata records how a
mapping was established and any relevant quality observations. Alternative
wordings or mappings remain distinct artifacts. Choosing a subset is a query,
not a destructive pipeline decision.

## 3. Data model

### 3.1 Artifacts and layer-specific identity

An artifact is the generic node type. Audio, responses, transcripts,
segmentations, alignments, metadata, fetch specifications, and correspondence
mappings differ in type and metadata but not in their status within the graph.
An artifact records its layer-typed identifier, artifact type, persistence
decision, optional content hash and locator, acquisition and document times,
metadata, and producing execution when one exists.

Identifiers are generated UUIDs scoped by layer. Equal content hashes in two
layers, or within one layer, do not assert that the artifacts are the same
object. Cross-layer or same-layer equivalence is expressed by correspondence,
with its own provenance. Materialized content has an independent SHA-256 digest
for verification and storage deduplication. Fetch specifications and responses
are separate UUID-identified artifacts rather than mutations of one node.

This scheme deduplicates content without collapsing graph history or inventing
a global “real-world object” identifier. If two executions produce identical
bytes, each may retain its own output artifact, metadata, and provenance while
both refer to the same content hash. This is essential for no-op and
byte-preserving transformations such as materializing a tool-specific view. If
two recordings capture the same event, they remain distinct until an explicit
correspondence relates them regardless of whether their bytes happen to be
equal.

Corpus metadata is intended to be versioned in Git. Artifact and logical batch
transactions receive UUIDs before commit; a Git commit hash identifies a
snapshot of the metadata graph rather than a domain object within it. This
avoids circular self-reference and keeps domain identity stable across amended
or rebased history.

### 3.2 Transformations and executions

A transformation records reusable behaviour: implementation reference,
instantiated configuration, and, where needed, a container image reference. It
contains no input or output identifiers. An execution records one application
of that transformation, including ordered and optionally role-labelled inputs,
outputs, timing, exit status, logs, resolved image digest, pinning state, and
transformation-specific execution metadata.

This separation permits an unexecuted transformation to exist without sentinel
fields, and it permits repeated executions over different inputs without
duplicating the transformation definition. Provenance edges are derived from
execution inputs and outputs. Multi-parent transformations are ordinary, and
storage rejects any execution that would introduce a cycle.

Two reproducibility modes are supported. Environment-sensitive transformations
may be containerized; their behavioural identity rests on the image reference
and instantiated configuration, while their source commit is also recorded.
Lightweight text transformations may be code-versioned; their identity rests on
a stable code reference and configuration. The current implementation includes
a direct Python-function runner and a separate Docker execution path.

### 3.3 Persistence and time

Persistence is an explicit property of an artifact, not a consequence of graph
depth. A costly raw response may be durable, while a downstream resampling can
be ephemeral if it is cheap to reproduce. Metadata and content storage are also
separated. Backends store structured records and content locators; content
remains in ordinary files usable by standard tools.

Three times are kept distinct. Acquisition time describes when material was
obtained, document time describes the time represented or stated by the
document, and processing time belongs to an execution. The distinction permits
late-arriving old material to retain its historical context without pretending
that it was acquired or published when it happened to be imported.

## 4. Acquisition as recorded evidence

Acquisition is represented as:

> specification → response → classification

The specification records what was requested. The response records what came
back, including a successful body, a redirect, an HTTP error, or a transport
failure. Classification is a separate versioned judgement over the stored
response. This separation permits a response to be reclassified after policy
changes without repeating the network transaction.

Specification state is derived rather than mutated. “Pending,” “obtained,”
“gone,” or “retry later” are views over classifications referring to the
specification. A redirect may produce a new specification. Several locators
converging on one target do not thereby become globally identical: unrelated
URLs often share landing or error pages. When a collection has moved and an old
and new locator are asserted to denote the same object, that claim is stored as
a correspondence artifact and may later be revised.

Redirect following is configurable. When redirects are followed inside the
HTTP client, the stored response records the original and final locations, that
the chain was collapsed, and the number of elided hops. Explicit chain walking
stores each hop and applies depth and loop bounds. The prototype can serialize
individual responses or chains to WARC, preserving intermediate hops as
resource records.

Pre-collected material enters through batch import rather than masquerading as
a live fetch. The importer reconstructs the original request specification when
possible and takes acquisition time from the source file's modification time or
an explicit batch date. Provenance, rather than an ad hoc “imported” flag,
distinguishes this route from live acquisition.

## 5. Prototype

The prototype is a typed Python library and command-line interface. The core
model uses frozen dataclasses and standard-library types. A storage protocol is
implemented independently by a filesystem conventions backend and SQLite. Both
run against one conformance suite. The SQLite store contains no content BLOBs;
the filesystem backend stores JSON sidecars and leaves content in ordinary
files.

Registration composes Hydra configuration, validates it against a structured
schema, refuses a dirty git working tree, and records the source revision.
Container references support digest-only, tag-only, and combined tag-plus-
digest forms. Combined references fail if current resolution disagrees with the
recorded digest. Docker execution exposes inputs, configuration, and outputs
only through fixed bind mounts, sends an empty environment, and disables the
network unless explicitly enabled.

Repeated execution of the same transformation over identical inputs is checked
for output-count or content-hash differences. Such non-determinism is noisy by
default and requires an explicit configuration opt-out. Function-based
execution applies the same check and persists a full execution record.

## 6. Evaluation and experience

The current suite contains 141 passing tests and the source passes strict static
type checking. Tests cover layer identity, content hashing, multi-parent
provenance and cycle rejection, both storage backends through a common
conformance suite, transformation registration, image resolution, real Docker
execution where a daemon is available, function execution, acquisition against
a local HTTP server, redirect loops, WARC round trips, batch import, extraction,
and correspondence-producing CTM alignment.

The motivating material is an archive of approximately 9,986 JSON documents
from an older Swedish parliament video API. A document includes a debate
identifier, debate date, downloadable media URL, and speaker-level timing and
text. The prototype imports old responses without assigning the import time as
their acquisition time and extracts media specifications from the archived API
shape. These components have focused fixture-based tests; a fully persisted
old-versus-current API case is still required before reporting an end-to-end
live-source result.

A second topology used a single hand-supplied Wikimedia/Wiktionary recording
and a stub ASR transformation in the same store as the parliamentary examples.
The same registered fetch definition was reusable across sources, a bare
specification worked as an execution-less graph root, and generic provenance
walking required no source-specific logic. This small probe also illustrated a
practical acquisition issue: server acceptance depended on request-header
policy. The fetch mechanism therefore accepts explicit headers rather than
embedding one global user-agent assumption.

Experience with parliamentary speech alignment further motivates queryable
correspondence. Word-level ASR, phonetic ASR, and an edited official transcript
can disagree in ways that cannot be resolved from either ASR/reference pair
alone. A token that appears to be a simple substitution may combine a
disfluency with a correctly spoken word omitted by the edited transcript. This
requires separable reconciliation, correction, paraphrase-acceptance, and
disfluency-splitting transformations while retaining their inputs. The graph
model accommodates that history; the full processing pipeline is future work.

## 7. Limitations and ongoing work

The prototype now generates UUIDv4 artifact IDs independently of content
hashes, and batch import assigns a shared UUID to each logical batch. Legacy
explicit identifiers remain deserializable during the design phase, but new
artifact-creation paths and examples use UUIDs. No recovery is possible for
distinctions already erased by the earlier content-hash identity model because
those missing nodes were never recorded.

The prototype does not yet implement acquisition-time validity windows or git-
ancestry supersession between transformation family members. Acquisition
functions exist, but several still need runner-compatible wrappers before every
fetch, classification, import, and extraction is represented by a persisted
execution. The planned human-readable, date- and hash-sharded content placement
convention is also not implemented. A live current-API parliamentary example
and an asserted old/current correspondence remain to be captured.

Configuration hierarchy storage has evolved during implementation. The
composed configuration is authoritative, while hierarchy information is
currently optional; detecting drift by recomposition remains missing. Query
support currently emphasizes provenance ancestry rather than rich metadata
filters over correspondence reliability. Staleness and rebuild planning are
deliberately deferred until transformation temporal versioning and collection
behaviour have been validated with real data.

## 8. Discussion

The main benefit of the graph is not novelty of storage technology. It is the
placement of judgement. Fetching records observations; classification records a
revisable interpretation of those observations; correspondence records an
explicit, revisable relationship; and consumer trust is resolved at query time.
This prevents early pipeline policy from deleting evidence needed by later
users.

The design also changes the meaning of reproducibility. A container digest is
appropriate when environment and executable behaviour are inseparable, but it
is unnecessary overhead for a deterministic text transformation whose code
revision is sufficient. Conversely, a source commit alone is inadequate for a
GPU model or codec-sensitive acoustic transform. Recording both cases under one
transformation/execution model permits provenance questions to expose the
difference rather than pretending every stage has the same reproducibility
surface.

For web-as-corpus research, retaining failed transactions is particularly
important. A 404, 503, redirect, and successful response are all observations
of a changing web. Discarding the first three turns acquisition policy into
missing data and makes later auditing impossible. Storing response evidence and
versioning classification policy makes source failure and policy error
distinguishable.

## 9. Conclusion

Continuously growing web speech corpora require more than repeatable scripts
over a directory tree. They require stable evidence, explicit temporal context,
versioned behaviour, and room for competing interpretations. Modelling the
corpus as a graph of UUID-identified artifacts, independently hashed content,
and recorded executions makes these properties structural. The prototype demonstrates the core graph,
storage, execution, and acquisition mechanisms across heterogeneous source
shapes. The remaining work is concrete rather than conceptual: complete
temporal transformation families, persist end-to-end acquisition executions,
and validate an archived/current source pair. These steps will test whether the
model can support a corpus that continues to grow while the web beneath it
changes.

## Notes for submission preparation

The final submission should add citations for content-addressable storage,
workflow provenance, WARC/web archiving, data versioning, and prior continuous
or monitor corpora; replace repository-specific implementation language with a
release identifier; report the exact archived sample used in the experiment;
and add a compact graph figure showing specification, response,
classification, transformation, execution, and correspondence nodes. Claims
about temporal family selection and live API evolution should remain framed as
ongoing work until the implementation-audit gaps are closed.
