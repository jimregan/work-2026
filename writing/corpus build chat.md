
ok. I've had another chat on this project, which took in the storage:

Right. The motivation is not a full off the shelf system for managing corpora, it's the basic underpinnings of such a system, that by itself can track provenance; when you notice one particular subset has an error, you rebuild from that point, with just the affected files; if there are 9000 files and 10 were incorporated later with a bad version of the script, we want to run over just those 10

---

That narrows the scope importantly. `corpus-build` is not intended to be a complete corpus-management product. It provides the provenance and dependency substrate on which such a system could be built. I would add this to the handoff: Scope and primary motivation `corpus-build` supplies the basic machinery for:

- recording immutable or versioned artifacts;
- recording exactly which transformation execution produced each artifact;
- identifying the inputs, code version, configuration, and relevant environment;
- tracing dependencies in both directions;
- finding outputs affected by a faulty or superseded execution;
- rebuilding only the affected portion of a corpus.

It does not need to provide every corpus workflow, annotation interface, scheduler, or distribution facility out of the box. The central operational example is: A corpus contains 9,000 files. Ten were added later using a defective version of a processing script. The graph must identify those ten executions and their affected descendants, then schedule only the necessary replacements—not rebuild all 9,000 files. This implies several concrete requirements:

- Transformation and execution identity must be separate.
- Every output must retain its exact input and execution provenance.
- Code, configuration, and environment versions must be queryable.
- Reverse dependency traversal must identify affected descendants.
- Rebuild planning must operate at individual-artifact or collection-member granularity.
- Correct existing outputs must remain reusable.
- Replacement outputs must coexist with old outputs until downstream differences have been processed.
- A collection update must not automatically invalidate every member.

That last point is especially important. The graph needs to distinguish member-wise operations from genuinely collection-wide operations:

- A `map`-like transformation over 9,000 independent files should rebuild only the ten affected members.
- An aggregate whose result depends on the entire collection may need rebuilding once after those ten replacements.
- Downstream member-wise transformations should likewise run only for descendants of the affected ten.

So the minimum useful staleness operation is approximately:

```
find executions using bad script version
    → find their output artifacts
    → traverse affected descendants
    → exclude nodes still valid under identical inputs
    → produce minimal rebuild graph
    → execute replacements
```

The multiple targets, selectors, annotation policies, and schedules exercise this substrate. They should not turn the project into an attempt to ship a comprehensive corpus-management platform. The core contribution is fine-grained, provenance-driven recomputation over a generic artifact graph.

---

Right; I think so far the specification has only had one direction for some operations, with the ability at the end of a phase to walk the graph with back tracking to add the backward links - I think in hindsight some operations are atomic enough that they can write in both directions between child and parent

---

That sounds right, with one important modelling distinction: the provenance relationship should exist once, but be traversable in both directions. For an atomic operation, completion can commit together:

- the execution node;
- its input-artifact relationships;
- its output-artifact relationships;
- indexes supporting parent-to-child and child-to-parent traversal.

Conceptually:

```
input artifacts → execution → output artifacts
```

From that single committed structure, the system can answer both:

- “What produced this artifact?”
- “What executions and artifacts depend on this artifact?”

If the backend stores explicit adjacency on both nodes, those are two physical representations of one logical relationship. They must be updated atomically or treated as a canonical edge plus a rebuildable reverse index; otherwise interrupted writes can leave contradictory provenance. A useful split would be: Atomic operations When inputs and outputs are known within one controlled commit:

1. Write or verify the output artifact.
2. Create the execution and its input/output edges.
3. Update both traversal directions.
4. Make the completed execution visible.

No later backlink pass is needed. Non-atomic or imported operations When evidence arrives in stages, comes from an external process, or predates the graph:

1. Record the available canonical relationship.
2. Leave the execution incomplete or mark its indexing state explicitly.
3. Run reconciliation to discover or construct missing reverse indexes.
4. Publish it as fully traversable once consistent.

The end-of-phase backtracking pass then remains useful as:

- an importer for legacy or externally generated state;
- a repair and integrity-check mechanism;
- a way to rebuild derived reverse indexes;
- support for operations whose outputs cannot be committed atomically.

It should not be required for every ordinary transformation if the transformation already knows its inputs and outputs. This matters directly for the “10 bad files out of 9,000” case. Immediate reverse traversal lets the system start at the defective execution version and find affected descendants without waiting for a corpus-wide backlink phase. The graph becomes incrementally usable after each atomic execution. I would phrase the requirement as: Completed local executions should atomically publish bidirectionally traversable provenance. Deferred graph backtracking remains available for staged, imported, or recoverable operations, but is not the default mechanism for establishing reverse dependencies. The implementation should still nominate one canonical representation—preferably first-class execution/input/output associations—and regard any duplicated parent or child lists as indexes rather than independently authoritative links.

---

Having multiple views was always a core part of the vision; something like a git checkout, but more likely a separate subdirectory with hard links

---

Yes—“view” here also has a concrete filesystem meaning, not merely a query over metadata. A graph snapshot, target build, or selected subcorpus should be materializable as a checkout-like directory:

```
views/
  spoken-swedish-2026/
  gesture-person-x-2026/
  sign-language-candidates/
```

Each directory is backed by a versioned manifest describing:

- the exact artifact identities included;
- their paths within the view;
- the target and selector that produced it;
- policy and transformation versions;
- any dependency builds used.

The files can be hard links to the content-addressed store, so several views can expose the same media without duplicating it:

```
content store artifact
    ├── spoken-swedish-2026/session-123/video.mp4
    └── gesture-person-x-2026/session-123/video.mp4
```

The graph remains authoritative. The directory is a materialization of a manifest node—closer to a Git checkout than to the repository itself. A few consequences should be explicit:

- Materialized files must be treated as immutable. Editing a hard link in place would alter the shared inode.
- Views should therefore be read-only where practical, or edits should use copy-on-write/import-as-new-artifact semantics.
- Different views may assign different paths or filenames to the same underlying artifact.
- Updating a view should materialize a new manifest generation and then move a `current` reference, rather than silently changing the meaning of the existing generation.
- Deleting a view removes one materialization, not the underlying artifact while graph nodes still reference it.
- Garbage collection should use graph reachability, not filesystem link counts alone.
- A view can be incomplete or partial and later gain a successor generation as more target members finish.

This also provides the user-facing expression of multiple corpus targets:

```
graph target + selector + policy versions
    → immutable manifest
    → hard-linked filesystem checkout
```

The “10 bad files” rebuild would create ten replacement artifacts and a successor manifest. Most entries in the new checkout would hard-link the same existing content; only the affected paths would point to new artifacts. That makes the provenance model visible as an efficient, practical corpus workspace rather than only as database machinery.

---

Right; input and output should be guaranteed to be different

---

Yes. Every execution should receive physically separate input and output locations. An output must never alias an input path or inode, even when the input came from a hard-linked view. A safe execution lifecycle is:

1. Materialize inputs read-only.
2. Allocate a fresh output staging directory on separate inodes.
3. Run the tool with distinct input and output paths.
4. Validate and hash completed outputs.
5. Import them into the content store.
6. Commit the execution and provenance edges atomically.
7. Materialize them into output views only afterward.

The executor should reject:

- identical input and output paths;
- output paths nested inside input directories where mutation is possible;
- input/output files resolving to the same inode;
- tools configured for in-place modification.

There is one identity nuance: a transformation may produce bytes identical to its input. The execution-time files must still be separate. After completion, content-addressed storage may deduplicate the bytes, but the provenance must still record that an execution occurred and produced an output in its declared semantic layer. Physical deduplication must not turn the operation into an in-place edit or erase its execution history. So the invariant is: Inputs are immutable, outputs are created independently, and no execution can modify the storage backing an input—even if content deduplication later makes identical bytes share storage.

---

A transformation that is effectively a no-op should be metadata only The key idea was that the node ID for concrete contents should be a hash of the contents, this makes the no-op automatic in a way

---

Right. I introduced an unnecessary distinction between “input artifact” and “output artifact.” If concrete-content node identity is derived from the content hash, then identical output bytes resolve to the existing input node automatically. A no-op transformation cannot create a second concrete-content node merely to preserve the appearance of an output. Instead:

```
content node H
    → execution
    → metadata/annotation: operation completed; content unchanged
```

The execution records:

- the transformation and its version;
- the input content node;
- parameters and environment;
- the fact that the computed result hashed to `H`;
- any annotations or validation results produced.

The temporary output must still be physically separate while the tool runs, so the input remains immutable. After hashing:

- if the hash differs, import a new concrete-content node;
- if the hash matches, discard the temporary duplicate and emit metadata only.

This avoids artificial duplicate artifacts and makes no-op detection intrinsic to content addressing. It also means downstream requirements can be satisfied by the original content plus the execution metadata proving that the transformation was applied. The invariant becomes: Different content creates a new content node. Identical content reuses the existing node; only the new provenance and annotations are added.

---

For later stages it's just the modification of a pointer, for metadata level queries the contents are not inspected

---

Exactly. Once concrete content has entered the graph and been identified by its hash, later metadata-level operations do not recreate, rehash, or inspect that content. They operate on references:

```
logical role/path
    → metadata and policy evaluation
    → selected content-node ID
```

For example, a policy deciding that supplied subtitles satisfy `silver_text` changes the binding from that logical role to an already-existing subtitle content node. Another policy might point the same role to a Whisper-derived node. The text itself need not be opened during policy evaluation; its provenance annotations are sufficient. The same applies when building a view:

- resolve target membership and labels from metadata;
- construct a new versioned manifest containing content-node pointers;
- materialize those pointers as hard links;
- leave the content store untouched.

If a metadata operation resolves to the same pointer as before, its only possible output is metadata recording the evaluation. If it resolves to a different existing node, the change is just a new pointer or manifest entry. So there are two distinct classes of work:

- Content transformations: read bytes and may produce a new content hash.
- Metadata transformations: inspect graph metadata and produce annotations, selections, or pointer bindings without reading content.

That separation is essential for cheap policy changes, subcorpus selection, view rebuilding, and fine-grained recovery. Most later-stage restructuring should be graph and manifest work, not corpus-file processing.

---

The idea of using hard links is that intermediate contents might be "garage collected" but as long as one hard link remains, the contents will remain For non posix file systems/other storage backends, this will need some implementation details, but that becomes a problem for the implementation of the backend

---

Yes. On a POSIX backend, hard links provide the physical retention mechanism naturally: removing the content-store pathname does not remove the inode while any materialized view still links to it. That means graph-level garbage collection can release the backend’s canonical reference without forcing deletion of content still used by a checkout. The filesystem performs the final reference counting. The generic layer should therefore specify a backend contract, not hard-link semantics directly. Something like:

- store content by ID;
- resolve a content ID to readable content;
- materialize content into a view;
- release a managed reference;
- report or preserve externally materialized references;
- collect content when the backend considers it unreferenced.

A POSIX implementation can use hard links. Other backends might use:

- explicit reference counts;
- object-store manifests;
- copy-on-write clones;
- leases or retention markers;
- ordinary copies with backend-managed lifecycle records.

One detail the POSIX backend must handle is discoverability. If the canonical content-store link is removed but a view link survives, the bytes still exist, but lookup by content hash may lose its usual pathname. The backend must either retain a locator for an extant link, recreate the canonical link before releasing it, or distinguish “remove from graph-managed storage” from “forget how to resolve this content ID.” So the generic invariant is: Content may be physically collected only according to the retention semantics of its storage backend; materialized views must remain valid independently of the graph’s canonical storage reference. How POSIX hard links, object-store references, or another storage system uphold that invariant is entirely a backend implementation concern.

---

"forget how to resolve" is a non goal; if the intermediates are necessary, it should be possible to recreate by walking the graph to the appropriate execution node and executing it

---

Right—maintaining resolution to every surviving hard link is unnecessary. The graph preserves the recipe, not a guarantee that every derived payload remains available in the managed content store. A concrete-content node can continue to exist after its stored payload has been evicted:

```
content identity and provenance: retained
managed payload: absent
materialized hard link elsewhere: possibly still present
```

If that content is requested again:

1. Check whether the backend still has a managed payload.
2. If not, walk to its producing execution.
3. Ensure that execution’s inputs are available, recursively rebuilding them if necessary.
4. Re-execute the transformation.
5. Verify that the result has the expected content hash.
6. Restore the managed payload.

Any hard-linked checkout that kept the inode alive remains usable, but rediscovering that inode for cache reuse is not a requirement. It is simply an unmanaged surviving materialization. This makes intermediate content a rebuildable cache. The graph retains:

- expected content identity;
- producing execution;
- transformation version and configuration;
- input dependencies;
- annotations and downstream relationships.

The backend may discard the bytes according to its own policy. The important exception is irreproducible source material. Acquisition artifacts that cannot reliably be fetched again need a retention policy different from reproducible intermediates. That is a storage-policy distinction, not a change to the graph model. So the correct invariant is: Derived content may be evicted while its identity and derivation remain in the graph. Availability is restored by replaying the recorded execution chain and verifying the expected hash.

---

I've had metadata in mind as something that could be git versioned, but I'm feeling more like something log structured is more appropriate now

---

A log-structured metadata store fits the model better than treating the live graph as a Git working tree. Most graph changes are naturally append-only events:

- artifact observed;
- execution started or completed;
- input/output edge added;
- annotation emitted;
- selection snapshot produced;
- pointer or target head advanced;
- payload evicted or restored;
- record superseded.

That gives atomic publication, efficient incremental indexing, concurrent execution, and a complete audit trail without manufacturing Git commits for every machine operation. A likely hybrid is: Log-structured operational metadata Use an append-only log as the authoritative history of graph events. Derived indexes provide:

- artifact lookup by hash;
- producer and consumer traversal;
- annotation queries;
- current target heads;
- execution state;
- payload availability;
- date-range resolution.

Indexes are rebuildable by replaying the log. Checkpoints make replay practical, and compaction can discard obsolete physical representations without changing graph semantics. Git-versioned human-authored definitions Git still makes sense for things people intentionally edit and review:

- target definitions;
- selection and interpretation policies;
- transformation registrations;
- annotation schemas;
- date-bounded API mappings;
- configuration templates.

An execution records the exact Git identity of the relevant definition or code, but its operational events go into the log. Exportable graph snapshots For reproducibility or publication, a graph scope can be exported as a stable manifest or metadata snapshot and optionally committed to Git. Git becomes a distribution and review format, not the transaction engine. This split also handles “replacement” cleanly. The log records that a newer node or pointer supersedes an older one. Nothing is rewritten in place during ordinary operation. Backend compaction may later remove obsolete records that have no dependants. The key design requirement is that a completed atomic execution should append one indivisible event batch containing its execution state, edges, outputs, and annotations. Readers should see either the whole completed execution or none of it. So I would now frame it as: The graph is logically persistent and mostly immutable; an append-only event log records its evolution, rebuildable indexes expose its current structure, and Git versions the human-authored code and policies that give executions their meaning.

---

I'm feeling it now, it doesn't need to be implemented now

---

Agreed. It is an architectural direction, not a current implementation requirement. For now, the important thing is to avoid choices that would prevent a later log-structured backend. The present storage interface only needs to preserve a few invariants:

- nodes and executions have stable identities;
- completed provenance is not silently mutated;
- graph relationships can be traversed in both directions;
- metadata updates can be represented as additions or supersessions;
- current pointers are distinct from immutable history;
- derived indexes are not treated as the sole source of truth.

The current backend can remain much simpler. Git-versioned metadata, files, or a conventional database are adequate while the graph semantics are being established. I would record the log-structured approach as an open design direction or ADR note: Operational graph metadata may eventually use an append-only log with rebuildable indexes. This is not required for the current implementation; current work should establish storage-neutral graph semantics and avoid depending on destructive in-place updates. That captures the insight without turning it into another prerequisite.

---

What I had in mind was that warc could be a first class medium - not just for ingest, where it's most natural, but for capturing the metadata

---

That makes the log-structured idea much more concrete: WARC could be the durable medium for both captured content and the metadata graph around it. WARC already supports:

- immutable records with stable record IDs;
- request, response, resource, metadata, conversion, and revisit records;
- digests for content identity;
- metadata records referring to content or other metadata records;
- recording a revisit without storing duplicate content again. Those capabilities align closely with acquisition, content addressing, no-op processing, and append-only annotation. [WARC 1.1 specification](https://iipc.github.io/warc-specifications/specifications/warc-format/warc-1.1/) A possible mapping is: | `corpus-build` concept | WARC representation | |---|---| | HTTP acquisition | request + response records | | Non-HTTP concrete artifact | resource record | | Derived concrete content | resource or conversion record | | Annotation | metadata record | | Execution provenance | metadata record with structured body | | Identical acquired content | revisit/reference semantics | | WARC-level description | warcinfo record | | Graph/index checkpoint | metadata or application-specific resource | This would let a tool append:

1. any newly produced concrete content;
2. an execution record identifying inputs, transformation, version, and configuration;
3. metadata/annotation records;
4. graph relationships required to connect them. The annotation scanner could then consume WARC records and build the query indexes discussed earlier. The WARC remains the durable record; the index is disposable. There is one boundary to preserve: standard `WARC-Refers-To` identifies a single referenced record, whereas a transformation execution may have several inputs and outputs. The complete graph relationship will therefore need a structured metadata body or a defined application profile rather than trying to encode the entire graph using standard WARC headers alone. Metadata records may themselves refer to other metadata records, so WARC does not prevent this extension. This also suggests a useful storage architecture:

```text
append-only WARC segments
    → annotation/graph scanner
    → rebuildable query indexes
    → target manifests and filesystem views
```

WARC files would not merely be intake packages. They could be the archival serialization of the graph’s evidence: acquisitions, execution records, annotations, and possibly derived content. POSIX hard-linked views would remain a separate backend materialization. I would keep this as a supported-medium direction rather than immediately making WARC the only metadata store. The near-term requirement is to ensure that graph records can be serialized append-only with stable IDs and explicit relationships. That preserves the option to make WARC a first-class graph backend once the node vocabulary has been demonstrated by the concrete workflows.

---

I think the thing going forward will be to consolidate on a dedicated graph library or database that is optimized for these operations, but that's an implementation detail

---

Yes. The semantic model should come first; the eventual graph library or database is a backend choice. The specification should define:

- node identities and concrete node types;
- execution/input/output relationships;
- immutable versus replaceable records;
- annotations and policy interpretation;
- collection membership and selection snapshots;
- target dependencies;
- traversal and affected-descendant queries;
- payload availability and rebuild semantics.

It should not commit the model to Git, WARC, a relational schema, or a particular graph database. Those can serve different implementation roles:

- a graph database for operational traversal and affected-subgraph queries;
- WARC for archival capture and interchange;
- content-addressed files for payload storage;
- hard-linked directories for materialized views;
- Git for human-authored policies and transformation definitions.

The critical requirement is portability of meaning: changing the graph backend must not change artifact identity, provenance, rebuild behaviour, or target semantics. For the coding-agent handoff, I would state: Treat storage technology as an implementation detail. Establish a storage-neutral graph contract and the queries required by the demonstrated workflows. A dedicated graph library or database is the likely operational backend, but choosing or optimizing it is not part of the immediate work unless the current implementation blocks the required semantics. That keeps the design grounded: prove the graph operations first, then choose infrastructure based on measured access patterns rather than designing around a hypothetical backend.

---

Like, having a basic implementation that uses a subset of rdf on a local schema would achieve the same goals

---

Exactly. A small, controlled RDF-style model over a local schema could provide the required semantics without prematurely choosing a production graph database. For example:

```
<artifact:sha256:…>  cb:producedBy   <execution:…>
<execution:…>        cb:used         <artifact:sha256:…>
<execution:…>        cb:transform    <transform:…>
<artifact:sha256:…>  cb:annotatedBy  <annotation:…>
<collection:…>       cb:member       <artifact:sha256:…>
<target-build:…>     cb:manifest     <collection:…>
```

The initial implementation only needs the subset required by real operations:

- stable subject identities;
- typed nodes;
- named predicates;
- literal metadata;
- graph insertion;
- forward and reverse traversal;
- filtering by transformation, version, date, or annotation;
- transitive ancestor and descendant queries;
- atomic publication of completed executions.

It need not become a Semantic Web project. There is no immediate need for public ontologies, general inference, ontology alignment, or arbitrary RDF reasoning. “RDF-shaped triples with a controlled local vocabulary” is sufficient. The local schema can give concrete semantics and validation to artifact, execution, annotation, collection, selector, and target nodes. The storage implementation could initially be a simple triple table plus indexes—or an RDF library—while preserving a migration path to a dedicated graph engine later. The success criterion is behavioural: Can the implementation represent the provenance graph, find the ten files produced by a bad script version, traverse their affected descendants, and construct the minimal rebuild set? If it can, the basic backend has achieved the goal. Whether the triples are later stored in SQLite, an RDF engine, or a property-graph database should not affect the graph model.

---

So one thing; I have an open question about multiple outputs: I think structuring them as multiple transformations that are later combined is ... maybe not the way to go, but an alternative implementation choice that doesn't require a whole bunch of machinery in place

---

Yes. Native multi-output transformations are useful, but they do not need to block the initial graph implementation. There are three viable representations.

1. Several independent transformations

```
input → transform A → output A
input → transform B → output B
```

This is appropriate when the outputs are logically independent and can genuinely be recomputed separately. A later transformation can combine them when needed. Its limitations are duplicated work and misleading provenance when one real tool invocation inherently produces both outputs. 2. One transformation producing an output bundle

```
input → execution → bundle manifest
                         ├── output A
                         └── output B
```

The execution has one formal output: a content-addressed manifest describing the produced files and their hashes. Downstream metadata-level projection operations select individual members. This avoids requiring native multi-output execution machinery while preserving the fact that the files were produced together. It fits the existing need for first-class collections and manifests. 3. Native multiple outputs from one execution

```
              ┌→ output A
input → execution
              └→ output B
```

This is the most direct model, especially when outputs are atomic as a group, but it requires execution/output cardinality, partial-failure rules, validation of the complete output set, and rebuild semantics for individual outputs. For the basic implementation, the bundle-manifest approach seems like the best escape hatch. It requires only:

- one ordinary transformation output;
- a manifest node;
- content hashes and logical names for members;
- metadata-level projection to obtain a member reference.

It avoids pretending that several separate transformations occurred when there was actually one tool invocation. It also avoids designing all the native multi-output machinery before a concrete case proves which semantics are required. The key is not to encode “one execution has exactly one meaningful artifact forever” into the graph contract. The first executor may support one formal output, with compound outputs represented through manifests. Native execution-to-many-output edges can be added later if bundle handling becomes awkward. A useful rule would be: Use separate transformations when results are independently computed. Use a bundle manifest when one atomic tool invocation produces several related files. Native multi-output execution remains an optional backend/executor capability rather than an initial prerequisite.

---

There do need to be transformations that can work on multiple inputs, however; that's unavoidable

---

Yes. Multiple inputs are fundamental and should be part of the graph model from the start. Alignment alone requires it:

```
ASR transcript ─────┐
                    ├→ alignment execution → correspondence
reference text ─────┘
```

Other unavoidable cases include:

- comparing two selection snapshots;
- combining speech and gesture streams;
- reconciling word-level and phonetic ASR;
- applying a delta to a previous collection;
- evaluating an artifact against annotations and a policy;
- aggregating members into a collection-level result.

An execution therefore needs an ordered or role-labelled input set, rather than a single parent:

```
execution inputs:
  hypothesis  → artifact H
  reference   → artifact R
  parameters  → artifact/config P
```

Role labels matter because many transformations are not symmetric. Swapping `reference` and `hypothesis` changes the meaning even when the input set is otherwise identical. Execution identity should include at least:

- transformation identity and version;
- each input role;
- artifact identity bound to each role;
- parameters and relevant environment.

For unordered inputs—such as aggregating a set of corpus members—the transformation schema can explicitly declare a collection-valued or order-insensitive role. Its identity calculation must canonicalize that collection. Ordered sequences must preserve order. This does not imply that native multiple outputs are also immediately required. The useful minimal asymmetry is: An execution may consume multiple role-labelled inputs but initially produces one formal output, which may itself be a bundle or collection manifest. That supports the real operations already identified while leaving output cardinality as an implementation choice.