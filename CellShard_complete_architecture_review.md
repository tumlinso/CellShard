# CellShard architectural and implementation review

## Executive verdict

CellShard has a good storage spine, but it is not yet the distributed biological execution architecture that should be allowed to harden around Cellerator. The durable archive versus generated execution-pack split, immutable generations, publication discipline, and the recent opaque execution-payload direction are strong. The central runtime model is still too storage-shaped: a physical shard is asked to act as partition, transfer object, placement unit, GPU residency object, and scheduled work unit. Local multi-GPU staging exists, but a biology-aware, cost-aware, topology-aware distributed planner does not. Remote and multi-node material remains largely a plan rather than an execution system.

The correct target is not a larger file-format library and not a second numerical runtime. CellShard should become Cellerator's biological placement and residency plane. It should own semantic domains and partition maps, generated-artifact catalogs, physical extents, storage and replica locality, placement, transfer scheduling, residency, generation pinning, and static cross-domain routing. Cellerator should own operators, CP-BP packing, execution order, execution-image internals, kernels, local reductions, and device execution. Baseplane should own sequence kernels, interval and halo semantics, and event encoding under the Cellerator umbrella. NCCL, MPI, UCX, CUDA IPC, GPUDirect, filesystems, and object-store clients should move bytes and perform collectives.

The most important redesign is to stop using one word, shard, for six different objects. The system needs an explicit many-to-many hierarchy:

```text
biological domain and ownership partition
        -> execution projection or partition
        -> immutable storage extent
        -> dynamically coalesced transport batch
        -> host or GPU residency object
        -> scheduled work and retry unit
```

Only the first two are biological or computational. Only the third is persistent. The remaining three are runtime decisions.

## Reconstruction of the present system

The current repository is best described as an immutable biological archive and generated-pack system with local access, local GPU staging, compatibility numerical paths, and an emerging opaque Cellerator payload boundary. It is not yet a complete distributed runtime.

| Area | Present role | Hot execution | Architectural classification |
|---|---|---:|---|
| `.csh5` | Canonical, HDF5-backed biological archive, with assays represented separately and observations coordinated | No, except cold fallback access | Implemented and intentional |
| `.cspack` | Generated, single-assay execution artifact and cache product | Yes | Implemented and intentional, but its outer contract needs tightening |
| Opaque execution payload | Caller-produced execution image carried without CellShard interpreting its internal tile grammar | Emerging primary hot path | Implemented and strongly intentional |
| Compatibility sparse payloads | CellShard-readable sparse representations used by older adapters and compute paths | Yes or fallback | Implemented but transitional |
| `.cshard` | Partially implemented physical shard or standalone distribution artifact | Not a mature invariant | Redesign or remove |
| `.cspool` | Build, staging, cache, or publication machinery around generated artifacts | Operational, not numerical | Internalize; it should not be a durable public format |
| Header-only access adapters | Compile-time access across archive, pack, and compatibility payloads | Yes | Preserve narrowly; do not let them hide representation conversion |
| Local multi-GPU code | Places and stages local payloads onto several devices | Yes | Useful vertical slice, but transitional as a distributed architecture |
| Generation, cache identity, publication | Immutable generations and generated-artifact publication or lookup | Yes, as correctness substrate | Preserve and unify into one identity graph |
| Remote and multi-node execution | Design and parking-lot material, without a complete worker, lease, transport, and failure model | No | Planned but absent |
| Masking, reduction, and compatibility compute | Numerical operations retained in CellShard for older payloads | Yes | Move to Cellerator, then delete compatibility implementations |
| Python lazy views | Convenient archive exploration without eager full materialization | No | Preserve as an archive API, not as a runtime abstraction |

The important current data path is conceptually:

```text
immutable .csh5 generation
        -> pack materialization and publication
        -> .cspack compatibility payload or opaque Cellerator image
        -> host access or staging adapter
        -> device placement
        -> local Cellerator execution
```

The archive-to-pack transform is legitimate when it is a one-time generation of a reusable execution cache. It becomes self-inflicted when a Cellerator-native image is decoded into a CellShard matrix vocabulary, reordered canonically, and encoded again before execution. The opaque payload work is therefore not a cosmetic refactor. It is the correct ownership boundary and should become the only optimized path.

`.csh5` participates in persistence and cold preparation. `.cspack` participates in hot execution. `.cspool` should remain operational machinery. `.cshard` does not currently justify a separate durable contract. Compatibility payloads should have an explicit removal plan.

## What CellShard should ultimately be

CellShard should be a small biological distributed substrate with four primary responsibilities:

1. **Semantic ownership.** It identifies biological domains, subsets, partition maps, replicas, and sparse relations between independently partitioned domains.
2. **Artifact and extent ownership.** It maps semantic selections and Cellerator or Baseplane projection identities to immutable, checksummed, independently addressable byte extents.
3. **Placement and residency.** It decides which projection should reside on which node or GPU, when it should be fetched, what should be replicated, and when it can be evicted.
4. **Routing.** It compiles static biological structure into destination-grouped routes for sequence events, regulatory edges, halos, and other cross-partition traffic.

It should not own an operator language, tensor algebra, kernel scheduler, collective implementation, general fault-tolerant database, object store, filesystem, or cluster resource manager.

A useful one-sentence contract is:

> CellShard maps Cellerator and Baseplane requirements over biological domains to execution-ready immutable images, placements, routes, and readiness events.

## Ranked architectural risks

1. **The shard is overloaded.** Persistent extent, semantic ownership, execution partition, transfer unit, residency unit, and retry unit are currently too closely coupled. This is the largest risk because every later multi-node API would freeze the coupling.
2. **Payload ownership is split between CellShard and Cellerator.** Compatibility sparse layouts and numerical helpers make CellShard a partial numerical ecosystem. The opaque image direction should replace them.
3. **Bytes are being asked to represent work.** Byte-balanced placement is capacity balancing, not execution balancing. Equal bytes can contain very different nonzero counts, CP-BP tile occupancy, active-cell counts, RHS widths, sequence-event density, or communication cuts.
4. **Identity is not yet one coherent graph.** Archive generation, cache identity, semantic structure, execution order, geometry, target architecture, and physical content digest must be distinct and connected. External-only generation metadata is unsafe for distributed use.
5. **Runtime APIs are storage-shaped.** Format-specific adapters are becoming runtime semantics. Distributed code should consume storage-independent descriptors and handles.
6. **The public format family is too large.** Two user-visible lifetimes are justified: durable archive and disposable execution pack. A spool is an implementation detail. A shard need not be a file format.
7. **The local multi-GPU path is not a natural multi-node model yet.** It proves staging, but not ownership epochs, source selection, topology penalties, worker credits, failure retry, or sparse routing.
8. **Canonical order remains a tempting interoperability crutch.** It should not be used as distributed synchronization. Local packed order must survive until data exits the ecosystem.
9. **Host-driven stage-by-stage control can serialize the pipeline.** The runtime needs asynchronous range reads, bounded pinned pools, ready fences, and a prefetch horizon, while leaving fine-grained kernel orchestration to Cellerator.
10. **Generic distributed infrastructure could accrete in the repository.** Any implementation of transport, collectives, membership, or storage that lacks biology-specific value should be replaced by a provider interface to existing systems.

## Strong decisions to preserve

1. The separation between a durable semantic archive and disposable generated execution artifacts.
2. Immutable generations and publication rather than in-place mutation of hot artifacts.
3. The recent opaque execution-payload boundary.
4. Pointer-free, relocatable, checksummed image direction where present.
5. Header-only zero-cost access views at the final bound hot interface.
6. Single-assay execution packs as the default physical unit, coordinated through a higher-level manifest rather than a monolithic multi-assay file.
7. Local multi-GPU staging as an implementation proving ground.
8. Lazy archive access and delegation of generic caching to HDF5, memory mapping, and the operating system where those mechanisms are already adequate.

## A replacement for the current shard model

The word `shard` should cease to be a foundational type. It can remain informal or describe an immutable delivery object, but the public API should use specific types.

| Level | Meaning | Owner | Persistent | May overlap or replicate | Dynamically regrouped |
|---|---|---|---:|---:|---:|
| Biological domain | Cells, genes, regulatory elements, sequence coordinates, modules, states, samples, modalities | CellShard manifest | Yes | No | No |
| Biological ownership partition | The authoritative semantic subset and its static dependencies | CellShard | Usually | Yes, through replicas and halos | Infrequently |
| Execution projection | Cellerator or Baseplane native physical image for one operation, order, geometry, precision, and target | Cellerator or Baseplane produces; CellShard catalogs | Generated | Yes | Regenerated, not reinterpreted |
| Storage extent | Immutable checksummed byte range or object | CellShard | Yes | Yes through content-addressed reuse | No |
| Transport batch | One or more extents coalesced for a destination and deadline | CellShard runtime | No | No | Yes |
| Residency object | Host, pinned-host, or GPU binding with readiness and lease state | CellShard tracks; Cellerator supplies device allocation | No | Yes | Yes |
| Work item | Cellerator tile groups or Baseplane interval work, with retry identity | Cellerator execution plan | No | Sometimes speculative | Yes |

The mappings are many-to-many. One biological partition can have several architecture-specific projections. One projection can span several storage extents. One transport batch can combine extents from several projections. One residency object can serve many work items. A failed work item should not force refetching or invalidating an entire persistent partition.

## Biological partitioning model

CellShard should maintain independent partition maps for independent biological coordinate spaces. A universal matrix partition is not sufficient.

```text
SequenceCoordinateDomain
       | sequence-to-regulatory relation
RegulatoryElementDomain
       | regulatory-to-gene relation
GeneDomain
       | gene-state or expression relation
CellDomain
       | state-space and trajectory relations
CellStateDomain
```

Each domain can have several operation-specific partition maps. The same cell archive may have a sample-major map for independent inference, a state-local map for neighborhood operations, and a lineage-window map for trajectory work. These maps should reference the same canonical data and generate different disposable execution projections rather than copying the archive.

Partitioning should use a weighted biological hypergraph assembled from information supplied by both repositories. Cellerator contributes execution atoms, tile counts, operator family, RHS width, memory footprint, output volume, expected reuse, and splittable boundaries. CellShard contributes source location, current residency, worker capability, topology, immutable relations, replica candidates, and storage cost.

A useful objective is:

```text
minimize
    maximum normalized worker load
  + remote read cost
  + weighted cross-partition biological traffic
  + reload cost
  + memory opportunity cost of replicas
  + topology penalty
  + fragmentation and launch overhead

subject to
    GPU and host memory
    worker capability
    dependency and ownership constraints
    available storage and network paths
```

CellShard does not need to invent a graph partitioner. It should construct the biology-specific weighted graph or hypergraph and use a mature partitioner or a simple deterministic greedy solver initially. Its invention is the graph, the weights, the hierarchy, and the route compiler.

CP-BP should accept distributed ownership constraints or a cut penalty, but CellShard should not perform CP-BP packing. The preferred hierarchy is:

```text
node ownership region
    -> GPU ownership region
        -> independently optimized Cellerator image
            -> CTA and warp geometry
```

A single global hierarchical permutation can be generated for a homogeneous deployment, but it should not be required. Several locally optimized images preserve better kernel geometry and support heterogeneous GPUs. Lightweight global-to-local mappings and route tables replace canonicalization.

## Exact CellShard, Cellerator, and Baseplane ownership

| Concern | CellShard | Cellerator | Baseplane | External provider |
|---|---:|---:|---:|---:|
| Canonical biological archive | Owns | Reads semantically | Reads semantically | HDF5 or storage client supplies bytes |
| Domain and partition identities | Owns | References | References | No |
| Operator DAG and numerical semantics | No | Owns | Sequence operators under Cellerator umbrella | No |
| CP-BP packing and execution order | Supplies distributed constraints | Owns | No | No |
| Sequence interval and halo execution | Catalogs and places | Coordinates downstream | Owns | No |
| Opaque execution-image encoding and validation | Stores and transports | Owns Cellerator images | Owns sequence images | No |
| Artifact manifest and checksummed extents | Owns | Produces descriptors | Produces descriptors | Storage persists bytes |
| Placement, replica, prefetch, and residency plan | Owns | Supplies requirements and consumes handles | Supplies requirements and consumes handles | No |
| GPU allocator and kernel workspace | Tracks leases only | Owns or supplies allocator callbacks | Uses Cellerator device runtime | CUDA allocator |
| Local numerical reduction | No | Owns | Owns sequence-local reductions | No |
| Distributed collective implementation | Selects group and route | Supplies buffers and operation | Supplies event buffers | NCCL, MPI, UCX |
| Work DAG and kernel scheduling | No | Owns | Participates through Cellerator | No |
| Dynamic worker leases and generation pins | Thin optional runtime | No | No | Optional caller/orchestrator integration |

The shared ABI can stay small. CellShard needs visible placement metadata, not CPK1 internals.

```cpp
struct DomainBinding {
    AxisRole role;
    DomainId domain;
    PartitionMapId map;
    PartitionId partition;
    OrderId order;
};

struct ProjectionKey {
    ProducerAbi producer;
    StructureId structure;
    GeometryId geometry;
    OperatorClass operation;
    ScalarEncoding encoding;
    TargetCapabilities target;
};

struct ImageDescriptor {
    ImageId image;
    std::span<const DomainBinding> domains;
    ProjectionKey projection;
    std::uint64_t file_bytes;
    std::uint64_t device_bytes;
    std::uint32_t alignment;
    ReuseClass reuse;
    Digest checksum;
    std::span<const ImageId> dependencies;
    std::span<const RouteTableId> routes;
};

struct DataRequirement {
    RequirementId id;
    std::span<const DomainSelection> selections;
    StructureId structure;
    ProjectionConstraints acceptable;
    AccessMode access;
    UseWindow use_window;
    ExecutionCostEnvelope cost;
};

struct ReadyImageLease {
    ImageDescriptor descriptor;
    DeviceSpan<std::byte> image;
    ReadyFence ready;
    ResidencyLease lease;
};
```

The interaction should look like this:

```cpp
auto requirements = prepared_execution.describe_requirements();
auto plan = cellshard.plan(snapshot, requirements, topology, residency);
auto leases = cellshard.prefetch(plan);
prepared_execution.bind(leases);       // no CSR, row rebuild, or canonicalization
prepared_execution.launch(streams);
```

Cellerator owns the overall execution plan. CellShard returns a placement and residency subplan. There must not be two schedulers.

## CSPACK and opaque execution images

CSPACK should be a transport-friendly execution-artifact envelope whose payloads remain producer-owned. CellShard should understand:

- semantic domains and partition ownership;
- structure, order, geometry, and image identities as opaque values;
- target architecture and ABI compatibility;
- file and device memory requirements;
- dependencies, reuse class, and route-table identities;
- alignment, offsets, lengths, and checksums.

It should not understand CP-BP tile grammar, sparse index encoding, quantization codebooks, or kernel dispatch metadata beyond opaque compatibility tags.

A pack should have an aligned header and independently addressable sections. The manifest may reference several files or objects, so a worker can range-read one image rather than fetch a monolith. Several projections may belong to one pack family, but they should remain independently addressable. The outer header and identity digest must travel with every image. A sidecar catalog may accelerate discovery, but it cannot be the sole source of generation identity.

A Cellerator image can be copied directly to GPU memory and rebound only if it is pointer-free, relocatable, aligned, and self-describing to Cellerator. The host-only pack manifest should not be copied merely because the image resides in the same file.

## Recommended format hierarchy

Only two public lifetimes are necessary.

### `.csh5`

Keep it as the durable semantic archive. Give this contract long stability. It should preserve domain identities, values, sparse relations, coordinates, observations, features, modalities, and enough provenance to regenerate future execution layouts. Blocked-ELL, Sliced-ELL, CP-BP tiles, target architecture, and local GPU order are derived data and should not be required canonical content.

### `.cspack`

Keep it as a disposable generated artifact. It can evolve aggressively because it is regenerable. Its stable outer envelope should contain identities, section addressing, checksums, capabilities, and dependencies. Its inner images are Cellerator or Baseplane owned.

### `.cshard`

Do not preserve it as an independent durable matrix format. If object stores or remote caches require standalone blobs, define a CSPACK-fragment profile using the same section and identity contract. A runtime partition or transfer batch does not need a file extension.

### `.cspool`

Keep only as an internal local spool, build transaction, cache namespace, or publication directory. It should have no durable interchange promise and should disappear from the public numerical API.

### Compatibility sparse payloads

Retain a read bridge for migration, then delete. A generic fallback can remain only if Cellerator owns its interpretation and pack generation.

The resulting hierarchy is:

```text
immutable archive generation (.csh5 or another semantic source)
    -> one or more pack families
        -> immutable pack blobs and independently addressable sections
            -> runtime extents, transfer batches, and residency leases
```

One archive generation may produce V100, H100, generic fallback, operation-specific, precision-specific, or partition-map-specific pack families. GPU architecture belongs to the projection key, not the archive generation. Placement for a particular topology belongs to a job manifest, not the pack bytes.

## Generation and consistency

Separate these identities explicitly:

- `ArchiveGenerationId`: semantic content and stable schema.
- `DomainId` and `PartitionMapId`: semantic universe and ownership mapping.
- `StructureId`: Cellerator semantic structure.
- `OrderId`: physical ordering of each bound domain.
- `GeometryId`: execution geometry or tile organization.
- `ImageId`: all producer, operation, precision, target, and projection choices.
- `ExtentId`: physical content digest.
- `SnapshotId`: the complete set of archive, pack-family, partition-map, and route-table identities pinned by one job.
- `PlacementEpochId`: runtime assignment only.

A new pack does not create a new biological generation. A new biological generation can reuse unchanged content-addressed extents. Packs are immutable once visible. Publication should write and validate temporary blobs, write and fsync the manifest, atomically publish a commit marker or manifest, and only then add the family to the catalog.

Workers pin one snapshot for a job or phase. Old and new generations may coexist. Mixed shards from incompatible snapshots are rejected before binding. Full checksums should be validated at construction and publication. Trusted local hot-cache reloads can validate the compact header and manifest digest, with per-extent checks on untrusted or remote sources, rather than repeatedly revalidating the whole pack.

## Multi-GPU architecture

The current local staging code should be retained as implementation material but rebased on storage-independent interfaces.

```text
Node runtime
  topology snapshot
  local NVMe and page-cache catalog
  bounded pinned-buffer pools per NUMA or NIC locality
  placement and residency manager
      GPU executor 0, Cellerator allocator and streams
      GPU executor 1, Cellerator allocator and streams
      ...
```

CellShard discovers topology at runtime and assigns affinities. It does not persist a preferred PCIe root, NVLink path, or device ordinal in `.csh5` or `.cspack`. It may persist target capability constraints such as producer ABI, required alignment, or supported compute capability.

The first redesigned local planner should use cost envelopes and relation cuts rather than equal bytes. Small common module dictionaries, order maps, and static regulatory metadata may be replicated. Large cell-specific state is normally partitioned. Interacting regions should prefer NVLink-connected devices. Device allocations should come from Cellerator's allocator or callback so Cellerator can account for workspaces and graphs; CellShard owns residency leases and eviction decisions.

## Multi-node architecture

A central coordinator is optional, not foundational.

For a static job, an immutable snapshot and deterministic placement plan let workers start without a service. For ephemeral workers, failures, dynamic credits, and rebalancing, use a thin coordinator that stores only worker capabilities, generation pins, placement epochs, work leases, and idempotent result tokens. It must not transport payload bytes, interpret images, or implement collectives.

Workers should pull by default because pull naturally expresses backpressure and available memory. Owner-push prefetch is an optimization when the plan and destination credits are known. Immutable data has authoritative storage sources and cache replicas, not mutable worker ownership. Mutable Cellerator state is job-scoped and checkpointed separately.

Worker disappearance expires work leases and creates a new placement epoch. A result is keyed by job, snapshot, work item, and attempt; duplicate execution is acceptable, duplicate commit is not. A failed source read retries another replica. A checksum failure quarantines that replica. GPU OOM causes regrouping or a fallback projection, not mutation of the pack.

## Streaming, prefetch, and residency

The target flow is:

```text
Cellerator future requirements
    -> CellShard resolves image and source
    -> asynchronous range read or object fetch
    -> page cache or bounded pinned staging, unless direct-to-device is selected
    -> GPU image sections become ready
    -> Cellerator binds and executes
    -> output is locally reduced or destination-bucketed
    -> residency is retained, demoted, or evicted
```

A shard is not simply loaded or unloaded. Useful states include absent, fetching, host-ready, partially device-ready, device-ready, bound, in-use, retained, evictable, and failed. Cellerator should declare which image sections are mandatory before binding and which work tiles can begin after partial delivery. CellShard should not guess streamability from opaque bytes.

The operating system should manage ordinary file pages. CellShard should explicitly manage only knowledge-rich levels: local pack cache, bounded pinned buffers, GPU residency, reuse leases, and source choice. L2, shared memory, and registers remain Cellerator concerns.

Generic LRU is insufficient. A useful retention score is proportional to expected near-term reuse multiplied by reload latency and regeneration cost, divided by memory bytes, with dependency fanout and network locality included. At minimum use explicit classes such as one-shot, phase-reused, epoch-reused, and pinned. Shared gene modules can remain resident while cell partitions rotate.

Prefetch depth should be time-based rather than a fixed shard count:

```text
prefetch horizon ~= read latency + transfer latency + safety margin
```

bounded by host and GPU credits. Cellerator's future requirement window and trajectory direction can improve the prediction.

## Theoretical copy paths

- Local `.cspack` to GPU has a theoretical minimum of one DMA movement from storage into HBM when direct storage is appropriate. The portable fallback uses storage to host memory and host to GPU, but still performs no representation conversion.
- Remote pack to GPU can use direct RDMA only when registration, alignment, size, and topology justify it. The fallback is destination pinned memory followed by H2D. CellShard selects the provider; it does not implement RDMA.
- `.csh5` to `.cspack` is an offline or cold-path representation transform. Repeated execution should never repeat it. One-shot work may use a direct generic projection builder, but it should remain a deliberate fallback.
- Baseplane sequence to remote Cellerator state should write compact destination-bucketed events, locally combine duplicates, and perform one sparse exchange to destination owners. It should not materialize or globally gather an event matrix.

Memory mapping, GDS, CUDA IPC, and RDMA are mechanisms, not architectural layers. Each belongs behind source and transfer capability interfaces.

## Communication and replication

Classify cross-partition dependencies before choosing a collective:

1. Small, immutable, high-fanout structures are replication candidates.
2. Large static structures with sparse cuts should be partitioned and routed.
3. Sparse dynamic activations should use destination-bucketed event exchange.
4. Neighbor dependencies should use explicit halo exchange.
5. Dense globally required reductions should use NCCL or MPI all-reduce or reduce-scatter.

A simple decision compares the memory opportunity cost of replicas with expected communication bytes, path cost, reuse count, and update cost. For immutable objects the update cost is zero. Explicit `ReplicaGroupId` and replica policy should be distinct from partition identity.

Typical defaults are:

- partition cells and large cell-state tensors;
- replicate compact gene dictionaries, TF metadata, and order maps;
- keep reference sequence once per node or memory-map it, with sparse sample-specific variant overlays;
- partition a regulatory graph by accumulation owner, often target gene or module;
- locally reduce edge contributions by destination before communication.

CellShard's role in collectives is to choose group membership, routing, buffer projection, and whether a sparse alternative is valid. NCCL, MPI, or UCX performs the operation.

## Sequence and Baseplane integration

Sequence and state require independent partition maps joined by static relations.

A sequence partition should contain an owned interval plus operation-specific left and right halos. Baseplane may compute over the halo, but event ownership is assigned by an anchor in the owned interval. This eliminates duplicates without a global deduplication pass. Long-range dependencies should be explicit relation extents rather than unbounded halos.

Reference sequence should be cached separately from sample variants. A reference image plus sparse variant overlay avoids redistributing whole genomes. Baseplane outputs globally meaningful coordinate or regulatory IDs. A compiled route table maps source-local events directly to destination regulatory or gene partition and destination-local slot.

```text
sequence partition: chr1 [10 Mb, 20 Mb), halo 256 bp
    Baseplane motif and grammar events
        -> route table grouped by RegulatoryPartitionId
            -> sparse network exchange
                -> Cellerator regulatory image in its local packed order
                    -> local gene-state accumulation
```

The route-table identity includes source and destination partition-map identities. If either map changes, the route table is regenerated. The static route skeleton can be precomputed; dynamic activity determines which entries actually produce messages.

## Multi-omics and trajectory locality

Keep physical execution packs single-assay by default. A pack-family manifest can bind RNA, ATAC, protein, and other projections to the same cell partition and assign an affinity group. A fused operator requests co-residency; modality-specific operators remain independently sharded and cached. This avoids a monolithic multi-assay pack while preserving coordinated observations.

Trajectory and state-space information should be separate indices or relations that influence partition and prefetch policies. They should not become mandatory ornaments in the archive format. Multiple partition indices can reference the same canonical assay extents. A state-neighborhood operation can request a state-local projection, while ordinary per-cell inference uses a sample-major projection.

Trajectory-window prefetch is promising when an operation follows directed developmental neighborhoods. It is not a universal cache policy and should be implemented only as a requirement hint from Cellerator.

## Minimal biological distributed IR

A small manifest-level IR is justified, but a general biological operator IR is not.

```cpp
struct DomainDesc {
    DomainId id;
    DomainKind kind;
    GenerationId generation;
    std::uint64_t cardinality;
};

struct PartitionDesc {
    PartitionId id;
    DomainId domain;
    PartitionMapId map;
    SelectionEncoding owned;
    std::span<const PartitionId> halos_or_replicas;
};

struct RelationDesc {
    RelationId id;
    DomainId source;
    DomainId destination;
    PayloadId destination_grouped_routes;
};

struct PlacementBinding {
    ImageId image;
    WorkerId worker;
    MemoryTier target;
    ReplicaGroupId replica_group;
};
```

This pays for itself because RNA, ATAC, sequence, regulatory elements, genes, and cells cannot safely share one partition map. It also gives route tables and execution-order mappings explicit identities. It must not contain kernels, expression trees, arbitrary tensors, or rich biological labels. Rich metadata remains in the archive; payload bytes remain opaque.

## Public API disposition

Durable public APIs should be limited to archive generations, pack catalogs, snapshot manifests, semantic domains and relations, placement requirements, residency handles, and ready fences.

Format-specific adapters, spool internals, physical shard files, and publication transactions should become internal. Python lazy views remain a user-facing archive convenience. Masking, reduction, quantization, packing, and interpretation of Cellerator images move to Cellerator. Baseplane owns sequence execution and event encoding. CellShard may expose storage-independent sources for raw extents and opaque images, not a taxonomy of matrix, bitplane, graph, and event payload classes unless a capability materially changes placement.

## Open-work disposition

| Thread | Recommendation | Reason |
|---|---|---|
| Opaque execution payload | Continue immediately | Correct Cellerator ownership boundary |
| Access-adapter refactor | Continue, then narrow | Preserve zero-cost bound views, remove format-polymorphic numerical algorithms |
| Local multi-GPU work | Redesign first, then continue | Reuse staging code under cost, topology, and residency abstractions |
| Distributed-pack plans | Merge into pack-family manifest and placement plan | Avoid a second distributed file semantics |
| Partial `.cshard` | Merge into CSPACK sections or delete | Transport and scheduling boundaries should be dynamic |
| `.cspool` | Internalize | Build and publication state is not a durable format |
| Quantized-pack work | Move image semantics to Cellerator | CellShard should see size, compatibility, and checksum only |
| Masking and reduction compatibility | Migrate, then delete | Numerical ownership belongs in Cellerator |
| GPU Direct Storage | Defer behind source capability API | Architecture first; direct path only for suitable aligned images |
| RDMA and multi-node transport | Use external provider after planner ABI | CellShard decides routes, not packet transport |
| Trajectory-native ideas | Redesign as partition and prefetch indices | Valuable planning signal, not a new base format |
| Sequence distribution | Move kernel and halo semantics to Baseplane; keep maps and routing in CellShard | Preserves one execution ecosystem |
| Old execution-layout assumptions | Delete after opaque migration | They freeze current Cellerator kernels into CellShard |
| Compatibility sparse payloads | Read-only migration bridge, then delete | Prevent a permanent parallel numerical vocabulary |

## Preserve, redesign, migrate, delete, defer

| Action | Items |
|---|---|
| Preserve | `.csh5` semantic archive; `.cspack` generated-artifact role; immutable generations; atomic publication; checksummed relocatable images; opaque payload; lazy archive views; sound local staging primitives |
| Redesign | shard and partition model; identity graph; runtime APIs; local placement; pack outer envelope; hot metadata; generation pinning; prefetch and residency; topology and cost model |
| Migrate | Cellerator-native image production and validation; masking and reductions; quantization and packing; compatibility readers; format-specific runtime callers |
| Delete | duplicate numerical formats; canonicalization hot paths; public spool semantics; independent `.cshard` schema without a unique lifetime; home-grown generic transport or collectives |
| Defer | GDS and RDMA optimization; sophisticated dynamic repartitioning; universal state-manifold policies; broad fault tolerance; heterogeneous auto-tuning, until the core descriptors and planner ABI are stable |

## Migration plan

### Phase 0: freeze the wrong surfaces

Do not add new distributed file formats or remote APIs. Inventory current public symbols, readers, writers, compatibility payloads, and callers. Add lifecycle labels and tests that preserve sound serialization and publication behavior.

### Phase 1: introduce the identity graph

Add domain, partition-map, structure, order, geometry, image, extent, snapshot, and placement-epoch identities. Embed correctness identity in the pack envelope while retaining sidecar catalogs for lookup. Existing `.cspack` readers can populate the new descriptors.

### Phase 2: make opaque images the primary path

Require Cellerator to produce and bind its native image. CellShard stores the bytes and visible descriptor. Keep compatibility sparse payloads behind an explicit bridge. Add a direct bind path in which a prepared execution consumes a `ReadyImageLease` without canonicalization.

### Phase 3: detach runtime from formats

Introduce `PayloadSource`, `ExtentRef`, `TransferPlan`, `ReadyFence`, and `ResidencyLease`. Implement `.cspack`, `.csh5`, local files, and existing shard or spool code as providers beneath them. A runtime plan should not name a file extension.

### Phase 4: rebase local multi-GPU support

Retain sound device staging and publication code. Replace byte-only assignment with cost envelopes, memory credits, topology, dependencies, and explicit replicas. Let Cellerator provide device allocations and execution futures.

### Phase 5: add the joint planning ABI

Cellerator emits requirements and static cost metadata. CellShard returns placement, source, prefetch, replica, route, and readiness actions. Feed ownership constraints and weighted cut information back into CP-BP, then generate independently optimized local images.

### Phase 6: add multi-node providers

Implement UCX, MPI, or another mature transport adapter, plus an optional thin lease coordinator. Do not add distributed transport before the runtime is storage-independent, or the transport API will freeze format assumptions.

### Phase 7: integrate sequence domains

Add sequence-owned intervals and halos, reference plus variant overlays, regulatory and gene partition maps, and compiled destination-grouped routes. Baseplane emits events directly into these routes.

### Phase 8: remove compatibility strata

Once all hot consumers use opaque images and new descriptors, delete old numerical adapters, canonicalization, public spool surfaces, and a standalone `.cshard` schema unless a distinct external contract has been demonstrated.

## Immediate priorities

1. Define the domain, partition-map, projection, extent, snapshot, and residency identities before adding any new distributed feature.
2. Make the opaque Cellerator execution image the only optimized CSPACK payload and move all image interpretation to Cellerator.
3. Replace format-specific runtime APIs with storage-independent requirement, source, placement, and lease interfaces.
4. Recast local multi-GPU code as the first implementation of the new planner and residency model.
5. Add Cellerator cost envelopes and CP-BP ownership constraints, replacing byte balance with weighted work and communication costs.
6. Collapse `.cshard` and `.cspool` into a CSPACK section profile and internal publication machinery unless code demonstrates a unique durable lifetime.
7. Establish snapshot pinning and embedded generation identity before remote workers can consume packs.
8. Only then add multi-node transport providers and Baseplane route compilation.

## Direct answers to the uncomfortable questions

| Question | Answer |
|---|---|
| Are shards defined by storage convenience? | Largely yes. The present physical unit is carrying more semantic and scheduling meaning than it should. |
| Is byte balance masquerading as load balance? | Yes wherever it is used as the primary placement objective. It is only a capacity proxy. |
| Does CellShard know too little biology? | In its placement logic, yes. It lacks first-class domain relations, weighted cuts, reuse, and operation cost. |
| Does it know too much about Cellerator? | In compatibility payloads and numerical helpers, yes. It should know identities and dependencies, not tile grammar. |
| Are current execution layouts stored too deeply? | They belong in disposable packs, not the durable archive. Any durable dependency on present sparse layouts should be removed. |
| Is CSPACK execution or transport? | It should be an execution-artifact envelope designed for efficient transport. That combination is good only with a strict outer-envelope and opaque-inner-image boundary. |
| Is `.csh5 -> .cspack -> GPU` optimal? | For repeated hot execution, yes as a cold-to-hot cache pipeline. It is not the only path for one-shot or streaming work, and it must never repeat representation conversion after pack generation. |
| Does `.cshard` solve a real need? | Not as a separate stable format today. Addressable CSPACK sections and dynamic transfer batches solve the architectural need more cleanly. |
| Are there too many formats? | Yes as public contracts. Keep archive and generated pack; internalize spool and collapse shard. |
| Are conversions caused by ownership drift? | Yes in the compatibility route. Cellerator-native images remove them. |
| Is canonical order reconstructed unnecessarily? | The compatibility model makes that possible and sometimes necessary. The target invariant is no canonicalization inside the ecosystem. |
| Are boundaries too rigid? | Yes if persistent extents determine placement, transport, work, or retry granularity. |
| Does local multi-GPU naturally extend to multi-node? | Not without the identity, source, topology, lease, and routing refactor. The staging primitives can extend; the current ownership model cannot simply be stretched. |
| Should scheduling live in CellShard? | Only placement, residency, prefetch, replica, and route planning. Cellerator owns the operation DAG and kernel schedule. |
| What is the smallest biology-specific scheduling information? | Domain selections, partition maps, sparse relations, cost and reuse envelopes, source locality, current residency, worker capability, topology, and readiness. |
| Should placement influence CP-BP? | Yes through ownership constraints and weighted cut terms. CP-BP implementation remains in Cellerator. |
| Should cells, genes, modules, and sequence coordinates have independent maps? | Yes. Relations and route tables connect them. |
| Can static structure precompute communication? | It can precompute routes and destination grouping for most static biological relations. Dynamic activity determines message presence and volume. |
| Can a worker receive an execution-ready object without translation? | This should become a hard invariant for the optimized path. |
| Is CellShard fundamentally a storage library? | Today its center of gravity is storage. Its correct future is a biological placement and residency substrate whose storage formats sit beneath the runtime. |
| What would be regrettable not to redesign now? | The overloaded shard, fragmented identity model, CSPACK ownership boundary, storage-shaped runtime APIs, and lack of independent domain partition maps. |

## Final judgment

If CellShard were designed today for Cellerator-scale work, the current architecture would not be selected unchanged. The archive and generated-pack foundation would be selected. The opaque payload direction would be accelerated. The overloaded shard abstraction, compatibility numerical layer, public format proliferation, and byte-shaped placement model would not be selected.

The right redesign is bounded. It does not require discarding `.csh5`, pack publication, checksums, lazy views, or sound staging code. It requires changing the nouns that connect them. Once biological domains, execution projections, storage extents, transport batches, residency leases, and work items are separate, existing transport libraries can do their jobs and CellShard can finally do the part only it can do: turn biological organization into less movement, better placement, and execution-ready data at the worker that needs it.

# Detailed code-audit appendix

# CellShard architecture review

The code-level review agent did not produce a standalone narrative report; the repository evidence inventory follows.

# CellShard source-evidence inventory

Repository clone was not available.

# Machine-generated source-evidence appendix

# CellShard source-evidence inventory

Repository clone was not available.
