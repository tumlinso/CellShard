# CS-FOUND Transition Map

This document is the bounded repository-archeology output for
`CS-FOUND-01`. It records the source state that CS-FOUND must preserve or
adapt; it is not a second task ledger and it does not claim that later
CS-FOUND contracts are implemented. Transactional task state remains in
todo-orchestrator.

## Baseline identity and workspace state

- Repository: CellShard only.
- Inspected source commit: `bbe5917bab88ffca2d16bc86166a8de368a4abbe`.
- Initial branch: `main`.
- Initial CellShard worktree state: clean; no staged, unstaged, or untracked
  files.
- Implementation branch: `cs-found-bootstrap`.
- Isolated implementation worktree: `/home/tumlinson/CellShard-CS-FOUND`.
- Registered CellShard worktrees at bootstrap: the original CellStack
  submodule checkout on `main` and the isolated CS-FOUND worktree on
  `cs-found-bootstrap`.
- No reset, clean, stash, overwrite, or absorption of unrelated work was
  performed.

The todo-orchestrator root-discovery behavior initially found CellStack's
pre-existing parent project when invoked from the nested checkout. No
CellShard plan was applied to that project. All CellShard migration, planning,
claims, gates, and evidence were subsequently isolated in the external
CellShard worktree and CellShard project identity.

## Todo-orchestrator reconciliation

Before CS-FOUND, CellShard had no local v2 transactional project. Its legacy
Markdown state contained one workstream,
`cellshard-cpp-access-adapter-refactor`, and that workstream was closed. The
legacy state was dry-run inspected and migrated without warnings. The closed
workstream remains closed and is retained as historical evidence; it was not
reopened, renamed, or duplicated.

The CellShard v2 project identity is
`a52537a5-20db-4aeb-a126-dd0128c71fda`. The migrated access-adapter record and
the CS-FOUND plan coexist in one SQLite authority. Markdown files are generated
projections only.

## Verified current reconstruction

### CPEXEC01 identity and ownership

`execution_payload_identity` is a legacy, native in-memory identity containing:

- `dataset_identity`;
- `dataset_generation_ref`, with `canonical_generation`,
  `execution_plan_generation`, `pack_generation`, and `service_epoch`;
- partition identity and global row interval/count metadata;
- feature count and feature-axis fingerprint;
- identity version, payload kind, payload schema version, row-domain tag, and
  payload identity.

`valid_execution_payload_identity` checks the legacy required fields and range
relationships. `execution_payload_identity_matches` performs exact identity
comparison. This model deliberately remains legacy: it conflates immutable
artifact facts, caller execution-plan facts, and a runtime service epoch.

The CPEXEC01 disk entry is a native `execution_payload_disk_header` containing
a native `execution_payload_identity`, payload length, a 64-bit FNV-1a checksum,
schema/endian markers, and reserved fields. Its native representation and
padding are platform/toolchain behavior, not a portable byte-layout promise.
CS-FOUND-02 protects the supported implementation without inventing a false
platform-independent golden representation.

### CSPACK01 layout and publication

The current top-level CSPACK file layout is:

```text
[8-byte CSPACK01 magic]
[native uint64 shard id]
[native uint64 partition count]
[native uint64 partition offsets]
[partition entry bytes]
```

`store_execution_cspack` validates identities, rejects duplicate partition
identities, writes a sibling `path.tmp`, flushes and `fsync`s the file, closes
it, and renames it into place. Failure cleanup removes the temporary file. It
does not currently `fsync` the containing directory. The legacy writer must
keep writing CPEXEC01; CPEXEC02 publication is separate work in CS-FOUND-08 and
CS-FOUND-09.

`load_execution_cspack_partition` checks CSPACK magic, shard/count expectations,
offset-table bounds and monotonicity, the CPEXEC01 header fields, exact expected
identity, payload bounds, and the FNV-1a checksum. It performs one host payload
allocation and one file read into that allocation.

### Current generation semantics

- `canonical_generation` identifies a canonical dataset generation in the
  compatibility model. It can map to a future `archive_generation_id` only
  through explicit caller policy.
- `execution_plan_generation` is a legacy plan-generation counter; it cannot
  substitute for explicit structure, order, geometry, operation, encoding, or
  producer identities.
- `pack_generation` remains legacy generated-pack metadata.
- `service_epoch` is runtime freshness/coordination metadata. It must not enter
  immutable image identity, content digest, CPEXEC02, or a snapshot.
- `dataset_runtime_generation` is the runtime pairing of canonical generation
  and service epoch and remains compatibility machinery.

No new identity may be silently manufactured by hashing these numeric values
together.

### Current `sharded<MatrixT>` semantics

`sharded<MatrixT>` is a row-centric, matrix-centric stitched metadata view. Its
`partition_offsets` split the matrix row domain into legacy contiguous
partitions. Its `shard_offsets` group one or more adjacent partitions into
coarser scheduling/transport units. `set_equal_shards` and
`set_shards_by_nnz` change those legacy groupings.

The hierarchy does not represent independent biological domains, multiple
partition maps per domain, explicit per-domain orders, non-contiguous biological
ownership, immutable images, storage extents, snapshots, or residency. It is
therefore preserved as transitional compatibility machinery and must not become
the new semantic foundation. In particular, `shard_offsets` do not become
biological partitions.

### Current local multi-GPU assignment

`shard_map` records device assignment for legacy shard groups.
`assign_shards_round_robin` assigns them cyclically. `assign_shards_by_bytes`
uses per-device available-byte information from Cellerator's local
`device_fleet`/`local_context` to balance estimated capacity. This is useful
local compatibility behavior; it is neither cost-aware placement nor a
topology-aware planner. It remains outside CS-FOUND's immutable identity and
artifact model.

### Current source, host, and device paths

`execution_payload_source`, `execution_payload_host`, and
`execution_payload_device` are legacy CPEXEC01 runtime types.
`clear_execution_payload_host` releases the owned host payload and clears its
state. `upload_execution_payload_async` currently selects a device, calls
`cudaMalloc`, enqueues one `cudaMemcpyAsync`, and does not synchronize. The
current function does not restore the previous current CUDA device; that is
characterized legacy behavior, not a CS-FOUND-02 fix. Device allocator
ownership and safe device restoration belong to CS-FOUND-11.

Current `.csh5` access adapters are complete and exercised by the closed
access-adapter workstream. They preserve format ownership boundaries and remain
a source of evidence. They are not reopened by CS-FOUND.

## Subsystem transition table

| Current symbol or format | Current owner | Current use | Target replacement or adapter | Migration task | Deletion task or later program | Compatibility requirement |
|---|---|---|---|---|---|---|
| `dataset_generation_ref` | CellShard I/O common | Four legacy generation/epoch numbers | Explicit new IDs plus caller-supplied legacy adapter context | CS-FOUND-03, CS-FOUND-12 | Later compatibility deletion program | Preserve type and semantics; exclude `service_epoch` from immutable images |
| `dataset_runtime_generation` | CellShard runtime/common | Runtime canonical generation plus service epoch | Runtime-only compatibility adapter | CS-FOUND-12 | Later runtime migration | Do not promote into snapshot or image identity |
| `execution_payload_identity` | CellShard pack compatibility | CPEXEC01 identity matching | `image_descriptor` plus explicit legacy context | CS-FOUND-03, 05, 12 | Later CPEXEC01 retirement | Preserve public API and exact matching behavior |
| `CPEXEC01` native entry | CellShard pack | Opaque Cellerator-owned payload envelope | Separate explicit little-endian `CPEXEC02` | CS-FOUND-08, 09 | Later legacy-format deletion decision | Remain readable and writable; no portable-padding claim |
| `CSPACK01` top-level table | CellShard pack | Generated multi-entry container | Unchanged outer container holding CPEXEC01 or CPEXEC02 entries | CS-FOUND-02, 09 | None in CS-FOUND | Preserve magic and top-level layout |
| `store_execution_cspack` | CellShard pack | Atomic-ish CPEXEC01 writer | Preserve; add separate `store_image_cspack` | CS-FOUND-02, 09 | Later adapter retirement | Must not silently write CPEXEC02 |
| `load_execution_cspack_partition` | CellShard pack | CPEXEC01 load/validate | Preserve; add source-based image inspection/loading | CS-FOUND-02, 09, 10 | Later adapter retirement | Exact mismatch and corruption rejection remain |
| `execution_payload_source` | CellShard pack runtime | Path and entry metadata for legacy payload | `extent_descriptor` plus `payload_source_ref` | CS-FOUND-06, 10, 12 | Later adapter retirement | Paths stay out of immutable image descriptors |
| `execution_payload_host` | CellShard pack runtime | Owned legacy host bytes | Move-only `host_residency_lease` plus view | CS-FOUND-10, 12 | Later adapter retirement | Legacy clear API remains |
| `execution_payload_device` | CellShard pack runtime | CellShard-owned `cudaMalloc` allocation | Caller allocator plus `device_residency_lease` | CS-FOUND-11, 12 | Later adapter retirement | Legacy upload/clear APIs remain |
| `upload_execution_payload_async` | CellShard CUDA pack path | One asynchronous H2D copy using `cudaMalloc` | New caller-allocated staging path | CS-FOUND-11 | Later compatibility deletion | No change under CS-FOUND-02; no synchronization added |
| `sharded<MatrixT>` | CellShard runtime layout | Row-partitioned stitched matrix metadata | Explicit adapter to domain/partition descriptors | CS-FOUND-04, 12 | Later shard-model migration | Preserve public headers and behavior |
| `partition_offsets` | CellShard runtime layout | Legacy contiguous row partitions | Contiguous `partition_selection` only with explicit domain context | CS-FOUND-04, 12 | Later compatibility cleanup | May map rows only when caller supplies identities and orders |
| `shard_offsets` | CellShard runtime layout | Coarse legacy shard grouping | Remains physical/transport compatibility metadata | CS-FOUND-12 | Later placement/transport program | Never map automatically to biological ownership |
| `set_equal_shards`, `set_shards_by_nnz` | CellShard runtime layout | Legacy shard regrouping | Compatibility-only operations | CS-FOUND-12 | Later placement program | Do not describe as biological partitioning |
| `shard_map` | CellShard distributed runtime | Legacy shard-to-device assignment | Placement remains absent from CS-FOUND | CS-FOUND-12 documentation | Expected CS-PLACE | Do not place it in immutable descriptors |
| `assign_shards_round_robin` | CellShard distributed runtime | Cyclic local device assignment | Legacy adapter only | CS-FOUND-12 | Expected CS-PLACE | Preserve; do not call topology-aware |
| `assign_shards_by_bytes` | CellShard distributed runtime | Available-byte capacity balancing | Legacy adapter only | CS-FOUND-12 | Expected CS-PLACE | Preserve; do not call cost-aware compute balancing |
| `.csh5` | CellShard archive I/O | Canonical archive family | Remains canonical archive | CS-FOUND-14 documentation | None in CS-FOUND | CS-FOUND creates no replacement archive |
| `.cspack` | CellShard pack I/O | Disposable generated artifact | CSPACK01 with legacy and new image envelopes | CS-FOUND-08, 09, 14 | None in CS-FOUND | No new extension or third public lifetime |
| `.cspool` | CellShard ingest | Internal staged ingest machinery | No CS-FOUND runtime promotion | Documentation only | Later ingest work if any | Remains internal, not public runtime API |
| `.cshard` / CSHARD format | CellShard experimental I/O | Experimental shard-oriented format | No role in new foundation | Documentation only | Later explicit format decision | Must not be presented as CS-FOUND runtime substrate |
| Raw `packfile` layouts | CellShard pack | Existing raw-format packing | Preserve alongside new image envelope | CS-FOUND-09 compatibility review | Later format-specific work | No gratuitous rewrite |
| `.csh5` access-adapter layer | CellShard access/I/O | Opaque access boundary across current formats | Preserve completed prerequisite | Reconciled in CS-FOUND-01 | None in CS-FOUND | Closed workstream history remains closed |
| CSH5 source/cache/service helpers | CellShard I/O/runtime | Archive-specific access and runtime generation checks | Remain archive-specific; new source ref is storage-object based | CS-FOUND-06, 10 | Later source integration where explicit | Do not put HDF5 into new source/residency headers |
| Mask/reduction compatibility paths | CellShard runtime | Current row-mask and local execution helpers | No foundational replacement | Baseline only | Later owning subsystem work | Known mask-group failure remains separately classified |
| Domain identities and partition maps | Not present as distinct contracts | Currently inferred through rows/generations | New strong IDs and descriptors | CS-FOUND-03, 04 | Foundation retained | No row-centric inference |
| Projection/image descriptors | Not present as distinct contracts | Identity is bound to CPEXEC01 legacy fields | Opaque producer ABI, projection key, and image descriptor | CS-FOUND-05 | Foundation retained | No Cellerator/Baseplane internals |
| Storage object and extent identities | Not present as distinct contracts | File offsets and paths are coupled to loads | Immutable object/extent descriptors | CS-FOUND-06 | Foundation retained | Extent is not partition, work, or residency |
| Artifact/source catalogs and snapshots | Not present | Ad hoc caller/file knowledge | In-memory catalog, separate source catalog, snapshot manifest | CS-FOUND-07 | Foundation retained | No database or public manifest format |
| Local-file `payload_source_ref` provider | Not present | Path-specific stdio reads | Non-owning ops table plus local exact-range provider | CS-FOUND-06, 10 | Foundation retained | No async framework, HDF5, mmap default, or network |
| Host/device residency leases | Not present | Legacy payload structs own buffers | Move-only leases plus POD-like views | CS-FOUND-10, 11 | Foundation retained | One host payload allocation and one caller-allocated H2D copy |

## Fixed CS-FOUND comparison

The current implementation is preserved where sound, but it does not yet
separate biological domain, partition map, biological partition, execution
projection, immutable image, storage object, extent, source location, snapshot,
and runtime residency. CS-FOUND adds those concepts beside the compatibility
model. It does not reinterpret the existing shard hierarchy as the new model.

The fixed vertical direction remains:

```text
biological identities and partitions
  -> opaque external execution projection
  -> immutable image descriptor
  -> checksummed storage extent
  -> storage-independent source
  -> owned host residency
  -> caller-allocated device residency
  -> external consumer bind
```

Networking, distributed collectives, topology discovery, placement, scheduling,
routing, cache policy, biological route compilation, and new durable archive
formats remain absent.

## Baseline environment and validation

- Host compiler and exact CMake detection: recorded by the CS-FOUND-01
  configure gate.
- CUDA compiler: NVIDIA HPC SDK CUDA 12.9 `nvcc`.
- Driver: 580.173.02.
- Available validation devices: four Tesla V100-SXM2-16GB GPUs, compute
  capability 7.0.
- External header dependency: a read-only `git archive` export of committed
  Cellerator commit `72f3012adf63bbd8308fea620c9d8d8841136b51` at
  `/tmp/cs-found-cellerator-72f3012adf63bbd8308fea620c9d8d8841136b51`.
  The dirty Cellerator working tree is not used by this build.
- Benchmark posture: no benchmark or profiler run.
- `compute-sanitizer`: the default installation lacks a usable payload; it is
  optional and not required for this baseline.

Exact gate commands and their exit status are authoritative ledger evidence.
The baseline configuration is:

```bash
cmake -S . -B build-cs-found-baseline \
  -DCELLSHARD_ENABLE_CUDA=ON \
  -DCELLSHARD_BUILD_TESTS=ON \
  -DCELLSHARD_BUILD_EXPORT=ON \
  -DCELLSHARD_CELLERATOR_SOURCE_DIR=/tmp/cs-found-cellerator-72f3012adf63bbd8308fea620c9d8d8841136b51 \
  -DCELLERATOR_DIST_NCCL_LIBRARY=FALSE
cmake --build build-cs-found-baseline -j 4
```

The first isolated configure omitted the external source directory and failed
before generation because `Cellerator::dist` could not be resolved. The next
build auto-discovered NCCL and reproduced the committed
`nccl_communicator.cuh` incomplete-`void`/undefined-`local_context` compile
failure. The final baseline explicitly disables the optional NCCL library
because CS-FOUND performs no collective work. Both earlier attempts are
classified setup/dependency failures, not CellShard source regressions.

Focused baseline executables cover the access adapter, CPEXEC01 payload,
export, CSHARD compatibility, installed package, dense runtime, and optimized
sliced runtime. `cellShardMaskGroupsRuntimeTest` is expected to exit 14 at the
historical row-keep assertion; an unchanged exit 14 is recorded as a
pre-existing classified failure, never as a passing test.

Final G0 results on the configuration above:

- configure and full build: exit 0;
- `cellShardAccessAdapterTest`: exit 0;
- `cellShardExecutionPayloadTest`: exit 0 on a ledger-locked V100;
- `cellShardExportRuntimeTest`: exit 0;
- `cellShardCshardTest`: exit 0;
- `cellShardInspectPackageTest`: built and its installed consumers compiled,
  exit 0;
- `cellShardDenseRuntimeTest`: exit 0 on a ledger-locked V100;
- `cellShardOptimizedSlicedExecutionTest`: exit 0 on a ledger-locked V100;
- `cellShardMaskGroupsRuntimeTest`: exit 14, matching the explicit historical
  expectation and classified as pre-existing.

No benchmark executable was run. The default full build compiled the existing
benchmark target because it is registered in the current build graph; that
compilation is not a benchmark result.

## Containment and rollback

CS-FOUND-01 changes documentation and transactional coordination state only. It
deletes no source, changes no public API, and changes no byte format. If this
map is found to disagree with source, the task must be reopened and corrected
before a later interface checkpoint is frozen. The current source and test
evidence remain authoritative over this document.
