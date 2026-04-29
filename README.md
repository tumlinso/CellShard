# CellShard

CellShard is a low-level, header-first library for large sharded sparse omics matrices.

Its scope is narrow:

- represent sparse matrices as partitions and shards
- persist them in CellShard container formats
- load metadata without materializing full payloads
- fetch and drop host-side partitions
- stage partitions and shards to GPU
- build and serve CSPACK generations from the canonical container
- provide the sparse conversion and layout helpers needed for that runtime
- optionally stream source matrices and metadata into canonical CellShard containers

CellShard is the storage, pack-delivery, and distributed execution base layer. It is not the analysis toolkit, model layer, or Torch integration layer. Those live on the Cellerator side.

## What CellShard Owns

- sparse matrix formats such as `blocked_ell`, `compressed`, `dense`, and `triplet`
- sharded matrix metadata and layout helpers
- on-disk matrix and dataset container persistence
- CSPACK generation and delivery from canonical dataset containers
- host fetch and drop operations
- device upload and staging helpers
- sparse row/feature masking and grouped row reductions for runtime-ready
  Blocked-ELL, Sliced-ELL, and compressed fallback views
- owner-hosted runtime metadata and distributed shard/CSPACK delivery
- optional bounded source ingest when configured with `CELLSHARD_BUILD_INGEST=ON`
- optional export helpers and optional Python bindings

CellShard does not own:

- biological preprocessing, normalization, and analytical row/column selection workflows
- clustering, embedding, or neighbor analysis APIs
- model training or inference
- Torch or libtorch integration

## Source Layout

CellShard is mainly a header-first C++/CUDA library. Public reusable surfaces live under `include/CellShard/`, while implementation files live under `src/`, `export/`, and `python/`.

Common file roles:

- `.cuh`: CUDA-aware headers for reusable library surfaces
- `.cu`: CUDA translation units
- `.cc`: C++ translation units
- `.hh`: used rarely here compared with `.cuh`; most public interfaces are CUDA-aware

Naming is mostly `snake_case` for files, functions, variables, and structs.

Main source areas:

- `include/CellShard/CellShard.hh`: thin umbrella include for the main public surface
- `include/CellShard/core/`: core types, spans, compatibility, and scalar helpers
- `include/CellShard/formats/`: concrete matrix layouts such as `compressed`, `blocked_ell`, `dense`, `triplet`, and `diagonal`
- `include/CellShard/runtime/`: public sharded layout, host fetch/drop, device staging, storage dispatch, and local multi-GPU headers
- `include/CellShard/io/`: public `.cspack`, `.csh5`, and standby `.cshard` entry headers
- `include/CellShard/ingest/`: optional source ingest headers, enabled with `CELLSHARD_BUILD_INGEST=ON`
- `include/CellShard/export/`: public dataset export and metadata snapshot APIs
- `src/io/pack/`: the per-partition packed payload codec used inside shard `.cspack` cache files
- `src/io/csh5/`: the `.csh5` container backend plus runtime cache materialization and fetch
- `src/io/cshard/`: the experimental HDF5-free `.cshard` v1 reader, validator, row reader, and converter
- `src/runtime/layout/`, `src/runtime/host/`, `src/runtime/device/`, `src/runtime/distributed/`, and `src/runtime/storage/`: internal runtime layout and staging implementation surfaces
- `src/convert/`: sparse conversion helpers and raw conversion kernels
- `src/bucket/`: row/major-axis nnz bucketing helpers
- `export/summary/`, `export/materialize/`, and `export/snapshot/`: non-Torch export implementation split by responsibility
- `python/`: optional pybind module and Python package wrapper
- `tests/`: focused runtime and package-consumer checks

Useful public/runtime waypoints:

- `include/CellShard/runtime/layout/sharded.cuh`: the metadata-only `sharded<T>` view and partition/shard boundary helpers
- `include/CellShard/runtime/storage/shard_storage.cuh`: shard-storage backend, role, and capability definitions for the active `.csh5` runtime
- `include/CellShard/runtime/host/sharded_host.cuh`: host-side fetch/drop and shard regrouping
- `include/CellShard/runtime/device/sharded_device.cuh`: single-GPU upload and staging
- `include/CellShard/runtime/distributed/distributed.cuh`: local multi-GPU shard placement and owner staging
- `include/CellShard/runtime/mask_groups.cuh`: generic sparse row/feature
  masks, grouped row reductions, fleet dispatch, and the explicit masked-layout
  reoptimization hook
- `include/CellShard/runtime/storage/disk.cuh`: `load_header()` dispatch for CellShard dataset files
- `include/CellShard/io/csh5/api.cuh`, `src/io/csh5/create.cc`, `src/io/csh5/metadata.cc`, `src/io/csh5/finalize_preprocess.cc`, `src/io/csh5/write.cc`, and `src/io/csh5/runtime/`: the `.csh5` container backend plus shard `.cspack` runtime caches
- `include/CellShard/io/cshard.hh` and `include/CellShard/io/cshard/spec.hh`: the standby `.cshard` archive API and fixed v1 POD records
- `include/CellShard/io/pack/packfile.cuh` and `src/io/pack/packfile.cu`: the per-partition packed payload codec used inside shard `.cspack` cache files

## Core Concepts

- `partition`: one stored matrix chunk with explicit row bounds
- `shard`: a group of partitions used as the higher-level fetch/staging unit
- `blocked_ell`: the native sparse execution and persistence layout for new `.csh5` output
- `compressed`: an explicit row-compressed interop path, not a supported `.csh5` file format
- `sharded<T>`: metadata plus optional loaded payload pointers for a partitioned matrix collection
- `shard_storage`: the bound storage backend used for lazy fetch/materialization
- `.csh5`: the canonical CellShard dataset container and archive format
- `.cshard`: the experimental standby HDF5-free native archive format
- `.cspool`: the local ingest-spool part format used before final `.csh5` assembly
- `.cspack`: the generated execution artifact used for fast multithreaded fetch and delivery

CellShard runtime masking is a generic sparse compute primitive, not a
biology-specific QC policy. Biological group definitions such as mitochondrial,
ribosomal, or hemoglobin feature rules are compiled by CellShardPreprocess and
passed in as ordinary `uint32_t` feature-group masks. Runtime row masks and
feature masks are independent: a feature can belong to one or more groups and
still be excluded from a particular masked reduction or explicit rebuild.

The explicit `manual_reoptimize_masked_sparse()` hook rebuilds a masked sparse
view only when the caller asks for it and reports compact kept-row,
kept-feature, live-nnz, layout, and byte metadata. Automatic reoptimization is
intentionally not enabled yet; the public prediction placeholder records the
future inputs: density change, row/feature drop ratios, layout padding change,
estimated rebuild cost, and expected repeated-use count.

## Python Binding Posture

The Python package is native-first for `.csh5` datasets. `cellshard.open(path).matrix()`,
`matrix(format="native")`, `rows(...)`, slicing, and `head(...)` return lazy
native view objects by default. Those objects expose metadata and only fetch
payloads when a partition, shard, or explicit conversion method is requested.

Expensive interop is opt-in:

- `format="csr"` returns `CsrMatrixExport`
- `format="scipy"` builds a SciPy CSR matrix
- `format="torch"` imports Torch and builds a `torch.sparse_csr_tensor`
- `NativeMatrixView.to_csr()`, `.to_scipy_csr()`, and `.to_torch_sparse_csr()`
  make full-matrix conversion explicit
- `NativeRowSelection.to_csr()`, `.to_scipy_csr()`, and
  `.to_torch_sparse_csr()` make selected-row conversion explicit

Native partition fetch returns typed objects such as `BlockedEllPartition` and
`SlicedEllPartition`. Stored values are exposed as raw 16-bit storage
(`values_storage`) by default; call `values_float32()` when inspection really
needs expanded floats.

## Ownership Boundary

- CellShard owns optional bounded source ingest, metadata capture, local `.cspool` staging, and one-pass emission of an immutable canonical sparse matrix when `CELLSHARD_BUILD_INGEST=ON`.
- Cellerator owns ML compute, model workflows, Torch interop, trajectory logic, and reusable analysis kernels above CellShard storage/runtime surfaces.
- CellShardPreprocess owns accelerated standard-biology preprocessing and requires CellShard ingest when installed through CellShard.
- CellShard owns partitioning, sharding, blocking, bucketing, rebucketing, CSPACK generation, CSPACK delivery, and append-only canonical/runtime generation management.
- Once ingest emits a canonical matrix to CellShard storage, row and column membership is immutable for that canonical generation.

## Storage Model

CellShard has two different storage roles, and the distinction matters:

- `.csh5` is the canonical container. It stores the durable dataset header, partition and shard layout tables, dataset and provenance metadata, optional browse and observation metadata, authoritative payload arrays, and runtime-service metadata such as generation identity and execution ownership hints.
- `.cspack` is the execution-facing runtime format. CellShard builds versioned CSPACK generations from `.csh5` and serves those CSPACK artifacts to executor clients for low-latency multithreaded access.
- `.cshard` is an experimental standby archive. It can be inspected,
  validated, converted from `.csh5`, and read directly by row range without
  HDF5, but it does not replace `.csh5` in production paths yet. Its
  multi-assay path writes one global observation table, per-assay feature
  tables, exact or partial observation-level row maps, and either CSR fallback
  payloads or already optimized bucketed Blocked-ELL/Sliced-ELL assay-shard
  blobs. CSPACK remains a coordinated single-assay execution artifact.

When optional CellShard ingest is enabled, there is also a bounded local ingest spool:

- ingest can spill intermediate `.cspool` partition artifacts to a machine-local SSD spool before the final `.csh5` is assembled
- that spool is not an archive format and is not part of the steady-state runtime contract
- its purpose is to avoid rereading an expensive source MTX while preserving `.csh5` as the durable source of truth
- row-aligned parts and shard-aligned fetch units are still the persistent contract; ingest does not split one cell across parts or shards

The intended posture is:

1. keep `.csh5` as the source of truth and archive
2. bind the `.csh5` container and inspect its metadata cheaply
3. materialize or refresh the active CSPACK generation from `.csh5`
4. serve CSPACK data from the `.csh5` owner host to local or remote executors
5. fetch host partitions from the active CSPACK generation
6. stage those fetched partitions or shards to GPU

The cache and delivery directories should stay legible on disk. The current
intended layout is:

```text
<cache_root>/
  instances/
    <fingerprint>/
      metadata/
        manifest.txt
      packs/
        plan.<execution_plan_generation>-pack.<pack_generation>-epoch.<service_epoch>/
          shard.<id>.cspack
```

The fingerprinted instance keeps source identity stable, while the directory
names around it make it obvious where manifests and the currently published
CSPACK generation live.

CSPACK carries the runtime-ready shard payload. For sliced datasets that is the
native bucketed sliced partition payload. For Blocked-ELL it is the bucketed
Blocked-ELL partition representation used by hot reads, with canonical order
reconstructed only for compatibility callers.

Do not think of `.csh5` as the final hot execution substrate. `.csh5` is the durable canonical source and append target, while `.cspack` is the runtime format used for high-throughput repeated access and delivery.

The standby `.cshard` command surface is:

```text
cellshard cshard inspect dataset.cshard
cellshard cshard validate dataset.cshard
cellshard cshard read-rows dataset.cshard --start 0 --count 16
cellshard cshard convert dataset.csh5 --out dataset.cshard
```

The intended workflow is:

1. bind a `.csh5` storage backend and load metadata
2. inspect partitions and shard boundaries
3. ensure the needed CSPACK generation exists for the active execution epoch
4. serve or fetch host partitions or shards from the active CSPACK generation
5. optionally upload or stage them to GPU, and keep repeated hot readers on device-resident execution partitions when the generation is unchanged
6. run higher-level compute outside CellShard

Note on naming:

- `runtime/storage/shard_storage.cuh` is the public storage-role surface
- the older `runtime/layout/shard_paths.cuh` name is retained only as a compatibility shim
- actual cache-pack path builders live in the `.csh5` runtime helpers, especially `src/io/csh5/preprocess_helpers_part.hh`

For large or remote MTX ingest, the practical write-side workflow is:

1. stream the source matrix once through bounded conversion windows
2. finish all filtering and shape-changing decisions before canonical CellShard emission
3. spill bounded `.cspool` part artifacts to machine-local spool storage
4. assemble or append the final `.csh5` from that local spool
5. build or refresh the active CSPACK generation on the machine that owns the `.csh5` source

## Operating Model

CellShard is intended to run as an owner-hosted runtime service, even when the
entire workflow stays on one machine.

The important rule is:

- `.csh5` stays on the owner host as the durable canonical source
- `.cspack` is the execution artifact that readers and executors consume
- optional CellShard ingest finishes filtering and immutable canonical emission
  before the runtime side takes over
- CellShard owns pack preparation, delivery, append staging, and generation
  cutover

### Single-Machine Operation

On one machine, CellShard should still behave like a service with clear roles:

- one owner-side coordinator controls the active generation and decides whether
  reads come from the current published generation or a staged append
- one master `.csh5` reader owns canonical file access during hot operation so
  HDF5/file-lock behavior stays predictable
- pack-preparation workers build or refresh CSPACK generations from `.csh5`
- local executor threads read the prepared CSPACK generation for fast
  multithreaded random access
- spool-write workers may receive new canonical or derived data and stage it
  without mutating the active generation in place

Executor-facing runtime APIs should therefore fail when the required published
pack is absent instead of reopening `.csh5` and rebuilding execution state on
the fly. Owner-side runtime code may still perform that recovery when it is
explicitly acting as the canonical reader and pack-preparation authority.

The network layer is absent in this mode, but the synchronization model should
stay the same as distributed mode:

- readers consume published CSPACK generations
- writers stage append-only updates separately
- the coordinator publishes a new generation only after pack rebuild is ready
- active canonical payloads are not moved or overwritten during hot reads

The intended single-machine flow is:

1. Cellerator finishes filtering and emits an immutable canonical matrix
2. CellShard assembles or appends `.csh5`
3. CellShard builds the active CSPACK generation locally
4. executor threads read from pack, not directly from `.csh5`
5. staged writes remain separate until a publish/cutover event

### Distributed Operation

In distributed mode, the same owner-service contract remains in force:

- one owner node holds the canonical `.csh5`
- executor nodes do not copy `.csh5` as their normal operating model
- the owner node prepares and delivers CSPACK data or CSPACK generations to remote
  executors
- remote nodes may keep local pack caches, but those are runtime artifacts and
  not sources of truth

The owner node should provide these responsibilities:

- canonical `.csh5` access and metadata inspection
- CSPACK generation and rebuild
- shard ownership and generation routing
- append staging and publish/cutover
- CSPACK delivery to remote executors

Executor nodes should provide these responsibilities:

- request shard or CSPACK data for the active generation
- keep node-local delivered packs only as execution caches
- execute against CSPACK artifacts without assuming ownership of `.csh5`
- accept cutover to newer generations when the owner publishes them

The intended distributed flow is:

1. Cellerator emits an immutable canonical matrix to the owner-side CellShard builder
2. the owner node appends or assembles `.csh5`
3. the owner node builds the active CSPACK generation
4. executor nodes fetch or receive CSPACK data for their assigned shards
5. reads continue on the published generation while append staging happens separately
6. the owner publishes a new generation only after the replacement pack set is ready

### Runtime Threads And Roles

The exact thread counts can change, but the runtime should scale around these
roles:

- owner-side coordinator for generation state and routing
- owner-side master `.csh5` reader
- owner-side pack preparation and delivery workers
- owner-side spool receive/write staging workers
- optional network delivery workers in distributed mode
- executor-side pack readers / execution workers

This is intentionally not a model where every worker opens and mutates `.csh5`
freely. The point of the service model is to keep canonical file access narrow
and make CSPACK delivery the fast path.

## Live Operation

- Hot reads use versioned CSPACK generations, not in-place mutation of active payloads.
- While reads are active, canonical matrix payloads are append-only: new generations may be appended, but active payloads may not be moved or overwritten.
- Writers may stage new spool content concurrently, but publish/cutover happens only when the runtime switches to a new generation.
- Maintenance mode is the explicit state where overwrite or relocation of canonical payloads is permitted.

Generation handling should follow these rules:

- `canonical_generation` identifies the immutable canonical matrix generation
- `execution_plan_generation` identifies the execution ownership/layout plan
- `pack_generation` identifies the currently prepared runtime pack set
- `service_epoch` identifies the currently published read epoch
- the active CSPACK path should be generation-qualified so read-only
  pack reuse is tied to the published runtime generation, not only to source
  file timestamps
- staged appends may create newer canonical or execution generations before
  they become the published read generation

## Build And Package Surface

CellShard can be built as a standalone project and exports component-scoped CMake targets:

- `headers`
- `inspect`
- `runtime`
- `export`
- `h5ad_python`

Typical package use:

```cmake
find_package(CellShard REQUIRED COMPONENTS inspect)
target_link_libraries(my_tool PRIVATE CellShard::inspect)
```

Inspect-only include:

```cpp
#include <CellShard/core/cuda_compat.cuh>
#include <CellShard/runtime/layout/sharded.cuh>
```

CUDA/runtime umbrella include:

```cpp
#include <CellShard/CellShard.hh>
```

If CellShard is configured with `CELLSHARD_ENABLE_CUDA=OFF`, the install does not export `CellShard::runtime`.

Python package basics:

```bash
python -m pip install .
python -m pip wheel . --no-deps
```

The Python package currently targets Linux `x86_64`, Python `3.10`-`3.13`, and the same HDF5/CUDA posture as the standalone CMake build. Wheel builds intentionally skip the standalone `cellshardH5adExport` app and package only the Python extension plus Python-facing helpers. The Python surface now exposes both low-level file helpers and an owner/client bootstrap model built around serialized metadata snapshots; the client-facing retrieval path is still layered on the canonical container plus pack runtime model.

Hosted CI and portability notes live in `SUPPORT.md`.

## Python Surface

The Python package exposes an easy high-level facade plus lower-level owner/client helpers:

- `open(...)` / `Dataset`
- `open_dataset` / `DatasetFile`
- `open_dataset_owner` / `DatasetOwner`
- `bootstrap_dataset_client` / `DatasetClient`
- `load_dataset_summary`
- `load_dataset_as_csr`
- `load_dataset_rows_as_csr`
- `materialize_derived_dataset`
- `load_dataset_global_metadata_snapshot`
- `serialize_global_metadata_snapshot`
- `deserialize_global_metadata_snapshot`
- `make_client_snapshot_ref`
- `validate_client_snapshot_ref`
- `stage_append_only_runtime_service`
- `publish_runtime_service_cutover`
- `describe_pack_delivery`
- `DatasetFile.materialize_partition`
- `DatasetClient.materialize_torch_sparse_csr`
- `DatasetClient.materialize_rows_torch_sparse_csr`
- `DatasetClient.materialize_scipy_csr`
- `DatasetClient.materialize_rows_scipy_csr`
- `write_h5ad`

For ordinary Python use, prefer the high-level facade:

```python
import cellshard

ds = cellshard.open("/path/to/dataset.csh5")
batch = ds[:256]                  # torch sparse CSR by default
preview = ds.head(32, format="scipy")
info = ds.describe()
```

For a quick smoke check after install:

```bash
python -c "import cellshard; print(cellshard.__version__)"
```

## Quick Orientation

If you are browsing the code for the first time:

- start with `include/CellShard/CellShard.hh`
- read `include/CellShard/formats/` to understand the matrix types
- read `include/CellShard/io/pack/packfile.cuh` and `src/io/pack/packfile.cu` if you care about the packed part codec used inside shard `.cspack` files
- read `include/CellShard/runtime/layout/sharded.cuh` and `include/CellShard/runtime/host/sharded_host.cuh` for the host-side model
- read `include/CellShard/runtime/device/sharded_device.cuh` for upload and staging
- read `include/CellShard/io/csh5/api.cuh`, `src/io/csh5/create.cc`, `src/io/csh5/metadata.cc`, `src/io/csh5/finalize_preprocess.cc`, `src/io/csh5/write.cc`, and `src/io/csh5/runtime/` if you care about the `.csh5` container and shard `.cspack` runtime caches
- read `include/CellShard/runtime/distributed/distributed.cuh` if you care about local multi-GPU placement

## Notes

- Blocked-ELL is the preferred sparse layout for persisted execution and staging.
- `.csh5` is the canonical container; `.cspack` generations are the hot runtime delivery format.
- CellShard stays Torch-free by design.
- Build output is ignored in the repo-level `.gitignore`.
