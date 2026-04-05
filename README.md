# CellShard

CellShard is a low-level, header-first library for very large sharded sparse omics matrices.

Its job is narrow:

- represent matrices as row-aligned parts and shards
- persist them in a native on-disk format
- fetch and drop parts efficiently on host
- stage shards to GPU
- provide the sparse conversion machinery needed to support that runtime

CellShard is not the bioinformatics toolkit. It is the storage and staging substrate that supports `Cellerator`.

Build output is ignored in the repo-level `.gitignore`.

## Scope

CellShard owns:

- per-part matrix formats
- sharded matrix metadata
- native packfile/container layout
- host fetch/drop operations
- GPU residency and staging helpers
- local multi-GPU shard partitioning and staging helpers
- sparse format conversion required by the above

CellShard does not own:

- biological file-format ecosystems beyond minimal source conversion support
- RNA preprocessing and normalization
- nearest-neighbor search, graph construction, clustering, or embedding
- state quantization logic
- model training or inference
- broader single-cell analysis workflows

Short version:

- `CellShard` stores and stages data
- `Cellerator` interprets and transforms data

## Role In The Repository

`CellShard` is the assistant library to `Cellerator`.

`Cellerator` should be able to rely on `CellShard` for:

- sharded matrix persistence
- packfile-backed loading
- explicit part and shard boundaries
- predictable host memory footprint
- direct GPU staging paths
- local multi-GPU shard distribution

That means the novelty of `CellShard` is not “another sparse matrix library”.

The novel combination is:

- sharded sparse omics matrices
- native packfile-backed storage
- part/shard-aware fetch and release
- GPU-first staging for out-of-core workloads
- direct same-machine multi-GPU shard execution support

## Local Multi-GPU Runtime

CellShard now includes a small local multi-GPU runtime layer in `src/sharded/distributed.cuh`.

This layer is intentionally narrow. It is meant to make all visible GPUs in one machine easy to use without turning CellShard into a workflow engine.

The key pieces are:

- `distributed::local_context`
- `distributed::shard_map`
- `distributed::device_fleet`

The intended flow is:

1. discover visible GPUs and create one stream per device
2. enable peer access where available
3. assign shards to devices, preferably by shard byte size
4. stage shards to their owner GPUs asynchronously
5. let `Cellerator` launch the biology-facing kernels

If NCCL is available, the same local context can also bootstrap one communicator per visible GPU. That support is optional and opportunistic; the core same-machine shard distribution path does not depend on NCCL being present.

## Copy Semantics

CellShard is explicit about ownership, but several convenience paths still do
real copies and real materialization work. Users need to treat these as data
movement operations, not cheap view construction.

The important rules are:

- `fetch_part()` opens the packfile, seeks to the stored extent, and materializes a fresh host-side matrix object for that part.
- `fetch_shard()` is currently a loop over `fetch_part()`. One shard fetch can mean many packfile seeks and many host allocations if a shard contains many parts.
- `upload_part()` and `upload_part_async()` allocate fresh device memory and perform host-to-device copies of the part payload. They also copy a small descriptor struct to device memory.
- `upload_shard()` and `upload_shard_async()` are loops over per-part uploads. They do not pack a whole shard into one bulk transfer.
- `stage_part()` combines host fetch and device upload. If the part is cold on host, it performs packfile I/O and then H2D copies.
- `stage_part_async()` is only async for the H2D upload path. If the part is absent on host, the packfile fetch still happens synchronously on the calling thread before the async copy is queued.
- `stage_shard()` and `stage_shard_async()` are loops over `stage_part*()`, so caller-visible shard shape matters a lot for launch count, allocation churn, and copy count.
- `swap_shard()` and `swap_shard_async()` stage the incoming shard before releasing the outgoing shard. Peak device residency briefly includes both shards.
- `reserve_parts()` and `reserve_shards()` reallocate and `memcpy` metadata tables on host when capacities grow.
- `bind_packfile()` copies the path string into owned storage.

What this means in practice:

- keep host parts resident if you will restage them soon
- keep device shards resident if the next kernel pass will reuse them
- choose shard boundaries to control part count, not just row count
- prefer `set_shards_by_device_bytes()` or `assign_shards_by_bytes()` when balancing work across GPUs
- avoid loops that repeatedly `fetch -> stage -> drop -> fetch -> stage -> drop` the same parts
- treat the async staging helpers as stream-ordered upload helpers, not as background disk I/O engines

If you want peak throughput, user behavior matters almost as much as the low-level code:

- stable shard reuse beats convenience restaging
- fewer larger parts usually beat many tiny parts
- device residency beats host convenience
- graph capture should happen above CellShard, after shard residency and stream layout are already stable

## Performance Incursion

The APIs below are grouped by how much work they usually incur.

`Very Low`

- boundary/metadata helpers in `src/offset_span.cuh` and `src/sharded/sharded.cuh`
- device-footprint estimators like `device_part_bytes()` and `device_shard_bytes()` in `src/sharded/sharded_device.cuh`
- raw conversion launchers in `src/convert/compressed_from_coo_raw.cuh`, `src/convert/coo_from_compressed_raw.cuh`, and `src/convert/compressed_transpose_raw.cuh`

These paths only:

- inspect metadata
- compute byte estimates
- choose launch geometry
- launch kernels over already-device-resident buffers

They do not:

- allocate host memory
- allocate device memory
- read from disk
- copy between host and device

`Low`

- `assign_shards_round_robin()` and `assign_shards_by_bytes()` in `src/sharded/distributed.cuh`
- `set_shards_to_parts()`, `set_equal_shards()`, `set_shards_by_nnz()`, and `set_shards_by_part_bytes()` in `src/sharded/sharded_host.cuh`
- `load_header()` in `src/sharded/disk.cuh` when you only need metadata

These paths mainly do:

- host metadata allocation
- host metadata copies
- cheap reductions over metadata

They do not move matrix payload.

`Moderate`

- `reserve_parts()` and `reserve_shards()` in `src/sharded/sharded_host.cuh`
- `reserve()` in `src/sharded/distributed.cuh` and `reserve()` in `src/sharded/sharded_device.cuh`
- `bind_packfile()` in `src/sharded/shard_paths.cu`

These paths do host-side allocation and copy work, but they do not necessarily
touch matrix payload and they do not move payload between host and device.

Typical costs:

- `memcpy` of metadata arrays
- path-string ownership copies
- packed-string blob growth copies
- temporary workspace growth

`High`

- `store()` / `load()` for standalone part files in `src/disk/matrix.cuh`
- `store_header()` / `load_header()` in `src/sharded/disk.cuh`
- `discover_local()` and `enable_peer_access()` in `src/sharded/distributed.cuh`

These paths do at least one of:

- synchronous file I/O
- host allocation proportional to metadata or one part
- file scans over large text sources
- CUDA runtime setup work per device

They are expensive, but they still usually avoid combining disk I/O and
host-device copies in the same operation.

`Very High`

- `fetch_part()` and `fetch_shard()` in `src/sharded/sharded_host.cuh`
- `upload_part()` / `upload_part_async()` and `upload_shard()` / `upload_shard_async()` in `src/sharded/sharded_device.cuh`
- `stage_part()` / `stage_part_async()` and `stage_shard()` / `stage_shard_async()` in `src/sharded/sharded_device.cuh`
- `swap_shard()` / `swap_shard_async()` in `src/sharded/sharded_device.cuh`
- full packfile `store()` in `src/sharded/disk.cuh`

These are the expensive paths because they combine multiple cost domains:

- file I/O
- host allocation
- host copies
- device allocation
- H2D copies
- sometimes D2H copies
- sometimes temporary double residency

The heaviest families are:

- `fetch -> upload -> drop`
- `stage_*`
- MTX text ingest
- pinned-triplet -> compressed conversion
- any loop that repeatedly rebuilds device buffers for the same shard

## Operation Notes

`fetch_part()`

- opens the packfile
- seeks to one part extent
- reads one packed matrix payload
- allocates fresh host arrays

`fetch_shard()`

- loops over `fetch_part()`
- one logical shard can mean many seeks and many allocations

`upload_part()`

- assumes the host part is already materialized
- allocates fresh device buffers
- performs one or more H2D copies depending on format
- allocates and copies a device-side descriptor

`upload_shard()`

- loops over `upload_part()`
- copy count scales with parts per shard

`stage_part()`

- may call `fetch_part()` first
- then calls `upload_part()`
- may immediately free the host part

`stage_part_async()`

- only the upload half is stream-ordered
- if the part is cold, the packfile fetch still happens synchronously on the calling thread

`stage_shard_async()`

- loops over `stage_part_async()`
- still pays synchronous host fetch for cold parts

`swap_shard()`

- stages the incoming shard before releasing the outgoing one
- transient device footprint includes both shards

`convert_single_mtx_to_sharded_coo()`

- scans the MTX file
- partitions rows
- counts part nnz
- allocates per-window COO payload
- parses and copies text entries into host COO buffers
- stores one packfile window at a time

`build_pinned_triplet_to_compressed()`

- copies pinned host triplets to device
- runs device scan/scatter conversion
- copies compressed output back to pinned host buffers
- synchronizes the stream

## User Rules

CellShard will only be fast if the caller behaves accordingly.

- Do not treat `stage_*` as a cheap accessor.
- Do not assume `*_async` means asynchronous disk I/O.
- Keep parts resident on host if you know you will upload them again soon.
- Keep shards resident on device if another kernel pass will reuse them.
- Prefer fewer larger parts over many tiny parts when launch count and copy count matter more than fine-grained scheduling.
- Prefer shard boundaries chosen by resident device bytes, not by aesthetics.
- Avoid repeated `drop -> fetch -> stage` loops over the same hot region.
- Use `load_header()` first, decide the schedule, then fetch and stage only what will actually run.
- Keep graph capture above CellShard after streams, shard ownership, and residency are already stable.
- Treat ingest as a preprocessing path, not a hot runtime path.

Short version:

- metadata helpers are cheap
- conversion launchers over existing device buffers are cheap
- host text ingest is expensive
- packfile fetch is expensive
- host-device staging is very expensive
- mixing disk I/O and staging in one loop is the most expensive behavior in the library

## Active Layout

The active code layout is intentionally small:

- `src/offset_span.cuh`: flat row-offset search helper
- `src/real.cuh`, `src/types.cuh`: scalar and index policy
- `src/formats/`: per-part matrix layouts
- `src/sharded/`: the sharded matrix mechanism, file layout, and device residency path
- source-format ingest now lives in `Cellerator`
- `src/convert/`: sparse conversion code and kernels
- `src/disk/`: native matrix persistence

The live implementations are in the smaller target homes above. The old flat compatibility paths have been removed.

## Header Use

Primary include:

```cpp
#include "src/CellShard.hh"
```

Lower-level includes:

```cpp
#include "src/sharded/sharded.cuh"
#include "src/disk/matrix.cuh"
#include "src/sharded/disk.cuh"
#include "src/sharded/sharded_device.cuh"
```

## Current Layout

```text
src/
├── CellShard.hh
├── disk/
│   ├── matrix.cu
│   └── matrix.cuh
├── offset_span.cuh
├── real.cuh
├── types.cuh
├── formats/
│   ├── compressed.cuh
│   ├── dense.cuh
│   ├── diagonal.cuh
│   └── triplet.cuh
├── sharded/
│   ├── shard_paths.cu
│   ├── shard_paths.cuh
│   ├── sharded.cuh
│   ├── sharded_device.cuh
│   ├── disk.cu
│   ├── disk.cuh
│   └── sharded_host.cuh
├── convert/
│   ├── compressed_from_coo_raw.cuh
│   ├── coo_from_compressed_raw.cuh
│   ├── compressed_transpose_raw.cuh
│   ├── routes/
│   │   ├── compressed_to_coo.cuh
│   │   ├── compressed_transpose.cuh
│   │   └── coo_to_compressed.cuh
│   └── kernels/
│       ├── csExpand.cuh
│       ├── csScatter.cuh
│       └── transpose.cuh
```

## Notes

- `src/formats/` is now organized by real per-part storage family only: dense, compressed sparse, triplet sparse, and diagonal.
- Each simple format now lives in one file; pure metadata/indexing helpers are `__host__ __device__`, while allocation and cleanup stay explicit host-only functions in the same header.
- `src/sharded/` is the center of the library now: sharded metadata, resharding, file headers, shard path lists, and GPU residency are all in one subsystem.
- Shard boundaries are now part-aligned, because fetch, drop, upload, and release all operate on whole parts.
- source-format ingest was moved out to `Cellerator`, so `CellShard` stays focused on native formats, packfiles, and residency
- `src/disk/` now carries native matrix persistence.
- `src/CellShard.hh` now includes the real format and binary headers directly instead of routing through umbrella headers.
- `src/convert/` is now organized around the three real device-resident conversion engines: COO -> compressed, compressed -> COO, and compressed transpose.
- `src/convert/routes/` holds the format-specific CSR/CSC entrypoints; the top-level `src/convert/*.cuh` files are the generic raw engines.
- `src/convert/kernels/transpose.cuh` is kernel-only; the raw compressed transpose entrypoint lives in `src/convert/compressed_transpose_raw.cuh`.
- The active conversion API is device-resident only; host-staged buffer helpers have been removed.
- The transpose path reuses the existing scatter-head initialization kernel from `src/convert/kernels/csScatter.cuh`; the actual transpose count/scatter kernels remain separate.
- The moved format, conversion, I/O, and device headers are now the real homes for that code.
- The larger scaffold directories still exist in the repo, but they are not the active design target for this library.

## Non-Goals

CellShard should stay narrow.

It should not grow into:

- a full single-cell toolkit
- a biological data interpretation layer
- a modeling framework
- a quantization research playground
- a general distributed orchestration system

If a feature is mainly about biological meaning, RNA workflow policy, or modeling ideas, it should probably live in `Cellerator` instead.
