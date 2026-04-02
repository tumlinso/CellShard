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

## Active Layout

The active code layout is intentionally small:

- `src/offset_span.cuh`: flat row-offset search helper
- `src/real.cuh`, `src/types.cuh`: scalar and index policy
- `src/formats/`: per-part matrix layouts
- `src/sharded/`: the sharded matrix mechanism, file layout, and device residency path
- `src/ingest/`: limited source-to-native conversion support
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
├── ingest/
│   ├── scan.cuh
│   ├── common/
│   ├── mtx/
│   └── series/
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
- `src/ingest/scan.cuh` is the active sequential text scanner for source conversion.
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
