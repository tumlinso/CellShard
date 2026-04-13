# CellShard

CellShard is a low-level, header-first library for large sharded sparse omics matrices.

Its scope is narrow:

- represent sparse matrices as partitions and shards
- persist them in CellShard storage formats
- load metadata without materializing full payloads
- fetch and drop host-side partitions
- stage partitions and shards to GPU
- provide the sparse conversion and layout helpers needed for that runtime

CellShard is a storage and staging layer. It is not the analysis toolkit, model layer, or Torch integration layer. Those live on the Cellerator side.

## What CellShard Owns

- sparse matrix formats such as `blocked_ell`, `compressed`, `dense`, and `triplet`
- sharded matrix metadata and layout helpers
- on-disk matrix and series persistence
- host fetch and drop operations
- device upload and staging helpers
- local multi-GPU shard distribution support
- optional export helpers and optional Python bindings

CellShard does not own:

- preprocessing and normalization workflows
- clustering, embedding, or neighbor analysis APIs
- model training or inference
- Torch or libtorch integration

## Source Layout

CellShard is mainly a header-first C++/CUDA library. Most reusable surfaces are defined in headers under `src/`.

Common file roles:

- `.cuh`: CUDA-aware headers for reusable library surfaces
- `.cu`: CUDA translation units
- `.cc`: C++ translation units
- `.hh`: used rarely here compared with `.cuh`; most public interfaces are CUDA-aware

Naming is mostly `snake_case` for files, functions, variables, and structs.

Main source areas:

- `src/CellShard.hh`: umbrella include for the main library surface
- `src/formats/`: concrete matrix layouts such as `compressed`, `blocked_ell`, `dense`, `triplet`, and `diagonal`
- `src/sharded/`: sharded metadata, host fetch/drop, device staging, disk backends, and local multi-GPU helpers
- `src/disk/`: standalone matrix persistence helpers
- `src/convert/`: sparse conversion helpers and raw conversion kernels
- `src/bucket/`: row/major-axis nnz bucketing helpers
- `src/offset_span.cuh`: small boundary and offset-span helpers
- `export/`: non-Torch export surfaces such as series export and H5AD writing
- `python/`: optional pybind module and Python package wrapper
- `tests/`: focused runtime and package-consumer checks

## Core Concepts

- `partition`: one stored matrix chunk with explicit row bounds
- `shard`: a group of partitions used as the higher-level fetch/staging unit
- `blocked_ell`: the preferred sparse execution and persistence layout
- `compressed`: the fallback row-compressed path used where CSR-style semantics are still needed
- `sharded<T>`: metadata plus optional loaded payload pointers for a partitioned matrix collection
- `shard_storage`: the bound storage backend used for lazy fetch/materialization

The intended workflow is:

1. load metadata or bind a storage backend
2. inspect partitions and shard boundaries
3. fetch host partitions or shards as needed
4. optionally upload or stage them to GPU
5. run higher-level compute outside CellShard

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

If CellShard is configured with `CELLSHARD_ENABLE_CUDA=OFF`, the install does not export `CellShard::runtime`.

## Quick Orientation

If you are browsing the code for the first time:

- start with `src/CellShard.hh`
- read `src/formats/` to understand the matrix types
- read `src/sharded/sharded.cuh` and `src/sharded/sharded_host.cuh` for the host-side model
- read `src/sharded/sharded_device.cuh` for upload and staging
- read `src/sharded/series_h5.cuh` if you care about `.csh5` series storage
- read `src/sharded/distributed.cuh` if you care about local multi-GPU placement

## Notes

- Blocked-ELL is the preferred sparse layout for persisted execution and staging.
- CellShard stays Torch-free by design.
- Build output is ignored in the repo-level `.gitignore`.
