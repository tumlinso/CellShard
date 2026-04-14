# CellShard Support Matrix

CellShard `0.1.x` is intentionally narrow.

## Stable Surface

- Installed CMake package components:
  - `CellShard::headers`
  - `CellShard::inspect`
  - `CellShard::runtime` when CUDA is enabled
  - `CellShard::export`
  - `CellShard::h5ad_python` when Python is enabled
- Python package/module:
  - `open`
  - `Dataset`
  - `open_dataset`
  - `DatasetFile`
  - `open_dataset_owner`
  - `DatasetOwner`
  - `bootstrap_dataset_client`
  - `DatasetClient`
  - `load_dataset_summary`
  - `load_dataset_as_csr`
  - `load_dataset_rows_as_csr`
  - `load_dataset_global_metadata_snapshot`
  - `serialize_global_metadata_snapshot`
  - `deserialize_global_metadata_snapshot`
  - `DatasetFile.materialize_partition`
  - `write_h5ad`

## Supported Environment

- OS / arch: Linux `x86_64`
- CMake: `3.24+`
- Python: `3.10` through `3.13`
- HDF5: C library available at build time
- CUDA: optional, but the main runtime/staging path assumes the repo's current CUDA toolchain posture

## Support Notes

- CPU-only builds are supported for inspect/materialize portability and packaging smoke checks, not for the normal high-throughput runtime path.
- CUDA-enabled release validation should use the same host/toolchain class as the main CellShard development environment.
- Binary wheels are Linux-only.
- The standalone `cellshardH5adExport` app is part of CMake builds, but wheel builds package the Python extension only.
- `.csh5` is the canonical archive/container format. High-throughput repeated fetch is expected to run through generated shard `.pack` cache files built from that container.
- The normal runtime path is therefore `bind .csh5` -> `materialize shard pack` -> `fetch from .pack` -> `stage to GPU`, not repeated direct HDF5 payload reads as the final execution substrate.
- Cellerator-side ingest may use a bounded local Blocked-ELL spool to avoid rereading an expensive source MTX before `.csh5` assembly. That spool is an implementation detail, not a supported archival surface.
- Persisted parts and shards remain row-aligned. One cell is not split across parts or shards.

## Not Promised In `0.1.x`

- Windows or macOS support
- Broad ABI portability across arbitrary compilers or CUDA toolchains
- Non-Linux wheels
- Torch or libtorch integration
