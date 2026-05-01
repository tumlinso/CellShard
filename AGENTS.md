# Repository Guidelines

## Scope And Ownership

CellShard is the CellStack storage, layout, pack-delivery, and runtime-staging
layer for large sharded sparse omics matrices. It owns:

- sparse matrix layouts and layout conversion helpers
- partition, shard, and sharded-matrix metadata
- `.csh5` canonical dataset storage
- `.cspack` runtime pack generation, delivery, and fetch paths
- `.cspool` bounded local ingest spool files
- experimental `.cshard` archive inspection, validation, conversion, and row reads
- host fetch/drop, device upload, staging, and local distributed placement helpers
- optional bounded ingest, export helpers, and Python bindings

Do not move model training, Torch/libtorch integration, trajectory logic, or ML
compute into CellShard. Those belong in Cellerator. Do not move biological
preprocessing policy, normalization decisions, marker/QC semantics, or workflow
policy into CellShard. Those belong in CellShardPreprocess. Neighbor-caller
orchestration and query policy belong in CellShardNeighbors.

CellShard runtime masking is a generic sparse compute primitive. Biological
feature groups may be passed in as ordinary feature masks, but CellShard should
not define biological QC policy.

## Runtime And Format Posture

Keep the hot path centered on:

```text
.csh5 -> .cspack -> GPU execution
```

`.csh5` is the durable canonical source and append target. `.cspack` is the
execution-facing runtime artifact. Avoid normalizing direct `.csh5` reads as
the performance path for repeated execution. Owner-side runtime code may build
or refresh packs; executor-facing code should consume published pack generations
when the required runtime artifact exists.

`.cspack` is a shard-pack family, not a metadata-rich archive. Keep rich dataset
metadata in `.csh5` today and future archive formats later. `.cspool` is a
machine-local ingest artifact, not a public archive. `.cshard` is experimental
standby archive work and must not silently replace `.csh5` as the production
source of truth.

When changing format or runtime behavior, read the relevant docs first:

- `README.md` for ownership, layout, runtime, and Python posture
- `SUPPORT.md` for the current supported surface
- `docs/SPEC_CSPACK_V1.md` for the current CSPACK byte-level contract
- `docs/SPEC_CSPOOL_V1.md` and `docs/SPEC_CSHARD_V1.md` for spool/archive work
- `docs/FORMAT_ROLES.md` for ecosystem format boundaries
- `docs/PARKING_LOT.md` for deferred ideas

## Source Layout

Public reusable surfaces live under `include/CellShard/`. Implementation lives
under `src/`, `export/`, and `python/`.

Key areas:

- `include/CellShard/formats/`: concrete sparse/dense matrix layouts
- `include/CellShard/runtime/`: sharded layout, storage dispatch, host/device
  staging, masking, and local distributed helpers
- `include/CellShard/io/`: `.cspack`, `.csh5`, `.cspool`, and `.cshard` public
  entry surfaces
- `include/CellShard/ingest/`: optional source ingest headers
- `include/CellShard/export/`: dataset export and metadata snapshot APIs
- `src/io/csh5/`: `.csh5` backend and runtime pack materialization/fetch
- `src/io/pack/`: packed partition payload codec used by shard `.cspack` caches
- `src/io/cshard/`: experimental HDF5-free archive reader/writer/converter
- `src/runtime/`: runtime layout, storage, host, device, and distributed pieces
- `export/`: non-Torch export helpers split by responsibility
- `python/`: optional pybind module and Python package wrapper
- `tests/`: focused compile, runtime, package, and format checks

## Coding Style

Match the existing C++17/CUDA17 style:

- use 4-space indentation and same-line opening braces
- prefer `snake_case` for files, functions, variables, structs, and CLI flags
- qualify standard-library names with `std::`
- use `.cuh` for CUDA-aware reusable headers and `.cu` for CUDA translation
  units; use `.cc` for plain C++ sources
- keep public APIs explicit about layout, ownership, host/device residency, and
  generation boundaries
- keep CellShard Torch-free and libtorch-free

In performance-sensitive paths, prefer explicit layouts, contiguous buffers,
preallocated storage, pointer-plus-size interfaces, and clear ownership over
generic abstractions that hide allocation, copy, transfer, or launch costs. Do
not add abstraction layers that obscure shard/partition boundaries, pack
generation state, HDF5 access, or device staging behavior.

Use `std::vector`, `std::string`, streams, and other standard-library helpers
freely in cold metadata, validation, export, tests, and Python-binding glue when
they keep code clear. Be conservative with them in hot runtime, ingest,
conversion, staging, and repeated fetch paths.

## Build And Test Commands

Typical local build:

```bash
cmake -S . -B build
cmake --build build -j 4
```

Common validation commands:

```bash
./build/cellShardExportRuntimeTest
./build/cellShardCshardTest
./build/cellShardMaskGroupsRuntimeTest
cmake --build build --target cellShardInspectPackageTest
```

`cellShardMaskGroupsRuntimeTest` exists only when CUDA runtime tests are enabled.
Documentation-only changes normally need docs/status verification rather than a
full build. Run builds or tests when a doc change updates commands, package
surface, generated examples, or behavior claims.

Useful CMake options include:

- `CELLSHARD_ENABLE_CUDA`
- `CELLSHARD_BUILD_TESTS`
- `CELLSHARD_BUILD_EXPORT`
- `CELLSHARD_ENABLE_PYTHON`
- `CELLSHARD_BUILD_INGEST`
- `CELLSHARD_INSTALL_PREPROCESS`

CPU-only builds are supported for inspect/materialize portability and packaging
smoke checks, not for the normal high-throughput runtime path.

Benchmark and profiler runs must be serialized. Record exact commands, hardware,
CUDA/HDF5/toolchain assumptions, and relevant timing/profiler context when
benchmark output matters to the change.

## Python And Package Surface

The Python package is native-first for `.csh5` datasets. Lazy native views are
the default; CSR, SciPy, and Torch conversions are explicit interop paths.
Torch is optional Python interop and must not become a CellShard build
dependency.

Installed CMake package components are intentionally narrow:

- `CellShard::headers`
- `CellShard::inspect`
- `CellShard::runtime` when CUDA is enabled
- `CellShard::export`
- `CellShard::h5ad_python` when Python is enabled

When changing public package components, Python module exports, command-line
surfaces, or install behavior, update `SUPPORT.md`, `README.md`, and the package
consumer checks as needed.

## Documentation And Generated Files

Update behavior documentation in the same change when touching storage, ingest,
runtime packs, export, Python APIs, file formats, or package surfaces. Prefer
updating the source/spec that owns the behavior over patching generated output.

Do not edit generated files, build directories, dependency output, or installed
artifacts. If generated output appears stale, update the source or script that
produces it and record the regeneration command.

## Git Hygiene

Keep CellShard changes inside the CellShard submodule. Do not use the CellStack
root to hide uncommitted submodule work. Before committing a CellStack root
submodule pointer update, check both root status and CellShard status.

Useful checks:

```bash
git status --short --branch
git diff --stat
```

For cross-repo work, land and verify the CellShard implementation first, then
update the CellStack submodule pointer as a separate coordination step unless
the task explicitly asks for both.
