# Repository Guidelines

CellShard is a performance-first, biology-native compiler and physical-runtime
system. It compiles recurrent biological organization into exact reusable
artifacts, placement, acquisition, residency, and transport plans. Its purpose
is to reduce total execution, movement, reconstruction, and communication cost,
not to turn the project into an artifact catalog or generic tensor runtime.

The frozen successor contract is
`docs/JBC/CELLSHARD_JBC_SUCCESSOR_CHARTER_V1.md`. Existing implementation and
format documents remain authoritative compatibility descriptions; new work
must converge toward that charter without presenting unfinished transitions as
complete.

Legacy row shards are compatibility/delivery machinery, not the semantic
definition of domains or partitions. Preserve opaque producer payload bytes
while allowing explicit CellShard-relevant operational metadata. Do not put
paths, replicas, service epochs, or placement epochs into immutable image
identity.

## Scope And Ownership

CellShard owns global biological recurrence compilation and physical
realization. This includes:

- bounded proposal evidence and independent exact coverage certification
- atom composition, grammars, bases, superatoms, partials, and lineage
- global physical views, topology, placement, routing, acquisition, caching,
  residency, transport, leases, pins, readiness, and reconstruction
- persisted dataset, partition, shard, image, extent, catalog, snapshot, and
  source metadata
- `.csh5` canonical dataset storage and `.cspack` compatibility publication,
  delivery, and fetch paths
- `.cspool` bounded local ingest spool files
- experimental `.cshard` archive inspection, validation, conversion, and row reads
- optional bounded ingest, export helpers, and Python bindings

Approximate evidence proposes only. Independently certified exact coverage is
required before execution. Shape, offsets, ordinal position, paths, pointers,
devices, replicas, or service/placement epochs never establish biological
identity.

Sparse matrix representation primitives such as Blocked-ELL, Sliced-ELL, and
quantized Blocked-ELL are Cellerator-owned compute/layout types. CellShard
exposes temporary compatibility headers and may serialize, cache, stage, and
ship those payloads, but it should not be the source of truth for their compute
layout policy.

Do not move model training, Torch/libtorch integration, trajectory inference or
scientific policy, or ML compute into CellShard. Those belong in Cellerator.
CellShard may compile caller-supplied trajectory/lineage structure for exact
reuse without inferring pseudotime or fate. Do not move biological
preprocessing policy, normalization decisions, marker/QC semantics, or workflow
policy into CellShard. Those belong in BioPrep. Neighbor-caller
orchestration and query policy belong in CellShardNeighbors.

Cellerator is the migration target for generic sparse compute primitives.
CellShard may temporarily host compatibility runtime wrappers while callers are
migrated. Biological feature groups may be passed in as ordinary feature masks,
but CellShard should not define biological QC policy.

## Compiler, Runtime, And Format Posture

The successor path is:

```text
typed observations and relations
  -> bounded recurrence evidence
  -> independent exact certification
  -> compiled catalogs/images/extents/routes
  -> acquisition and residency
  -> Cellerator-owned numerical execution
```

No single archive or pack is the universal successor path. Selection includes
publication, acquisition, transfer, reconstruction, synchronization, and
expected reuse. Steady-state execution performs no discovery, catalog parsing,
hidden allocation, global sorting, topology search, or silent canonicalization.

Compatibility remains explicit: `.csh5` is the current durable canonical
source; `.cspack` is a generated execution-container family; `.cspool` is a
machine-local ingest artifact; `.cshard` is experimental standby. Preserve
their byte contracts and reference paths until versioned replacements have
adapters, migrated consumers, and differential fixtures.

When changing format or runtime behavior, read the relevant docs first:

- `README.md` for ownership, layout, runtime, and Python posture
- `SUPPORT.md` for the current supported surface
- `docs/SPEC_CSPACK_V1.md` for the current CSPACK byte-level contract
- `docs/SPEC_CSPOOL_V1.md` and `docs/SPEC_CSHARD_V1.md` for spool/archive work
- `docs/FORMAT_ROLES.md` for ecosystem format boundaries
- `docs/JBC/CELLSHARD_JBC_SUCCESSOR_CHARTER_V1.md` for the frozen successor
  ownership and convergence contract
- `docs/JBC/CS_JBC_B02_SOURCE_TRANSITION_MAP.md` for source-level migration
  classification
- `docs/PARKING_LOT.md` for deferred ideas

## Source Layout

Public reusable surfaces live under `include/CellShard/`. Implementation lives
under `src/`, `export/`, and `python/`.

Key areas:

- `include/CellShard/formats/`: compatibility headers for Cellerator-owned
  sparse layouts plus CellShard-local dense/compressed fallback layouts
- `include/CellShard/runtime/`: sharded layout, storage dispatch, host/device
  staging, masking, and local distributed helpers
- `include/CellShard/compiler/`: reserved successor recurrence compiler
  contracts and cold builders; populated only by owning tasks
- `include/CellShard/artifact/atom_store/`: reserved successor immutable
  catalog/image/extent publication contracts
- `include/CellShard/runtime/v2/`: reserved successor acquisition, topology,
  placement, residency, and transport contracts
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
- `tests/jbc/` and `bench/jbc/`: reserved exact-oracle, adversarial,
  differential, complete-cost, and no-promotion evidence

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

Keep CellShard changes inside the CellShard repository. Do not hide
uncommitted CellShard work in another checkout or coordination repository.

Useful checks:

```bash
git status --short --branch
git diff --stat
```

For cross-repo work, land and verify the CellShard implementation first, then
update each consuming repository independently when the task explicitly asks
for that integration.
