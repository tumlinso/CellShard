# Format Roles

This document defines the intended role of each file format in the CellShard / Cellerator ecosystem.

The guiding rule is simple:

> Archive formats preserve meaning. Pack formats preserve speed.

CellShard should own storage, conversion, sharding, packing, and inspection.
Cellerator should consume execution-ready packs and run biological sparse compute.

---

## `.h5ad`

### Role

Ecosystem interchange format.

### Purpose

`.h5ad` exists so CellShard and Cellerator can communicate with the broader single-cell ecosystem, especially AnnData, Scanpy, scvi-tools, and existing public datasets.

It is not the native performance format.

### Use `.h5ad` for

- Importing public single-cell datasets.
- Exporting results back to standard analysis tools.
- Interoperability with Python biology workflows.
- Sharing data with users who are not using CellShard or Cellerator.

### Do not use `.h5ad` for

- Hot execution.
- GPU-native sparse compute.
- Runtime sharding.
- Blocked-ELL execution.
- Repeated training/inference loops.

### Policy

`.h5ad` is an interchange boundary, not the internal execution substrate.

---

## `.csh5`

### Role

Current CellShard archive format.

### Purpose

`.csh5` is the current durable CellShard container. It may store canonical sparse data, metadata, preprocessing state, shard information, and enough information to regenerate execution packs.

It is useful because HDF5 is accessible and inspectable, but it should not be treated as the long-term hot execution format.

### Use `.csh5` for

- Current CellShard datasets.
- Durable intermediate storage.
- Reproducible preprocessing outputs.
- Metadata-rich local archives.
- Generating `.cspack` files.
- Compatibility while the native archive format is still evolving.

### Do not use `.csh5` for

- Low-latency execution.
- Inner-loop GPU workloads.
- Repeated sparse matrix projection.
- Performance-critical runtime loading.
- Direct Cellerator hot paths.

### Policy

`.csh5` is the current archive format, not the final performance story.

The hot path should move from:

```text
.csh5 -> runtime conversion -> GPU execution
```

to:

```text
.csh5 -> .cspack -> GPU execution
```

---

## `.cspool`

### Role

Bounded local ingest-spool artifact.

### Purpose

`.cspool` is the machine-local pre-archive spool format used during ingest
before final `.csh5` assembly.

It exists to keep ingest bounded-memory and avoid rereading an expensive source
matrix while CellShard finishes shard planning and archive assembly.

It is not a durable archive and not a hot runtime contract.

### Use `.cspool` for

- Local ingest staging.
- One-part-at-a-time spill files during bounded source conversion.
- Reconstructing the final `.csh5` without rereading the original source.
- Narrow layout-specific ingest caches.

### Do not use `.cspool` for

- Long-term storage.
- Published runtime artifacts.
- Distributed delivery.
- Rich metadata.
- Community interchange.

### v1 policy

`.cspool` v1 is intentionally narrow.

Supported in v1:

- Blocked-ELL layout.
- Sliced-ELL layout.
- FP16 values.
- `uint32` rows, cols, nnz, layout metadata, and column-addressing indices.
- Little-endian host-native encoding.
- One partition payload per file.

Not supported in v1:

- Quantized spool files.
- Multi-part container files.
- Rich metadata tables.
- Runtime generation metadata.

### Policy

`.cspool` is a local ingest cache, not a user-facing storage promise.

Current ingest generally emits sliced spool parts, but blocked spool parts are
also in scope for v1 as long as the staged payload is reoptimized as needed
before canonical conversion.

The intended flow is:

```text
source -> .cspool -> .csh5 -> .cspack
```

---

## `.cspack`

### Role

Execution artifact.

### Purpose

`.cspack` is the GGUF-like execution-pack format for CellShard and Cellerator.

It is generated from an archive or interchange source and consumed directly by Cellerator for low-overhead sparse GPU execution.

This is the current priority.

### Use `.cspack` for

- GPU-native runtime execution.
- Blocked-ELL sparse matrix payloads.
- Fast shard loading.
- Memory-mapped inspection.
- Low-overhead device staging.
- Reproducible execution layouts.
- Benchmarkable runtime artifacts.

### Do not use `.cspack` for

- General-purpose analysis metadata.
- Arbitrary user annotations.
- Raw ecosystem interchange.
- Long-term metadata-rich archiving.
- Storing every possible sparse layout.
- Replacing `.h5ad` as a community format.

### v1 policy

`.cspack` v1 should be intentionally narrow.

Supported in v1:

- Blocked-ELL layout.
- Sliced-ELL layout.
- FP16 values.
- `uint32` column-addressing indices.
- Little-endian encoding.
- 64-bit offsets.
- Fixed payload alignment.
- Explicit section directory.
- Explicit shard and partition directories.
- Required feature-order hash.
- Required generation identifiers.

Not supported in v1:

- Quantized Blocked-ELL.
- Distributed packs.
- Trajectory indices.
- Dynamic graph updates.
- Training checkpoints.
- Rich observation metadata.

### Policy

`.cspack` is the performance contract.

Cellerator hot paths should consume `.cspack`, not `.csh5` or `.h5ad`.

---

## `.cshard`

### Role

Future native archive format.

### Purpose

`.cshard` is the possible future HDF5-free native archive format for CellShard.

It should eventually replace `.csh5` if HDF5 overhead becomes too limiting or if a fully controlled binary layout becomes necessary for reproducibility, portability, and performance.

It is not the current priority.

### Use `.cshard` for

- Future native durable archives.
- HDF5-free canonical storage.
- Generation-aware biological datasets.
- Columnar observation and feature metadata.
- Pack manifests.
- Preprocessing provenance.
- Regeneration of `.cspack`.

### Do not use `.cshard` for

- The first public execution milestone.
- Replacing `.cspack`.
- Hot GPU execution unless explicitly materialized into execution sections.
- Avoiding the immediate need to freeze `.cspack`.

### Policy

`.cshard` should wait until `.cspack` is stable.

The correct development order is:

```text
.csh5 compatibility -> .cspack execution -> .cshard native archive
```

---

## Summary Table

| Format | Role | Hot Path | Priority |
|---|---|---:|---:|
| `.h5ad` | Ecosystem interchange | No | Ongoing compatibility |
| `.csh5` | Current CellShard archive | No | Current support |
| `.cspool` | Local ingest spool | No | Narrow local support |
| `.cspack` | Execution artifact | Yes | Immediate priority |
| `.cshard` | Future native archive | Maybe later | Defer |

---

## Development Rule

When adding a feature, decide where it belongs before writing code.

Ask:

1. Is this needed for ecosystem import/export?
   - Put it near `.h5ad` interop.

2. Is this durable biological state or preprocessing provenance?
   - Put it in `.csh5` now, and later `.cshard`.

3. Is this only needed for bounded local ingest staging before archive assembly?
   - Put it in `.cspool`.

4. Is this required for fast GPU execution?
   - Put it in `.cspack`.

5. Is this a future idea that does not serve the current release?
   - Put it in `PARKING_LOT.md`.

---

## Core Principle

CellShard stores and delivers sparse biological matrices in execution-ready layouts.

Cellerator runs biological sparse compute on those layouts.

Archive formats preserve meaning.

Pack formats preserve speed.
