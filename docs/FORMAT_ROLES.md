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

`.csh5` is the current durable CellShard container. It may store canonical sparse data, metadata, preprocessing state, shard information, and enough information to regenerate CSPACK files.

It is useful because HDF5 is accessible and inspectable, but it should not be treated as the long-term hot execution format.

For multiome datasets, `.csh5` owns archive-level assay semantics: one global
observation table, per-assay sparse matrices and feature tables, and row maps
that connect global observations to assay-local rows.

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

`.cspack` is the shard-pack runtime format for CellShard and Cellerator.

It is generated from `.csh5` and consumed by CellShard/Cellerator runtime paths
for low-overhead sparse fetch, staging, and GPU execution.

This is the current priority.

CSPACK remains a single-assay execution artifact. Multiome execution uses
multiple coordinated CSPACK files plus manifest metadata, not one mixed-modality
CSPACK payload.

### Use `.cspack` for

- GPU-native runtime execution.
- Blocked-ELL sparse matrix payloads.
- Fast shard loading.
- Memory-mapped inspection.
- Low-overhead device staging.
- Reproducible execution layouts.
- Benchmarkable runtime artifacts.
- Per-assay sparse payloads that can be co-sharded by global observation range.

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

- Per-shard files with magic `CSPACK01`.
- Bucketed Blocked-ELL execution partitions.
- Bucketed Sliced-ELL execution partitions.
- Quantized Blocked-ELL runtime partitions.
- Little-endian host-native encoding.
- 64-bit shard/container counts and offsets.
- `uint32` rows, cols, nnz, and index metadata inside partition payloads.
- Raw matrix payload metadata carried inline by `io/common/raw_format.hh`.
- Generation-qualified pack paths and external runtime metadata.

Not supported in v1:

- Rich in-band metadata tables.
- Explicit in-band section directories.
- In-band shard and partition directories beyond the current offset table.
- Required in-band feature-order hashes.
- Required in-band generation identifiers.
- Distributed packs.
- Trajectory indices.
- Dynamic graph updates.
- Training checkpoints.
- Rich observation metadata.
- Multi-assay biological semantics inside the matrix payload.

### Policy

`.cspack` is the performance contract.

Cellerator hot paths should consume `.cspack`, not `.csh5` or `.h5ad`.

---

## `.cshard`

### Role

Experimental standby native archive format.

### Purpose

`.cshard` is the HDF5-free native archive format under active standby
development for CellShard.

It is implemented enough to inspect, validate, convert from `.csh5`, and read
row batches directly. It is not the production durable format yet and it does
not make existing `.csh5` runtimes prefer `.cshard`.

The v1 layout is offset-based and ELL-family native. Blocked-ELL is the default
stored layout because current CellShard/Cellerator execution policy is
Blocked-ELL-first. Sliced-ELL remains a native ELL-family layout. CSR is
allowed only as an explicit compatibility/export fallback.

Its assay directory records measurement semantics using the same numeric
vocabulary as `cudaBioTypes`, while keeping matrix payload sections in the
existing optimized sparse layouts.

The public multi-assay `.cshard` path is archive-oriented: it writes one global
observation table, per-assay feature-table ranges, exact or partial
observation-level row maps, and either CSR fallback payloads or assay-local
optimized bucketed Blocked-ELL/Sliced-ELL shard blobs. CSPACK remains
single-assay; coordinated multiome execution still uses separate CSPACK
artifacts plus manifest metadata.

### Use `.cshard` for

- Experimental native durable archives.
- HDF5-free canonical storage.
- Generation-aware biological datasets.
- Columnar observation and feature metadata.
- Pack manifests.
- Preprocessing provenance.
- Regeneration of `.cspack`.
- Direct archive inspection and sparse row reads without `.cspack`.

### Do not use `.cshard` for

- Replacing `.csh5` in production paths yet.
- Replacing `.cspack`.
- Owning model weights or training checkpoints.
- Hiding execution behind CSR as the conceptual default.

### Policy

`.cshard` is standby v1. Keep `.csh5` as the current compatibility and durable
archive source until `.cshard` has broader mileage.

`.cshard` may record external references, including future references to model
artifacts, but it must not own model weights.

The current development order is:

```text
.csh5 compatibility -> .cspack execution cache -> .cshard standby native archive
```

---

## `.cellerator`

### Role

Cellerator-owned future model/checkpoint artifact.

### Policy

`.cellerator` belongs to Cellerator, not CellShard.

CellShard may record external references to future `.cellerator` artifacts, but
it must not define, write, own, or validate that model/checkpoint format. Do
not write model weights, optimizer state, or training checkpoints into
`.cshard` to approximate it.

---

## `.csbundle`

### Role

Reserved future bundle/distribution artifact.

### Policy

`.csbundle` does not exist yet. It is reserved for a future packaging role and
is not implemented in this milestone.

---

## Summary Table

| Format | Role | Hot Path | Priority |
|---|---|---:|---:|
| `.h5ad` | Ecosystem interchange | No | Ongoing compatibility |
| `.csh5` | Current CellShard archive | No | Current support |
| `.cspool` | Local ingest spool | No | Narrow local support |
| `.cspack` | Execution artifact | Yes | Immediate priority |
| `.cshard` | Experimental standby native archive | No | Read/convert standby |
| `.cellerator` | Cellerator-owned model artifact | No | Outside CellShard |
| `.csbundle` | Reserved bundle artifact | No | Future only |

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
