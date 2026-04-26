# CSPACK v1 Specification

## Purpose

CSPACK is the current shard-pack file family for CellShard runtime caches. It is
generated from `.csh5` and consumed by CellShard/Cellerator runtime paths for
low-overhead shard fetch, staging, and execution.

CSPACK is not the archive format. It is not the full biological dataset. It is
the shard-pack artifact family used by the runtime/cache layer.

## Non-goals

- Not a replacement for AnnData/H5AD as an ecosystem interchange format.
- Not a metadata-rich analysis container.
- Not the durable canonical dataset source of truth.
- Not a whole-cache manifest format.
- Not the owner/runtime metadata bootstrap format.

## Scope

The current repository implements one shard-pack encoding:

- CSPACK shard files with magic `CSPACK01`

CSPACK files are per-shard files. The cache directory, generation metadata, and owner
metadata live outside the file payloads.

## Current Path Conventions

CSPACK files are currently published at paths such as:

```text
packs/plan.<execution_plan_generation>-pack.<pack_generation>-epoch.<service_epoch>/shard.<id>.cspack
```

The CSPACK container is format-owned by `io/pack/`. Raw matrix payload
vocabulary shared with CSPOOL and future CSHARD staging lives in
`include/CellShard/io/common/raw_format.hh`.

## Supported Layouts

CSPACK files currently support:

- Bucketed Blocked-ELL execution partitions
- Bucketed Sliced-ELL execution partitions
- Quantized Blocked-ELL runtime partitions

Shared current assumptions:

- little-endian host-native encoding
- 64-bit shard/container counts and offsets
- `uint32` rows, cols, nnz, and index metadata inside partition payloads
- one shard per file

## External Metadata Contract

Current CSPACK files do not embed the full metadata contract described in older
planning notes.

Today, metadata is split across:

- the `.csh5` runtime metadata
- `metadata/manifest.txt` in the cache instance directory
- owner/client metadata snapshots used for runtime delivery
- generation-qualified pack paths

In particular, fields such as:

- `generation.canonical`
- `generation.execution_plan`
- `generation.pack`
- `service_epoch`
- owner-node delivery metadata

are currently external to the shard-pack bytes.

`dataset.feature_order_hash` is not currently embedded in the shard-pack file
payloads.

## Shard Pack File Layout

CSPACK files use the magic `CSPACK01` and the following top-level
layout:

```text
[magic: 8 bytes]
[shard_id: uint64]
[partition_count: uint64]
[partition_offsets[partition_count]: uint64]
[payload region]
```

Each payload begins at the corresponding `partition_offsets[i]`.

## Payload Region

### Bucketed Blocked-ELL Execution Partition

Blocked execution partition blobs store:

```text
[rows: uint32]
[cols: uint32]
[nnz: uint32]
[segment_count: uint32]
[segment_row_offsets[segment_count + 1]: uint32]
[exec_to_canonical_rows[rows]: uint32]
[canonical_to_exec_rows[rows]: uint32]
[exec_to_canonical_cols[cols]: uint32]
[canonical_to_exec_cols[cols]: uint32]
[segment_0 raw blocked_ell payload]
...
[segment_n raw blocked_ell payload]
```

Each segment payload is serialized through the shared raw matrix payload format.
The raw `disk_header` and `disk_format_*` tags are defined in
`include/CellShard/io/common/raw_format.hh`; they are not part of the CSPACK
container header.

### Bucketed Sliced-ELL Execution Partition

Sliced execution partition blobs store:

```text
[rows: uint32]
[cols: uint32]
[nnz: uint32]
[segment_count: uint32]
[canonical_slice_count: uint32]
[segment_row_offsets[segment_count + 1]: uint32]
[exec_to_canonical_rows[rows]: uint32]
[canonical_to_exec_rows[rows]: uint32]
[canonical_slice_row_offsets[canonical_slice_count + 1]: uint32]
[canonical_slice_widths[canonical_slice_count]: uint32]
[segment_0 raw sliced_ell payload]
...
[segment_n raw sliced_ell payload]
```

Again, each segment payload is a shared raw matrix payload.

### Quantized Blocked-ELL Runtime Partition

Quantized Blocked-ELL partitions currently use the raw quantized Blocked-ELL
payload format directly as their CSPACK payload. That format stores block
layout, quantized values, and decode parameters after the common raw
`disk_header`.

## Validation Rules

A valid current CSPACK v1 file must:

1. Start with the correct file magic: `CSPACK01`.
2. Keep all recorded offsets within file bounds.
3. Use a payload layout consistent with the declared top-level family.
4. Have a valid `shard_id`, `partition_count`, and
   partition offset table.
5. Have partition payloads that decode successfully under the expected raw
   packfile or execution-partition codec.
6. Be paired with external runtime metadata when generation-aware delivery is
   required.

## Important Clarification

Older design notes described CSPACK v1 as a future container with:

- a metadata table
- a section directory
- a shard directory
- a partition directory
- required in-band generation metadata
- required in-band feature-order hashes

That is not the current implementation in this repository.

The current implementation is simpler:

- CSPACK files use a concrete `CSPACK01` header and partition offsets
- runtime/generation metadata is external
- raw partition payload metadata is carried inline by the shared raw payload
  format in `io/common/raw_format.hh`
