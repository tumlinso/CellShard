# CSPOOL v1 Specification

## Purpose

CSPOOL is the bounded local ingest-spool format for CellShard and Cellerator.
It is generated during source ingest before final `.csh5` assembly and exists to
avoid rereading an expensive source matrix while preserving `.csh5` as the
durable source of truth.

CSPOOL is not the archive format. It is not the execution-pack format. It is
the local ingest staging artifact.

## Non-goals

- Not a replacement for `.csh5`, `.cshard`, or `.h5ad`.
- Not the hot execution format consumed by Cellerator.
- Not a metadata-rich analysis container.
- Not a stable remote interchange format.
- Not a multi-layout sparse container.
- Not intended for long-term compatibility promises beyond the local ingest path.

## v1 Supported Layout

CSPOOL v1 supports exactly one layout:

- Sliced-ELL
- FP16 values
- `uint32` rows, cols, nnz, slice metadata, and column indices
- little-endian host-native encoding
- fixed per-file header
- one partition payload per file

CSPOOL v1 is intentionally narrow and machine-local. It should be treated as a
bounded ingest cache, not as a durable published interface.

## Filename Convention

The current ingest path writes one file per partition under a spool root such as:

```text
<dataset>.csh5.ingest_spool/part.00000000.cspool
```

The spool root naming is an implementation convention. The `.cspool` suffix is
the part-file format marker.

## File Layout

```text
[disk_header]
[slice_count]
[slice_row_offsets]
[slice_widths]
[col_idx]
[values]
```

## Required In-Band Fields

Every CSPOOL v1 file must encode:

- layout format tag
- rows
- cols
- nnz
- slice_count

There is no separate metadata table in v1.

## Header

The fixed header matches the current CellShard raw disk codec:

- `format` : `uint8`
- `rows` : `uint32`
- `cols` : `uint32`
- `nnz` : `uint32`

For CSPOOL v1, `format` must equal the CellShard `disk_format_sliced_ell` tag.

## Payload Region

Immediately after the header, the file stores:

- `slice_count` : `uint32`
- `slice_row_offsets[slice_count + 1]` : `uint32`
- `slice_widths[slice_count]` : `uint32`
- `col_idx[total_slots]` : `uint32`
- `values[total_slots]` : FP16

`total_slots` is derived as:

```text
sum((slice_row_offsets[i + 1] - slice_row_offsets[i]) * slice_widths[i])
for i in [0, slice_count)
```

## Validation Rules

A valid CSPOOL v1 file must:

1. Have the correct `disk_format_sliced_ell` format tag.
2. Use the declared v1 layout constraints: Sliced-ELL, FP16 values, and `uint32`
   dimensions/indexing.
3. Have `slice_row_offsets[0] == 0`.
4. Have nondecreasing `slice_row_offsets`.
5. Have `slice_row_offsets[slice_count] == rows`.
6. Have payload arrays fully contained within file bounds.
7. Have `total_slots` derived consistently from the slice offsets and widths.
8. Load without touching HDF5.
