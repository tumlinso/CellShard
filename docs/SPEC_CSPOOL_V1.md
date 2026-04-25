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
- Not a general-purpose sparse container.
- Not intended for long-term compatibility promises beyond the local ingest path.

## v1 Supported Layouts

CSPOOL v1 supports exactly two layouts:

- Blocked-ELL
- Sliced-ELL
- FP16 values
- `uint32` rows, cols, nnz, layout metadata, and column-addressing indices
- little-endian host-native encoding
- fixed per-file header
- one partition payload per file

CSPOOL v1 is intentionally narrow and machine-local. It should be treated as a
bounded ingest cache, not as a durable published interface.

CSPOOL files may be emitted in either layout during ingest. A blocked or sliced
spool artifact is valid as long as the ingest pipeline reoptimizes as needed
before converting the staged payload into the canonical artifact.

## Filename Convention

The current ingest path writes one file per partition under a spool root such as:

```text
<dataset>.csh5.ingest_spool/part.00000000.cspool
```

The spool root naming is an implementation convention. The `.cspool` suffix is
the part-file format marker.

## File Layout

Blocked-ELL payload:

```text
[disk_header]
[block_size]
[ell_cols]
[block_col_idx]
[values]
```

Sliced-ELL payload:

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
- layout-specific descriptor fields

There is no separate metadata table in v1.

## Header

The fixed header matches the current CellShard raw disk codec:

- `format` : `uint8`
- `rows` : `uint32`
- `cols` : `uint32`
- `nnz` : `uint32`

For CSPOOL v1, `format` must equal either the CellShard
`disk_format_blocked_ell` tag or the CellShard `disk_format_sliced_ell` tag.

## Payload Region

Blocked-ELL files store:

- `block_size` : `uint32`
- `ell_cols` : `uint32`
- `block_col_idx[row_blocks * ell_width]` : `uint32`
- `values[rows * ell_cols]` : FP16

where:

```text
row_blocks = ceil(rows / block_size)
ell_width = ell_cols / block_size
```

Sliced-ELL files store:

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

1. Have either the `disk_format_blocked_ell` or `disk_format_sliced_ell`
   format tag.
2. Use the declared v1 layout constraints: Blocked-ELL or Sliced-ELL, FP16
   values, and `uint32` dimensions/indexing.
3. For Blocked-ELL files, have nonzero layout metadata only when the payload is
   nonempty, and have payload array sizes consistent with `rows`, `block_size`,
   and `ell_cols`.
4. For Sliced-ELL files, have `slice_row_offsets[0] == 0`.
5. For Sliced-ELL files, have nondecreasing `slice_row_offsets`.
6. For Sliced-ELL files, have `slice_row_offsets[slice_count] == rows`.
7. Have payload arrays fully contained within file bounds.
8. Load without touching HDF5.
