# CSHARD v1 Specification

CSHARD v1 is the experimental standby native archive format for CellShard.
It is HDF5-free, little-endian, offset-based, and designed to preserve a
biological sparse dataset plus enough metadata to inspect and regenerate
execution caches.

It does not replace `.csh5` yet. Existing production paths continue to treat
`.csh5` as the durable compatibility source, and `.cspack` remains the optional
derived execution cache.

## Role

`.cshard` is an archive format, not a model artifact and not the hot execution
contract.

Allowed:

- canonical sparse matrix payloads
- observation and feature tables
- provenance records
- optional pack manifests
- external references

Not allowed:

- owned model weights
- optimizer state
- training checkpoints
- treating CSR as the native sparse default

Future `.cellerator` model/checkpoint artifacts are Cellerator-owned, not
CellShard-owned. CSHARD may record an external reference to one, but it must not
own model weights or define the `.cellerator` format. `.csbundle` is a reserved
future bundle role only.

## Encoding

- Magic: `CSHARD01`
- Version: `1.0`
- Endian tag: `0x01020304`
- Header: fixed-width 256-byte POD
- Offsets: unsigned 64-bit byte offsets from file start
- Payload alignment: 64 bytes
- Values: FP16 for v1 sparse payloads
- Indices: unsigned 32-bit for v1 sparse payloads

All fixed records live in `include/CellShard/io/cshard/spec.hh`.

## Top-Level Layout

```text
[header]
[metadata records]
[matrix directory]
[obs table directory]
[var table directory]
[section directory]
[aligned payload region]
```

The header points to every directory. The section directory points to payload
sections. Matrix descriptors refer to section ids rather than embedding offsets
directly.

## Required Header Fields

A valid v1 file must have:

- matching magic and major version
- matching endian tag
- file size equal to the header's `file_size`
- nonzero required feature-order hash
- supported canonical layout
- valid section bounds
- contiguous matrix partition row coverage
- matrix descriptor `nnz` values summing to the header `nnz`
- obs table row count equal to matrix rows
- var table row count equal to matrix columns

## Matrix Layouts

### Blocked-ELL

Blocked-ELL is the v1 default stored layout.

Descriptor fields:

- `layout = matrix_layout_blocked_ell`
- `block_size`
- `ell_width`
- `aux0 = ell_cols`
- `section_a_id`: `uint32` block-column indices
- `section_b_id`: FP16 values

Rows are read directly from the row's block-index row and value row. Empty
block slots use `blocked_ell_invalid_col`.

### Sliced-ELL

Sliced-ELL is a native ELL-family layout in v1.

Descriptor fields:

- `layout = matrix_layout_sliced_ell`
- `slice_count`
- `aux0 = total_slots`
- `section_a_id`: `uint32` slice row offsets
- `section_b_id`: `uint32` slice widths
- `section_c_id`: `uint32` column indices
- `section_d_id`: FP16 values

Empty slots use `sliced_ell_invalid_col`.

### CSR

CSR is explicitly secondary. It exists for compatibility and export fallback
tests, not as the preferred archive representation.

Descriptor fields:

- `layout = matrix_layout_csr`
- `section_a_id`: `uint64` row pointer
- `section_b_id`: `uint32` column indices
- `section_c_id`: FP16 values

## Tables

Observation and feature metadata are columnar. v1 supports:

- text columns: `uint64` string offsets plus byte payload
- float32 columns
- uint8 columns

The feature table should include a stable feature id or name column. The reader
validates the feature-order hash when that column is present.

## Pack Manifest

The header and POD structs reserve a pack-manifest directory. It may describe
derived `.cspack` files, but v1 `.cshard` readers must not require a pack
manifest for direct inspection or row reads.

## Current Implementation

Implemented commands:

```text
cellshard cshard inspect dataset.cshard
cellshard cshard validate dataset.cshard
cellshard cshard read-rows dataset.cshard --start 0 --count 16
cellshard cshard convert input.csh5 --out output.cshard
```

Implemented Python surface:

```python
ds = cellshard.open("dataset.cshard")
ds.describe()
ds.var.head(5)
ds.obs.head(5)
ds.read_rows(0, 1024)
```
