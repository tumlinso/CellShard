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
[assay directory]
[pairing directory]
[matrix directory]
[obs table directory]
[var table directory]
[section directory]
[aligned payload region]
```

The header points to every directory. The section directory points to payload
sections. Matrix descriptors refer to section ids rather than embedding offsets
directly. For multi-assay archives, the observation table is dataset-global and
each assay descriptor owns the matrix descriptors, feature-table range,
feature-order hash, and row-map sections for one modality.

## Required Header Fields

A valid v1 file must have:

- matching magic and major version
- matching endian tag
- file size equal to the header's `file_size`
- nonzero required feature-order hash
- supported canonical layout
- valid section bounds
- contiguous matrix partition row coverage
- matrix descriptor `nnz` values summing to the relevant assay or file `nnz`
- obs table row count equal to the global observation count
- per-assay feature table row count equal to that assay's matrix columns

## Assays And Pairing

`.cshard` now has fixed-width assay descriptors for measurement-agnostic
archives. The numeric semantic fields mirror `cudaBioTypes`:

- modality
- observation unit
- feature type
- value semantics
- processing state
- matrix row and column axis meaning
- feature namespace

Each assay descriptor records its own matrix descriptor range, feature-table
range, row count, feature count, nonzero count, feature-order hash, and two
required row-map sections for multi-assay archives:

- global observation row to assay-local row
- assay-local row to global observation row

The invalid assay row sentinel is `0xffffffff`. This allows exact and partial
RNA+ATAC-style pairing without forcing the archive to drop observations that are
missing one modality. Pairing metadata records the dataset-level pairing kind;
v1 execution pairing supports exact and partial observation-level pairing.
Donor- and sample-level relationships are metadata only until a future
execution contract is defined.

Multi-assay writers may store CSR fallback payloads or already optimized
bucketed Blocked-ELL/Sliced-ELL shard blobs. The optimized writers do not
convert CSR into ELL inside `.cshard`; callers provide the assay-local optimized
shards in memory. Readers resolve global observations through the loaded
row-map sections and read assay rows against only that assay's matrix
descriptor range. Optimized shard descriptors also carry the shared global row
window in `aux0`/`aux1`, while `row_begin` and `rows` remain the assay-local row
range.

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

CSPACK payloads remain single-assay matrix artifacts. Multiome pack metadata
associates each pack with an `assay_id`, generation, shard id, global
observation row range, local assay row range, and pack path. Paired lookup is
therefore resolved by archive or manifest row maps before fetching rows from the
per-assay CSPACK payloads.

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
