# CSPACK v1 Specification

## Purpose

CSPACK is the execution-pack format for CellShard. It is generated from an archive format such as `.csh5`, `.cshard`, `.h5ad`, or MTX-derived input, and is consumed by Cellerator for low-overhead sparse GPU execution.

CSPACK is not the archive format. It is not the full biological dataset. It is the runtime artifact.

## Non-goals

- Not a replacement for AnnData/H5AD as an ecosystem interchange format.
- Not a general-purpose sparse matrix format.
- Not a training checkpoint format.
- Not a metadata-rich analysis container.
- Not optimized for arbitrary sparse layouts.

## v1 Supported Layouts

CSPACK v1 supports exactly two execution layouts:

- Blocked-ELL
- Sliced-ELL
- FP16 values
- uint32 column-addressing indices
- little-endian
- 64-bit offsets
- fixed alignment

The declared execution layout in metadata determines which partition descriptors
and payload arrays are present for each shard/partition payload.

CSR, quantized sparse layouts, trajectory indices, and distributed packs are
not part of v1.

## File Layout

[header]
[metadata table]
[section directory]
[shard directory]
[partition directory]
[payload region]

## Required Metadata

- format.name
- format.version
- dataset.n_obs
- dataset.n_vars
- dataset.nnz
- dataset.modality
- dataset.feature_order_hash
- matrix.execution_layout
- matrix.value_dtype
- matrix.index_dtype
- execution.target_arch
- execution.block_size
- generation.canonical
- generation.execution_plan
- generation.pack

## Header

TODO: define fixed binary header.

## Section Directory

TODO: define payload section descriptors.

## Partition Directory

TODO: define Blocked-ELL and Sliced-ELL partition descriptors.

## Payload Region

TODO: define how layout-specific arrays are packed.

- Blocked-ELL payloads need descriptors for block column indices, values, row
  maps, and feature permutation arrays.
- Sliced-ELL payloads need descriptors for slice segmentation metadata, column
  indices, values, row maps, and feature permutation arrays.

## Validation Rules

A valid CSPACK v1 file must:

1. Have the correct magic bytes.
2. Match the declared version.
3. Use a supported execution layout and supported dtypes.
4. Have aligned payload offsets.
5. Have section offsets within file bounds.
6. Have a valid feature_order_hash.
7. Have partition descriptors and payload sections that match the declared
   execution layout.
8. Load without touching HDF5.
