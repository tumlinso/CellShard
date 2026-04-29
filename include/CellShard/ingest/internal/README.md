# `ingest`

Source-format ingest and dataset conversion live here.

This area owns MTX parsing, one-pass preprocessing/filtering, manifest-driven
dataset ingest, and the shared feature/barcode/metadata tables needed to emit
immutable canonical matrices into `CellShard`. Keep parsing, staging, and
emission boundaries explicit here.

## Current Dataset Ingest Posture

- Cellerator finishes row/column filtering before `CellShard` emission.
- The emitted matrix is immutable for that canonical generation.
- CellShard owns final blocking, bucketing, rebucketing, pack generation, and `.csh5` assembly.
- In single-machine mode, the emitted dataset still runs through an owner-side `.csh5` reader/coordinator and published pack generations rather than direct ad hoc file mutation.
- In distributed mode, ingest still targets the owner-side `CellShard` builder; executor nodes consume delivered pack artifacts rather than their own copied `.csh5`.
- Parts are observation-major row ranges. One cell may not be split across parts.
- Shards are contiguous ranges of whole parts. One cell and one part may not be split across shards.
- `convert_window_bytes` bounds temporary MTX conversion windows.
- `target_shard_bytes` sizes persisted runtime shards independently from ingest windows.

The ingest path is intentionally bounded-memory rather than zero-memory or
whole-dataset-in-RAM:

1. scan MTX row counts and plan row-aligned parts
2. stream bounded part windows from the source MTX
3. apply filtering and canonical metadata alignment before emission
4. emit immutable canonical sparse parts into CellShard-owned build/spool paths
5. let CellShard finalize shard offsets, blocking, bucketing, and pack planning
6. assemble `dataset.csh5` from local `.cspool` parts instead of rereading the source MTX
7. optionally warm the active CellShard pack generation from the written `dataset.csh5`

The local spool is effectively a backwards cache for ingest: it avoids paying a
second expensive source read while still keeping the authoritative output as
`dataset.csh5`. The per-part spool artifacts use the `.cspool` suffix.
