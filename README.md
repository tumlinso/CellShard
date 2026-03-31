# CellShard

CellShard is a low-level, header-first library for large omics matrices that are split across many files and too large to treat as one host-resident object. The point is to keep the data partitioned, fetch only the pieces you need, and preserve shard boundaries that are useful for GPU staging and distributed work.

Build output is ignored in the repo-level `.gitignore`.

The active code layout is intentionally small:

- `src/offset_span.cuh`: flat row-offset search helper
- `src/real.cuh`, `src/types.cuh`: scalar and index policy
- `src/formats/`: per-part matrix layouts
- `src/sharded/`: the sharded matrix mechanism, file layout, and device residency path
- `src/convert/`: sparse conversion code and kernels
- `src/io/binary/`: binary per-matrix I/O

The live implementations are in the smaller target homes above. The old flat compatibility paths have been removed.

## Header Use

Primary include:

```cpp
#include "src/CellShard.hh"
```

Lower-level includes:

```cpp
#include "src/sharded/sharded.cuh"
#include "src/io/binary/matrix_file.cuh"
#include "src/sharded/sharded_file.cuh"
#include "src/sharded/sharded_device.cuh"
```

## Current Layout

```text
src/
├── CellShard.hh
├── offset_span.cuh
├── real.cuh
├── types.cuh
├── formats/
│   ├── compressed.cuh
│   ├── compressed_host.cuh
│   ├── dense.cuh
│   ├── dense_host.cuh
│   ├── diagonal.cuh
│   ├── diagonal_host.cuh
│   ├── triplet.cuh
│   └── triplet_host.cuh
├── sharded/
│   ├── shard_paths.cu
│   ├── shard_paths.cuh
│   ├── sharded.cuh
│   ├── sharded_device.cuh
│   ├── sharded_file.cu
│   ├── sharded_file.cuh
│   └── sharded_host.cuh
├── convert/
│   ├── compressed_from_coo_raw.cuh
│   ├── coo_from_compressed_raw.cuh
│   ├── compressed_transpose_raw.cuh
│   ├── routes/
│   │   ├── compressed_to_coo.cuh
│   │   ├── compressed_transpose.cuh
│   │   └── coo_to_compressed.cuh
│   └── kernels/
│       ├── csExpand.cuh
│       ├── csScatter.cuh
│       └── transpose.cuh
├── io/
│   └── binary/
│       ├── matrix_file.cu
│       └── matrix_file.cuh
```

## Notes

- `src/formats/` is now organized by real per-part storage family only: dense, compressed sparse, triplet sparse, and diagonal.
- Each family exposes the low-level layout file directly next to its host-only mutation file; there is no generic helper layer in the active tree.
- `src/sharded/` is the center of the library now: sharded metadata, resharding, file headers, shard path lists, and GPU residency are all in one subsystem.
- `src/io/binary/` now only carries single-matrix disk I/O.
- `src/CellShard.hh` now includes the real format and binary headers directly instead of routing through umbrella headers.
- `src/convert/` is now organized around the three real device-resident conversion engines: COO -> compressed, compressed -> COO, and compressed transpose.
- `src/convert/routes/` holds the format-specific CSR/CSC entrypoints; the top-level `src/convert/*.cuh` files are the generic raw engines.
- `src/convert/kernels/transpose.cuh` is kernel-only; the raw compressed transpose entrypoint lives in `src/convert/compressed_transpose_raw.cuh`.
- The active conversion API is device-resident only; host-staged buffer helpers have been removed.
- The transpose path reuses the existing scatter-head initialization kernel from `src/convert/kernels/csScatter.cuh`; the actual transpose count/scatter kernels remain separate.
- The moved format, conversion, I/O, and device headers are now the real homes for that code.
- The larger scaffold directories still exist in the repo, but they are not the active design target for this library.
