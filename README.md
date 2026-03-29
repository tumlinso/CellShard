# CellShard

CellShard is a low-level, header-first library for large omics matrices that are split across many files and too large to treat as one host-resident object. The point is to keep the data partitioned, fetch only the pieces you need, and preserve shard boundaries that are useful for GPU staging and distributed work.

Build output is ignored in the repo-level `.gitignore`.

The active code layout is intentionally small:

- `src/matrix.cuh`: umbrella matrix header
- `src/real.cuh`, `src/types.cuh`: scalar and index policy
- `src/formats/`: matrix layouts and sharded container
- `src/convert/`: sparse conversion code and kernels
- `src/io/binary/`: binary matrix and sharded-header I/O
- `src/device/cache/`: GPU staging/cache path

The old flat paths under `src/` and `src/sparse/` are now compatibility shims where practical. The live implementations have been copied into the smaller target homes above.

## Header Use

Primary include:

```cpp
#include "src/CellShard.hh"
```

Lower-level includes:

```cpp
#include "src/matrix.cuh"
#include "src/io/binary/matrix_io.cuh"
#include "src/device/cache/shard_cache.cuh"
```

`matrix.cuh` no longer drags the explicit CSR conversion buffer into the umbrella path. Buffer headers stay opt-in.

## Current Layout

```text
src/
├── CellShard.hh
├── matrix.cuh
├── real.cuh
├── types.cuh
├── formats/
│   ├── dense.cuh
│   ├── matrix_base.cuh
│   ├── sharded.cuh
│   └── sparse/
│       ├── buffer/
│       │   └── csr_buffer.cuh
│       ├── coo.cuh
│       ├── csc.cuh
│       ├── csr.cuh
│       ├── csx.cuh
│       └── dia.cuh
├── convert/
│   ├── csc_to_coo.cuh
│   ├── coo_to_csc.cuh
│   ├── coo_to_csr.cuh
│   ├── coo_to_csx.cuh
│   ├── csr_to_coo.cuh
│   ├── csx_to_coo.cuh
│   └── kernels/
│       ├── cs_expand.cuh
│       └── cs_scatter.cuh
├── io/
│   └── binary/
│       ├── matrix_io.cu
│       └── matrix_io.cuh
└── device/
    └── cache/
        └── shard_cache.cuh
```

## Notes

- `src/sparse/csx.cuh` uses the shared policy from `src/types.cuh` and `src/real.cuh`.
- `src/convert/coo_to_csr.cuh` and `src/convert/coo_to_csc.cuh` are thin wrappers in `cellshard::convert` over the shared csX conversion path.
- `src/convert/csr_to_coo.cuh` and `src/convert/csc_to_coo.cuh` are the matching thin wrappers over the shared csX-to-COO path.
- The moved format, conversion, I/O, and device headers are now the real homes for that code.
- The larger scaffold directories still exist in the repo, but they are not the active design target for this library.
