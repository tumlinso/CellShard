#pragma once

// Umbrella include for the active CellShard surface.
// This intentionally pulls in:
// - type/layout policy
// - per-partition matrix formats
// - sharded metadata and `.csh5` dataset persistence
// - host fetch/drop helpers
// - device staging helpers
// - local multi-GPU helpers
//
// Downstream code that wants a smaller compile surface should include only the
// specific headers it needs instead of this umbrella.
//
// Sparse layout posture:
// - `blocked_ell` is the native sparse type
// - `compressed` is an explicit in-memory interop type, not a native `.csh5`
//   file format

#include "types.cuh"
#include "offset_span.cuh"
#include "formats/dense.cuh"
#include "formats/compressed.cuh"
#include "formats/blocked_ell.cuh"
#include "formats/quantized_blocked_ell.cuh"
#include "formats/sliced_ell.cuh"
#include "formats/triplet.cuh"
#include "formats/diagonal.cuh"
#include "sharded/sharded.cuh"
#include "sharded/sharded_host.cuh"
#include "sharded/shard_paths.cuh"
#include "disk/csh5.cuh"
#include "disk/packfile.cuh"
#include "sharded/disk.cuh"

#if CELLSHARD_ENABLE_CUDA
#include "convert/blocked_ell_from_compressed.cuh"
#include "convert/filtered_blocked_ell_to_compressed.cuh"
#include "sharded/sharded_device.cuh"
#include "sharded/distributed.cuh"
#include "bucket/routes/compressed_major_nnz.cuh"
#include "bucket/routes/sharded_major_nnz.cuh"
#include "repack/routes/sharded_blocked_ell.cuh"
#endif
