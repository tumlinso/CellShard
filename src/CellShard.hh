#pragma once

// Umbrella include for the active CellShard surface.
// This intentionally pulls in:
// - type/layout policy
// - per-part matrix formats
// - sharded metadata and packfile persistence
// - host fetch/drop helpers
// - device staging helpers
// - local multi-GPU helpers
// - MTX ingest helpers
//
// Downstream code that wants a smaller compile surface should include only the
// specific headers it needs instead of this umbrella.

#include "types.cuh"
#include "offset_span.cuh"
#include "formats/dense.cuh"
#include "formats/compressed.cuh"
#include "formats/triplet.cuh"
#include "formats/diagonal.cuh"
#include "sharded/sharded.cuh"
#include "sharded/sharded_host.cuh"
#include "sharded/shard_paths.cuh"
#include "disk/matrix.cuh"
#include "sharded/disk.cuh"
#include "sharded/sharded_device.cuh"
#include "sharded/distributed.cuh"
#include "ingest/mtx/compressed_parts.cuh"
