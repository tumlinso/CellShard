#pragma once

// Thin route alias kept so callers can include a shard-specific entrypoint
// while the real implementation lives in the operator/raw bucket layers.
#include "../operators/sharded_major_nnz.cuh"
