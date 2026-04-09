#pragma once

// Thin route alias kept so callers can include a bucket-specific entrypoint
// while the real implementation lives in the operator/raw bucket layers.
#include "../operators/major_nnz.cuh"
