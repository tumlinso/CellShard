#pragma once

#include <Cellerator/compute/matrix/convert/bucket.cuh>

namespace cellshard {
namespace bucket {

// Compatibility surface: generic major-axis bucket math is implemented by
// CelleratorCompute. CellShard keeps storage/shard wrappers.
using ::cellerator::compute::matrix::convert::bucket::build_major_nnz_bucket_plan_raw;
using ::cellerator::compute::matrix::convert::bucket::build_major_nnz_bucket_plan_custom_raw;
using ::cellerator::compute::matrix::convert::bucket::build_major_nnz_bucket_plan_library_raw;
using ::cellerator::compute::matrix::convert::bucket::build_shard_major_nnz_bucket_plan_raw;
using ::cellerator::compute::matrix::convert::bucket::clamp_bucket_count;
using ::cellerator::compute::matrix::convert::bucket::major_nnz_bucket_plan_view;
using ::cellerator::compute::matrix::convert::bucket::major_nnz_bucket_scan_scratch_bytes;
using ::cellerator::compute::matrix::convert::bucket::major_nnz_bucket_sort_scratch_bytes;
using ::cellerator::compute::matrix::convert::bucket::rebuild_bucketed_shard_compressed_raw;
using ::cellerator::compute::matrix::convert::bucket::rebuild_compressed_major_order_raw;

} // namespace bucket
} // namespace cellshard
