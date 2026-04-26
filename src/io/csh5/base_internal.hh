#pragma once

// Private shared implementation for `.csh5` translation units.
// Keep this include local to `src/io/csh5`; it is not part of the public header surface.

#include "api.cuh"

#include "../../../include/CellShard/io/common/layout_policy.hh"
#include "../../bucket/blocked_ell_bipartite_optimize_host.cuh"
#include "../../convert/blocked_ell_from_compressed.cuh"
#include "../../sharded/disk.cuh"
#if CELLSHARD_ENABLE_CUDA
#include "../../sharded/sharded_device.cuh"
#endif
#include "../../sharded/sharded_host.cuh"

#include <hdf5.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cerrno>
#include <filesystem>
#include <limits>
#include <string>
#include <vector>
#include <condition_variable>
#include <deque>
#include <mutex>
#include <thread>

#include <fcntl.h>
#include <sys/stat.h>
#include <sys/statvfs.h>
#include <sys/types.h>
#include <unistd.h>

namespace cellshard {

namespace {
#include "internal_base_part.hh"
} // namespace
