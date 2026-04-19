#pragma once

// Compatibility shim: this header used to expose shard_storage even though the
// name suggests cache-pack path builders. Prefer runtime/storage/shard_storage.cuh.
#include "../storage/shard_storage.cuh"
