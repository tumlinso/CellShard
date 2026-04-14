#pragma once

#include <cstdint>

namespace cellshard {

struct shard_storage;
typedef int (*shard_storage_open_fn)(shard_storage *s);
typedef void (*shard_storage_close_fn)(shard_storage *s);

enum {
    shard_storage_backend_none = 0,
    shard_storage_backend_dataset_h5 = 1
};

// shard_storage owns the durable source path plus backend-specific state for
// the active `.csh5` runtime.
struct shard_storage {
    unsigned int backend;
    char *source_path;
    void *backend_state;
    shard_storage_open_fn open_backend;
    shard_storage_close_fn close_backend;
};

// Metadata-only lifecycle; init() does not open any backend handle.
void init(shard_storage *s);
void clear(shard_storage *s);

} // namespace cellshard
