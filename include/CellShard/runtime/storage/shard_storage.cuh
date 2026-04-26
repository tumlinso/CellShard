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

enum {
    shard_storage_role_unknown = 0,
    shard_storage_role_builder = 1,
    shard_storage_role_owner_runtime = 2,
    shard_storage_role_executor = 3
};

enum {
    shard_storage_cap_none = 0u,
    shard_storage_cap_canonical_read = 1u << 0u,
    shard_storage_cap_canonical_write = 1u << 1u,
    shard_storage_cap_materialize_pack = 1u << 2u,
    shard_storage_cap_read_published_pack = 1u << 3u
};

// shard_storage owns the durable source path plus backend-specific state for
// the active `.csh5` runtime. It does not define the on-disk cache-pack path
// builders; those live under the `.csh5` runtime helpers.
struct shard_storage {
    unsigned int backend;
    std::uint32_t role;
    std::uint32_t capability_flags;
    char *source_path;
    void *backend_state;
    shard_storage_open_fn open_backend;
    shard_storage_close_fn close_backend;
};

// Metadata-only lifecycle; init() does not open any backend handle.
void init(shard_storage *s);
void clear(shard_storage *s);
int set_shard_storage_role(shard_storage *s, std::uint32_t role);
int shard_storage_has_capability(const shard_storage *s, std::uint32_t capability);
const char *shard_storage_role_name(std::uint32_t role);

} // namespace cellshard
