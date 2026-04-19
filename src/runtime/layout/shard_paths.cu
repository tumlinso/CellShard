#include "shard_paths.cuh"

#include <cstdlib>

namespace cellshard {

namespace {

inline std::uint32_t default_capabilities_for_role(std::uint32_t role) {
    switch (role) {
        case shard_storage_role_builder:
            return shard_storage_cap_canonical_read
                | shard_storage_cap_canonical_write
                | shard_storage_cap_materialize_canonical_pack
                | shard_storage_cap_materialize_execution_pack
                | shard_storage_cap_read_published_pack;
        case shard_storage_role_owner_runtime:
            return shard_storage_cap_canonical_read
                | shard_storage_cap_materialize_canonical_pack
                | shard_storage_cap_materialize_execution_pack
                | shard_storage_cap_read_published_pack;
        case shard_storage_role_executor:
            return shard_storage_cap_read_published_pack;
        default:
            return shard_storage_cap_none;
    }
}

} // namespace

void init(shard_storage *s) {
    s->backend = shard_storage_backend_none;
    s->role = shard_storage_role_unknown;
    s->capability_flags = shard_storage_cap_none;
    s->source_path = 0;
    s->backend_state = 0;
    s->open_backend = 0;
    s->close_backend = 0;
}

void clear(shard_storage *s) {
    if (s->close_backend != 0) s->close_backend(s);
    std::free(s->source_path);
    init(s);
}

int set_shard_storage_role(shard_storage *s, std::uint32_t role) {
    if (s == nullptr) return 0;
    if (role != shard_storage_role_unknown
        && role != shard_storage_role_builder
        && role != shard_storage_role_owner_runtime
        && role != shard_storage_role_executor) {
        return 0;
    }
    s->role = role;
    s->capability_flags = default_capabilities_for_role(role);
    return 1;
}

int shard_storage_has_capability(const shard_storage *s, std::uint32_t capability) {
    if (s == nullptr) return 0;
    return (s->capability_flags & capability) == capability ? 1 : 0;
}

const char *shard_storage_role_name(std::uint32_t role) {
    switch (role) {
        case shard_storage_role_builder:
            return "builder";
        case shard_storage_role_owner_runtime:
            return "owner_runtime";
        case shard_storage_role_executor:
            return "executor";
        default:
            return "unknown";
    }
}

} // namespace cellshard
