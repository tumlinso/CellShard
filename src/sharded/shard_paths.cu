#include "shard_paths.cuh"

#include <cstdlib>

namespace cellshard {

void init(shard_storage *s) {
    s->backend = shard_storage_backend_none;
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

} // namespace cellshard
