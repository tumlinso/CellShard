#pragma once

#include "../../io/binary/matrix_io.cuh"

namespace cellshard {
namespace ingest {
namespace mtx {

static inline int store_sharded_coo(const char *header_path,
                                    const char *part_prefix,
                                    const sharded<sparse::coo> *view) {
    shard_storage files;
    int ok = 0;

    init(&files);
    if (!bind_sequential(&files, (unsigned int) view->num_parts, part_prefix)) goto done;
    ok = store(header_path, view, &files);

done:
    clear(&files);
    return ok;
}

} // namespace mtx
} // namespace ingest
} // namespace cellshard
