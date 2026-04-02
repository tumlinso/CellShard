#pragma once

#include <cstddef>
#include <cstdint>

namespace cellshard {

struct shard_locator {
    std::uint64_t offset;
    std::uint64_t bytes;
};

// shard_storage owns a heap copy of the packfile path and one locator per part.
// It does not own any open file handles or host/device matrix payload.
struct shard_storage {
    unsigned int capacity;
    char *packfile_path;
    shard_locator *locators;
};

void init(shard_storage *s);
void clear(shard_storage *s);
int reserve(shard_storage *s, unsigned int capacity);
int bind_packfile(shard_storage *s, const char *path);
int bind_part(shard_storage *s, unsigned int partId, std::uint64_t offset, std::uint64_t bytes);

} // namespace cellshard
