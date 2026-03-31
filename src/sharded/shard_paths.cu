#include "shard_paths.cuh"

#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace cellshard {

void init(shard_storage *s) {
    s->capacity = 0;
    s->packfile_path = 0;
    s->locators = 0;
}

void clear(shard_storage *s) {
    std::free(s->packfile_path);
    std::free(s->locators);
    s->capacity = 0;
    s->packfile_path = 0;
    s->locators = 0;
}

int reserve(shard_storage *s, unsigned int capacity) {
    shard_locator *locators = 0;
    unsigned int i = 0;

    if (capacity <= s->capacity) return 1;
    locators = (shard_locator *) std::calloc((std::size_t) capacity, sizeof(shard_locator));
    if (locators == 0) return 0;
    for (i = 0; i < s->capacity; ++i) locators[i] = s->locators[i];
    std::free(s->locators);
    s->locators = locators;
    s->capacity = capacity;
    return 1;
}

int bind_packfile(shard_storage *s, const char *path) {
    std::size_t len = 0;
    char *copy = 0;

    std::free(s->packfile_path);
    s->packfile_path = 0;
    if (path == 0) return 1;

    len = std::strlen(path);
    copy = (char *) std::malloc(len + 1);
    if (copy == 0) return 0;
    std::memcpy(copy, path, len + 1);
    s->packfile_path = copy;
    return 1;
}

int bind_part(shard_storage *s, unsigned int partId, std::uint64_t offset, std::uint64_t bytes) {
    if (partId >= s->capacity || s->locators == 0) return 0;
    s->locators[partId].offset = offset;
    s->locators[partId].bytes = bytes;
    return 1;
}

} // namespace cellshard
