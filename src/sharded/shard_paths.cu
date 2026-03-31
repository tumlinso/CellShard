#include "shard_paths.cuh"

#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace cellshard {

void init(shard_storage *s) {
    s->capacity = 0;
    s->paths = 0;
}

void clear(shard_storage *s) {
    unsigned int i = 0;

    if (s->paths != 0) {
        for (i = 0; i < s->capacity; ++i) std::free(s->paths[i]);
    }
    std::free(s->paths);
    s->capacity = 0;
    s->paths = 0;
}

int reserve(shard_storage *s, unsigned int capacity) {
    char **paths = 0;
    unsigned int i = 0;

    if (capacity <= s->capacity) return 1;
    paths = (char **) std::calloc((std::size_t) capacity, sizeof(char *));
    if (paths == 0) return 0;
    for (i = 0; i < s->capacity; ++i) paths[i] = s->paths[i];
    std::free(s->paths);
    s->paths = paths;
    s->capacity = capacity;
    return 1;
}

int bind(shard_storage *s, unsigned int partId, const char *path) {
    std::size_t len = 0;
    char *copy = 0;

    if (partId >= s->capacity) return 0;
    std::free(s->paths[partId]);
    s->paths[partId] = 0;
    if (path == 0) return 1;

    len = std::strlen(path);
    copy = (char *) std::malloc(len + 1);
    if (copy == 0) return 0;
    std::memcpy(copy, path, len + 1);
    s->paths[partId] = copy;
    return 1;
}

int bind_sequential(shard_storage *s, unsigned int count, const char *prefix) {
    char path[1024];
    unsigned int i = 0;

    if (!reserve(s, count)) return 0;
    for (i = 0; i < count; ++i) {
        if (std::snprintf(path, sizeof(path), "%s.%u", prefix, (unsigned int) i) <= 0) return 0;
        if (!bind(s, i, path)) return 0;
    }
    return 1;
}

} // namespace cellshard
