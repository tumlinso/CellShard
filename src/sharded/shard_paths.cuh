#pragma once

#include <cstddef>

namespace cellshard {

struct shard_storage {
    unsigned int capacity;
    char **paths;
};

void init(shard_storage *s);
void clear(shard_storage *s);
int reserve(shard_storage *s, unsigned int capacity);
int bind(shard_storage *s, unsigned int partId, const char *path);
int bind_sequential(shard_storage *s, unsigned int count, const char *prefix);

} // namespace cellshard
