#pragma once

#include <cstddef>
#include <cstdio>
#include <cstdint>

namespace cellshard {

struct shard_locator {
    std::uint64_t offset;
    std::uint64_t bytes;
};

struct shard_storage;
typedef int (*shard_storage_open_fn)(shard_storage *s);
typedef void (*shard_storage_close_fn)(shard_storage *s);

enum {
    shard_storage_backend_none = 0,
    shard_storage_backend_packfile = 1,
    shard_storage_backend_series_h5 = 2
};

// shard_storage owns a heap copy of the packfile path, one locator per part,
// and one lazily-open packfile handle used by host fetch/stage paths.
//
// The open FILE* is a deliberate performance cache:
// - shard fetches stop paying fopen()/fclose() per part
// - sequential part loads inside one shard can preserve cursor locality
// - higher layers still own host/device payload residency explicitly
struct shard_storage {
    unsigned int capacity;
    unsigned int backend;
    char *packfile_path;
    shard_locator *locators;
    std::FILE *packfile_fp;
    void *backend_state;
    shard_storage_open_fn open_backend;
    shard_storage_close_fn close_backend;
};

void init(shard_storage *s);
void clear(shard_storage *s);
int reserve(shard_storage *s, unsigned int capacity);
int bind_packfile(shard_storage *s, const char *path);
int bind_part(shard_storage *s, unsigned int partId, std::uint64_t offset, std::uint64_t bytes);
int ensure_packfile_open(shard_storage *s);
void close_packfile(shard_storage *s);

} // namespace cellshard
