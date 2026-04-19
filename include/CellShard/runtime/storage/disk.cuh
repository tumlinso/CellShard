#pragma once

#include <cstdint>
#include <cstdio>
#include <cstring>

#include "../../formats/compressed.cuh"
#include "../../formats/blocked_ell.cuh"
#include "../../formats/quantized_blocked_ell.cuh"
#include "../../formats/sliced_ell.cuh"
#include "../layout/sharded.cuh"
#include "../layout/shard_paths.cuh"
#include "../../io/csh5/api.cuh"

namespace cellshard {

inline int sharded_from_u64(std::uint64_t value, unsigned long *out, const char *label, const char *filename) {
    *out = (unsigned long) value;
    if ((std::uint64_t) *out != value) {
        std::fprintf(stderr, "Error: %s does not fit target sharded index type in %s\n", label, filename);
        return 0;
    }
    return 1;
}

template<typename MatrixT>
inline int load_header(const char *filename, sharded<MatrixT> *m) {
    std::fprintf(stderr,
                 "Error: sharded load_header is only implemented for .csh5 sparse dataset types: %s\n",
                 filename != 0 ? filename : "(null)");
    (void) m;
    return 0;
}

template<typename MatrixT>
inline int load_header(const char *filename, sharded<MatrixT> *m, shard_storage *s) {
    std::fprintf(stderr,
                 "Error: sharded load_header is only implemented for .csh5 sparse dataset types: %s\n",
                 filename != 0 ? filename : "(null)");
    (void) m;
    (void) s;
    return 0;
}

inline int load_header(const char *filename, sharded<sparse::compressed> *m, shard_storage *s) {
    std::fprintf(stderr,
                 "Error: legacy compressed .csh5 dataset loading is no longer supported: %s\n",
                 filename != 0 ? filename : "(null)");
    (void) m;
    (void) s;
    return 0;
}

inline int load_header(const char *filename, sharded<sparse::compressed> *m) {
    return load_header(filename, m, 0);
}

inline int load_header(const char *filename, sharded<sparse::blocked_ell> *m, shard_storage *s) {
    const char *ext = std::strrchr(filename != 0 ? filename : "", '.');
    if (ext != 0) {
        if (std::strcmp(ext, ".csh5") == 0 || std::strcmp(ext, ".h5") == 0 || std::strcmp(ext, ".hdf5") == 0) {
            return load_dataset_blocked_ell_h5_header(filename, m, s);
        }
    }
    std::fprintf(stderr,
                 "Error: blocked-ELL sharded load_header requires a .csh5/.h5/.hdf5 dataset file: %s\n",
                 filename != 0 ? filename : "(null)");
    return 0;
}

inline int load_header(const char *filename, sharded<sparse::blocked_ell> *m) {
    return load_header(filename, m, 0);
}

inline int load_header(const char *filename, sharded<sparse::quantized_blocked_ell> *m, shard_storage *s) {
    const char *ext = std::strrchr(filename != 0 ? filename : "", '.');
    if (ext != 0) {
        if (std::strcmp(ext, ".csh5") == 0 || std::strcmp(ext, ".h5") == 0 || std::strcmp(ext, ".hdf5") == 0) {
            return load_dataset_quantized_blocked_ell_h5_header(filename, m, s);
        }
    }
    std::fprintf(stderr,
                 "Error: quantized blocked-ELL sharded load_header requires a .csh5/.h5/.hdf5 dataset file: %s\n",
                 filename != 0 ? filename : "(null)");
    return 0;
}

inline int load_header(const char *filename, sharded<sparse::quantized_blocked_ell> *m) {
    return load_header(filename, m, 0);
}

inline int load_header(const char *filename, sharded<sparse::sliced_ell> *m, shard_storage *s) {
    const char *ext = std::strrchr(filename != 0 ? filename : "", '.');
    if (ext != 0) {
        if (std::strcmp(ext, ".csh5") == 0 || std::strcmp(ext, ".h5") == 0 || std::strcmp(ext, ".hdf5") == 0) {
            return load_dataset_sliced_ell_h5_header(filename, m, s);
        }
    }
    std::fprintf(stderr,
                 "Error: sliced-ELL sharded load_header requires a .csh5/.h5/.hdf5 dataset file: %s\n",
                 filename != 0 ? filename : "(null)");
    return 0;
}

inline int load_header(const char *filename, sharded<sparse::sliced_ell> *m) {
    return load_header(filename, m, 0);
}

template<typename MatrixT>
inline int store(const char *filename, const sharded<MatrixT> *m, shard_storage *s) {
    std::fprintf(stderr,
                 "Error: sharded store is no longer implemented here; use the .csh5 dataset writers instead: %s\n",
                 filename != 0 ? filename : "(null)");
    (void) m;
    (void) s;
    return 0;
}

} // namespace cellshard
