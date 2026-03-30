#pragma once

#include "../../matrix.cuh"

namespace cellshard {
namespace sparse {

struct csc {
    unsigned int rows;
    unsigned int cols;
    unsigned int nnz;
    unsigned char format;

    unsigned int *colPtr;
    unsigned int *rowIdx;
    __half *val;
};

inline void init(csc *m, unsigned int rows = 0, unsigned int cols = 0, unsigned int nnz = 0) {
    m->rows = rows;
    m->cols = cols;
    m->nnz = nnz;
    m->format = format_csc;
    m->colPtr = 0;
    m->rowIdx = 0;
    m->val = 0;
}

inline std::size_t bytes(const csc *m) {
    return sizeof(*m)
        + (std::size_t) (m->cols + 1) * sizeof(unsigned int)
        + (std::size_t) m->nnz * sizeof(unsigned int)
        + (std::size_t) m->nnz * sizeof(__half);
}

inline void clear(csc *m) {
    std::free(m->colPtr);
    std::free(m->rowIdx);
    std::free(m->val);
    m->colPtr = 0;
    m->rowIdx = 0;
    m->val = 0;
    m->rows = 0;
    m->cols = 0;
    m->nnz = 0;
    m->format = format_csc;
}

inline int allocate(csc *m) {
    std::free(m->colPtr);
    std::free(m->rowIdx);
    std::free(m->val);
    m->colPtr = 0;
    m->rowIdx = 0;
    m->val = 0;
    if (m->cols != 0) m->colPtr = (unsigned int *) std::malloc((std::size_t) (m->cols + 1) * sizeof(unsigned int));
    if (m->nnz != 0) {
        m->rowIdx = (unsigned int *) std::malloc((std::size_t) m->nnz * sizeof(unsigned int));
        m->val = (__half *) std::malloc((std::size_t) m->nnz * sizeof(__half));
    }
    if (m->cols != 0 && m->colPtr == 0) return 0;
    if (m->nnz != 0 && (m->rowIdx == 0 || m->val == 0)) {
        std::free(m->colPtr);
        std::free(m->rowIdx);
        std::free(m->val);
        m->colPtr = 0;
        m->rowIdx = 0;
        m->val = 0;
        return 0;
    }
    return 1;
}

inline const __half *at(const csc *m, unsigned int r, unsigned int c) {
    for (unsigned int i = m->colPtr[c]; i < m->colPtr[c + 1]; ++i) {
        if (m->rowIdx[i] == r) return m->val + i;
    }
    return 0;
}

inline __half *at(csc *m, unsigned int r, unsigned int c) {
    for (unsigned int i = m->colPtr[c]; i < m->colPtr[c + 1]; ++i) {
        if (m->rowIdx[i] == r) return m->val + i;
    }
    return 0;
}

} // namespace sparse
} // namespace cellshard
