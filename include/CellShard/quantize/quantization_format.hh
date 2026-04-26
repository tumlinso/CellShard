#pragma once

#include <stdint.h>

enum qbell_quantization_layout : uint32_t {
    qbell_affine_gene   = 0,
    qbell_affine_block  = 1,
    qbell_codebook_gene = 2,
    qbell_state_delta   = 3,
};

enum qbell_quantization_level : uint32_t {
    qbell_i8     = 0,
    qbell_u8     = 1,
    qbell_i4     = 2,
    qbell_u4     = 3,
    qbell_u2     = 4,

    // One-bit logical/binary encodings.
    qbell_bit01  = 5, // values decode to {0, 1}
    qbell_bitpm1 = 6, // values decode to {-1, +1}
};