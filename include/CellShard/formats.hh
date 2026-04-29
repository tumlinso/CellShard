#pragma once

#ifndef CELLSHARD_ENABLE_CELLERATOR_QUANTIZED
#define CELLSHARD_ENABLE_CELLERATOR_QUANTIZED 1
#endif

#include "formats/dense.cuh"
#include "formats/compressed.cuh"
#include "formats/blocked_ell.cuh"
#if CELLSHARD_ENABLE_CELLERATOR_QUANTIZED
#include "formats/quantized_blocked_ell.cuh"
#endif
#include "formats/sliced_ell.cuh"
#include "formats/triplet.cuh"
#include "formats/diagonal.cuh"
