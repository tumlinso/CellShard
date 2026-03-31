#pragma once

namespace cellshard {

template<typename OffsetT>
__host__ __device__ __forceinline__ OffsetT find_offset_span(OffsetT row, const OffsetT *offsets, OffsetT count) {
    OffsetT lo = 0;
    OffsetT hi = count;
    while (lo < hi) {
        const OffsetT mid = lo + ((hi - lo) >> 1);
        const OffsetT mid_offset = offsets[mid];
        const OffsetT next_offset = offsets[mid + 1];
        if (row < mid_offset) hi = mid;
        else if (row >= next_offset) lo = mid + 1;
        else return mid;
    }
    return count;
}

} // namespace cellshard
