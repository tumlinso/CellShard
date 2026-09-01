#include <CellShard/runtime/v2/read_plan.hh>

#include <limits>

namespace cellshard::runtime_v2 {

status_code build_read_plan(
    array_view<atom_range> ranges, std::uint64_t maximum_gap_bytes,
    std::uint64_t maximum_span_bytes, read_span *spans,
    std::size_t span_capacity, read_copy *copies, std::size_t copy_capacity,
    read_plan *out) noexcept {
    if (ranges.empty() || maximum_span_bytes == 0 || spans == nullptr
        || copies == nullptr || out == nullptr || copy_capacity < ranges.size) {
        return status_code::invalid_input;
    }
    std::size_t span_count = 0;
    std::uint64_t staging_bytes = 0;
    std::uint64_t requested_bytes = 0;
    for (std::size_t i = 0; i < ranges.size; ++i) {
        const auto &range = ranges[i];
        if (!range.object.valid() || range.bytes == 0
            || range.object_offset > std::numeric_limits<std::uint64_t>::max()
                                         - range.bytes
            || requested_bytes > std::numeric_limits<std::uint64_t>::max()
                                     - range.bytes) {
            return status_code::invalid_input;
        }
        requested_bytes += range.bytes;
        bool coalesced = false;
        if (span_count != 0) {
            read_span &previous = spans[span_count - 1];
            const std::uint64_t previous_end =
                previous.object_offset + previous.bytes;
            if (range.object == previous.object
                && range.object_offset >= previous.object_offset) {
                const std::uint64_t range_end = range.object_offset + range.bytes;
                const std::uint64_t gap = range.object_offset > previous_end
                    ? range.object_offset - previous_end
                    : 0;
                const std::uint64_t combined = range_end > previous_end
                    ? range_end - previous.object_offset
                    : previous.bytes;
                if (gap <= maximum_gap_bytes && combined <= maximum_span_bytes) {
                    previous.bytes = combined;
                    coalesced = true;
                }
            }
        }
        if (!coalesced) {
            if (span_count == span_capacity || range.bytes > maximum_span_bytes
                || staging_bytes > std::numeric_limits<std::uint64_t>::max()
                                       - range.bytes) {
                return status_code::allocation_failure;
            }
            spans[span_count++] =
                read_span{range.object, range.object_offset, range.bytes,
                          staging_bytes};
        }
        const read_span &span = spans[span_count - 1];
        copies[i] = read_copy{static_cast<std::uint32_t>(span_count - 1),
                              range.object_offset - span.object_offset,
                              range.destination_offset, range.bytes};
        staging_bytes = span.staging_offset + span.bytes;
    }
    *out = read_plan{{spans, span_count}, {copies, ranges.size}, staging_bytes,
                     requested_bytes};
    return status_code::success;
}

} // namespace cellshard::runtime_v2
