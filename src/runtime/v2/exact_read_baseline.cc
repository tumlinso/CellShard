#include <CellShard/runtime/v2/exact_read_baseline.hh>

namespace cellshard::runtime_v2 {

status_code synchronous_read_exact(
    const atom_source_request &request,
    array_view<payload_source_ref> sources) noexcept {
    if (!valid_atom_source_request(request) || sources.empty()) {
        return status_code::invalid_input;
    }
    for (std::size_t range_index = 0; range_index < request.ranges.size;
         ++range_index) {
        const auto &range = request.ranges[range_index];
        const payload_source_ref *source = nullptr;
        for (std::size_t source_index = 0; source_index < sources.size;
             ++source_index) {
            if (sources[source_index].object == range.object) {
                if (source != nullptr) {
                    return status_code::invalid_input;
                }
                source = &sources[source_index];
            }
        }
        if (source == nullptr) {
            return status_code::missing_object;
        }
        const status_code status = read_exact(
            *source,
            exact_read_request{range.object, range.object_offset, range.bytes,
                               request.destination + range.destination_offset,
                               static_cast<std::size_t>(
                                   request.destination_bytes
                                   - range.destination_offset)});
        if (!status_ok(status)) {
            return status;
        }
    }
    return status_code::success;
}

} // namespace cellshard::runtime_v2
