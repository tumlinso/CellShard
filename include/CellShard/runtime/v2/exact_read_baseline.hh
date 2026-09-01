#pragma once

#include <CellShard/runtime/source/payload_source.hh>
#include <CellShard/runtime/v2/atom_source.hh>

namespace cellshard::runtime_v2 {

[[nodiscard]] status_code synchronous_read_exact(
    const atom_source_request &request,
    array_view<payload_source_ref> sources) noexcept;

} // namespace cellshard::runtime_v2
