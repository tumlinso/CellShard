#include <CellShard/runtime/v2/atom_source.hh>

#include <array>
#include <cassert>

using namespace cellshard;
using namespace cellshard::runtime_v2;

int main() {
    std::array<std::byte, 32> destination{};
    std::array ranges{
        atom_range{storage_object_id{1}, 64, 8, 0},
        atom_range{storage_object_id{2}, 96, 16, 8},
    };
    atom_source_request request{{ranges.data(), ranges.size()},
                                destination.data(), destination.size()};
    assert(valid_atom_source_request(request));
    ranges[1].destination_offset = 4;
    assert(!valid_atom_source_request(request));
    ranges[1].destination_offset = 24;
    assert(!valid_atom_source_request(request));
    ranges[1].destination_offset = 8;
    ranges[1].object = {};
    assert(!valid_atom_source_request(request));
}
