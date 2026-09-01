#include <CellShard/artifact/atom_store/frame_extent_map_v1.hh>

#include <array>
#include <cassert>

using namespace cellshard::artifact::atom_store;

int main() {
    atom_frame_map_record_v1 frame{{1, 2}, 1, 3, 128, 96, 4, 2};
    std::array<frame_extent_slice_v1, 2> slices{{
        {{1, 2}, 1, cellshard::storage_object_id{7}, cellshard::extent_id{8}, 10, 0, 32},
        {{1, 2}, 1, cellshard::storage_object_id{7}, cellshard::extent_id{9}, 0, 32, 64},
    }};
    assert(valid_atom_frame_map_record_v1(frame));
    assert(frame_extent_slices_cover_v1(frame, slices.data(), slices.size()));

    auto bad = slices;
    bad[1].frame_offset = 31;
    assert(!frame_extent_slices_cover_v1(frame, bad.data(), bad.size()));
    bad = slices;
    bad[1].bytes = 63;
    assert(!frame_extent_slices_cover_v1(frame, bad.data(), bad.size()));
    bad = slices;
    bad[1].atom = {2, 1};
    assert(!frame_extent_slices_cover_v1(frame, bad.data(), bad.size()));
}
