#include <CellShard/artifact/atom_store/atom_frame_v1.hh>

#include <cassert>

using namespace cellshard::artifact::atom_store;

int main() {
    atom_frame_header_v1 frame{};
    frame.atom = {1, 2};
    frame.materialization = {3, 4};
    frame.content.bytes[0] = std::byte{1};
    frame.logical_bytes = 64;
    frame.encoded_bytes = 64;
    frame.payload_offset = 128;
    frame.payload_alignment = 64;
    assert(valid_atom_frame_header_v1(frame, 192));
    assert(!valid_atom_frame_header_v1(frame, 191));

    auto invalid = frame;
    invalid.payload_offset = 129;
    assert(!valid_atom_frame_header_v1(invalid, 256));
    invalid = frame;
    invalid.encoded_bytes = 32;
    assert(!valid_atom_frame_header_v1(invalid, 192));
    invalid = frame;
    invalid.content.bytes[0] = std::byte{0};
    assert(!valid_atom_frame_header_v1(invalid, 192));
}
