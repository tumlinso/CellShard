#include <CellShard/artifact/atom_store/multi_range_source_v1.hh>
namespace cellshard::artifact::atom_store {
range_read_status_v1 read_exact_atom_frame_v1(const atom_frame_map_record_v1 &frame, const frame_extent_slice_v1 *slices, std::size_t slice_count, const content_digest_v1 &expected, exact_range_read_fn_v1 read, void *context, std::byte *output, std::size_t output_bytes) noexcept {
    if (!frame_extent_slices_cover_v1(frame,slices,slice_count) || !valid_content_digest_v1(expected) || read==nullptr || output==nullptr) return range_read_status_v1::invalid_mapping;
    if (frame.logical_bytes>output_bytes) return range_read_status_v1::insufficient_output;
    for (std::size_t i=0;i<slice_count;++i) {
        const auto &s=slices[i];
        if (!read(context,s.object,s.extent,s.extent_offset,output+s.frame_offset,static_cast<std::size_t>(s.bytes))) return range_read_status_v1::short_read;
    }
    const auto actual=sha256_digest_v1(output,static_cast<std::size_t>(frame.logical_bytes));
    for (std::size_t i=0;i<actual.bytes.size();++i) if (actual.bytes[i]!=expected.bytes[i]) return range_read_status_v1::digest_mismatch;
    return range_read_status_v1::success;
}
}
