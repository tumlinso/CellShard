#include <CellShard/artifact/atom_store/multi_range_source_v1.hh>
#include <array>
#include <cassert>
#include <cstring>
using namespace cellshard::artifact::atom_store;
struct source { std::array<std::byte,8> bytes{}; };
bool read(void *ctx,cellshard::storage_object_id,cellshard::extent_id,std::uint64_t off,std::byte *out,std::size_t n) { auto *s=static_cast<source*>(ctx); if(off>s->bytes.size() || n>s->bytes.size()-off)return false; std::memcpy(out,s->bytes.data()+off,n); return true; }
int main() {
    source src{}; for(std::size_t i=0;i<src.bytes.size();++i)src.bytes[i]=static_cast<std::byte>(i+1);
    atom_frame_map_record_v1 frame{{1,2},0,1,0,8,0,2};
    frame_extent_slice_v1 slices[2]{{{1,2},0,cellshard::storage_object_id{1},cellshard::extent_id{1},0,0,3},{{1,2},0,cellshard::storage_object_id{1},cellshard::extent_id{2},3,3,5}};
    auto expected=sha256_digest_v1(src.bytes.data(),src.bytes.size()); std::array<std::byte,8> output{};
    assert(read_exact_atom_frame_v1(frame,slices,2,expected,read,&src,output.data(),output.size())==range_read_status_v1::success);
    expected.bytes[0]^=std::byte{1}; assert(read_exact_atom_frame_v1(frame,slices,2,expected,read,&src,output.data(),output.size())==range_read_status_v1::digest_mismatch);
}
