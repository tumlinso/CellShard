#include <CellShard/artifact/atom_store/cpu_block_codec_v1.hh>
#include <array>
#include <cassert>
using namespace cellshard::artifact::atom_store;
int main(){std::array<std::byte,64>input{};std::array<std::byte,128>encoded{};cpu_block_choice_v1 choice{};assert(select_cpu_block_codec_v1(input.data(),input.size(),encoded.data(),encoded.size(),&choice)==codec_status_v1::success);assert(choice.codec_identity==byte_rle_codec_identity_v1&&choice.encoded_bytes==2);std::array<std::byte,64>decoded{};std::size_t n=0;auto codec=byte_rle_codec_provider_v1();assert(codec.decode(encoded.data(),choice.encoded_bytes,decoded.data(),decoded.size(),&n)==codec_status_v1::success&&n==input.size());for(std::size_t i=0;i<input.size();++i)input[i]=static_cast<std::byte>(i);assert(select_cpu_block_codec_v1(input.data(),input.size(),encoded.data(),encoded.size(),&choice)==codec_status_v1::success);assert(choice.codec_identity==raw_codec_identity_v1);}
