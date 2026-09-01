#include <CellShard/artifact/atom_store/codec_registry_v1.hh>
#include <array>
#include <cassert>
#include <cstring>
using namespace cellshard::artifact::atom_store;
int main(){std::array<codec_provider_v1,2>s{};codec_registry_v1 r(s.data(),s.size());assert(r.register_provider(raw_codec_provider_v1()));assert(r.register_provider(delta_u64_index_codec_provider_v1()));assert(!r.register_provider(raw_codec_provider_v1()));std::array<std::uint64_t,3>values{{2,5,9}},decoded{};std::array<std::byte,sizeof(values)>encoded{};std::size_t n=0;auto*p=r.find(delta_u64_index_codec_identity_v1);assert(p&&p->encode(reinterpret_cast<const std::byte*>(values.data()),sizeof(values),encoded.data(),encoded.size(),&n)==codec_status_v1::success);assert(p->decode(encoded.data(),n,reinterpret_cast<std::byte*>(decoded.data()),sizeof(decoded),&n)==codec_status_v1::success);assert(decoded==values);}
