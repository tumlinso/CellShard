#include <CellShard/artifact/atom_store/codec_registry_v1.hh>
#include <cstring>
namespace cellshard::artifact::atom_store {
bool codec_registry_v1::register_provider(codec_provider_v1 p)noexcept{if(slots_==nullptr||p.codec_identity==0||p.encode==nullptr||p.decode==nullptr||size_==capacity_||find(p.codec_identity)!=nullptr)return false;slots_[size_++]=p;return true;}
const codec_provider_v1*codec_registry_v1::find(std::uint64_t id)const noexcept{for(std::size_t i=0;i<size_;++i)if(slots_[i].codec_identity==id)return &slots_[i];return nullptr;}
namespace {
codec_status_v1 raw(const std::byte*in,std::size_t n,std::byte*out,std::size_t cap,std::size_t*written)noexcept{if((n!=0&&in==nullptr)||out==nullptr||written==nullptr)return codec_status_v1::invalid_input;if(cap<n)return codec_status_v1::insufficient_output;std::memmove(out,in,n);*written=n;return codec_status_v1::success;}
std::uint64_t load(const std::byte*p){std::uint64_t v=0;for(unsigned i=0;i<8;++i)v|=static_cast<std::uint64_t>(p[i])<<(i*8);return v;}void store(std::byte*p,std::uint64_t v){for(unsigned i=0;i<8;++i)p[i]=static_cast<std::byte>((v>>(i*8))&0xffu);}
codec_status_v1 encode_index(const std::byte*in,std::size_t n,std::byte*out,std::size_t cap,std::size_t*w)noexcept{if(in==nullptr||out==nullptr||w==nullptr||n%8!=0)return codec_status_v1::invalid_input;if(cap<n)return codec_status_v1::insufficient_output;std::uint64_t prev=0;for(std::size_t o=0;o<n;o+=8){auto value=load(in+o);if(o!=0&&value<prev)return codec_status_v1::invalid_input;store(out+o,o==0?value:value-prev);prev=value;}*w=n;return codec_status_v1::success;}
codec_status_v1 decode_index(const std::byte*in,std::size_t n,std::byte*out,std::size_t cap,std::size_t*w)noexcept{if(in==nullptr||out==nullptr||w==nullptr||n%8!=0)return codec_status_v1::invalid_input;if(cap<n)return codec_status_v1::insufficient_output;std::uint64_t prev=0;for(std::size_t o=0;o<n;o+=8){auto delta=load(in+o);if(o!=0&&delta>UINT64_MAX-prev)return codec_status_v1::corrupt_input;auto value=o==0?delta:prev+delta;store(out+o,value);prev=value;}*w=n;return codec_status_v1::success;}
}
codec_provider_v1 raw_codec_provider_v1()noexcept{return{raw_codec_identity_v1,raw,raw};}
codec_provider_v1 delta_u64_index_codec_provider_v1()noexcept{return{delta_u64_index_codec_identity_v1,encode_index,decode_index};}
}
