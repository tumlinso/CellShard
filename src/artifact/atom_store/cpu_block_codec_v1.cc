#include <CellShard/artifact/atom_store/cpu_block_codec_v1.hh>
namespace cellshard::artifact::atom_store {
namespace {
codec_status_v1 encode(const std::byte*in,std::size_t n,std::byte*out,std::size_t cap,std::size_t*w)noexcept{if((n!=0&&in==nullptr)||out==nullptr||w==nullptr)return codec_status_v1::invalid_input;std::size_t o=0;for(std::size_t i=0;i<n;){std::size_t run=1;while(i+run<n&&in[i+run]==in[i]&&run<255)++run;if(o>cap||cap-o<2)return codec_status_v1::insufficient_output;out[o++]=static_cast<std::byte>(run);out[o++]=in[i];i+=run;}*w=o;return codec_status_v1::success;}
codec_status_v1 decode(const std::byte*in,std::size_t n,std::byte*out,std::size_t cap,std::size_t*w)noexcept{if(in==nullptr||out==nullptr||w==nullptr||n%2!=0)return codec_status_v1::invalid_input;std::size_t o=0;for(std::size_t i=0;i<n;i+=2){auto run=static_cast<std::size_t>(in[i]);if(run==0)return codec_status_v1::corrupt_input;if(o>cap||run>cap-o)return codec_status_v1::insufficient_output;for(std::size_t j=0;j<run;++j)out[o++]=in[i+1];}*w=o;return codec_status_v1::success;}
}
codec_provider_v1 byte_rle_codec_provider_v1()noexcept{return{byte_rle_codec_identity_v1,encode,decode};}
codec_status_v1 select_cpu_block_codec_v1(const std::byte*input,std::size_t n,std::byte*buffer,std::size_t cap,cpu_block_choice_v1*choice)noexcept{if(choice==nullptr)return codec_status_v1::invalid_input;std::size_t encoded=0;const auto status=encode(input,n,buffer,cap,&encoded);if(status==codec_status_v1::success&&encoded<n)*choice={byte_rle_codec_identity_v1,encoded};else *choice={raw_codec_identity_v1,n};return status==codec_status_v1::invalid_input?status:codec_status_v1::success;}
}
