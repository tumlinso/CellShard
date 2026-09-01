#include <CellShard/artifact/atom_store/atom_compression_experiment_v1.hh>
#include <limits>
namespace cellshard::artifact::atom_store {
codec_status_v1 run_atom_compression_experiment_v1(const atom_block_view_v1*atoms,std::size_t count,std::size_t metadata,std::byte*scratch,std::size_t scratch_bytes,atom_compression_experiment_v1*out)noexcept{
    if(atoms==nullptr||count==0||scratch==nullptr||out==nullptr)return codec_status_v1::invalid_input;
    atom_compression_experiment_v1 result{};auto codec=byte_rle_codec_provider_v1();
    for(std::size_t i=0;i<count;++i){if(atoms[i].data==nullptr||atoms[i].bytes==0||result.raw_bytes>SIZE_MAX-atoms[i].bytes)return codec_status_v1::invalid_input;result.raw_bytes+=atoms[i].bytes;std::size_t encoded=0;auto status=codec.encode(atoms[i].data,atoms[i].bytes,scratch,scratch_bytes,&encoded);const auto selected=status==codec_status_v1::success&&encoded<atoms[i].bytes?encoded:atoms[i].bytes;if(selected==encoded&&encoded<atoms[i].bytes)++result.atoms_using_rle;if(selected>SIZE_MAX-metadata||result.atom_aware_bytes>SIZE_MAX-(selected+metadata))return codec_status_v1::invalid_input;result.atom_aware_bytes+=selected+metadata;}
    std::size_t whole_encoded=0;result.monolithic_bytes=result.raw_bytes;
    // A monolithic RLE measurement is valid only when callers provide one contiguous atom.
    if(count==1&&codec.encode(atoms[0].data,atoms[0].bytes,scratch,scratch_bytes,&whole_encoded)==codec_status_v1::success&&whole_encoded<result.monolithic_bytes)result.monolithic_bytes=whole_encoded;
    result.atom_aware_wins=result.atom_aware_bytes<result.monolithic_bytes;*out=result;return codec_status_v1::success;
}
}
