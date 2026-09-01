#include <CellShard/artifact/atom_store/consolidation_v1.hh>
#include <limits>
namespace cellshard::artifact::atom_store {
consolidation_status_v1 plan_consolidation_v1(const consolidation_source_v1 *sources,std::size_t source_count,consolidation_copy_v1 *copies,std::size_t capacity,std::size_t *count,std::uint64_t *output_bytes) noexcept {
    if(sources==nullptr||copies==nullptr||count==nullptr||output_bytes==nullptr)return consolidation_status_v1::invalid_input;
    std::size_t needed=0;for(std::size_t i=0;i<source_count;++i)if(sources[i].live==1)++needed;else if(sources[i].live!=0)return consolidation_status_v1::invalid_input;
    if(needed>capacity)return consolidation_status_v1::insufficient_output;
    std::uint64_t cursor=0;std::size_t out=0;
    for(std::size_t i=0;i<source_count;++i){const auto&s=sources[i];if(s.live==0)continue;bool nonzero=false;for(auto b:s.content.bytes)nonzero=nonzero||b!=std::byte{0};
        if(!valid_content_digest_v1(s.content)||!nonzero||s.bytes==0||s.alignment==0||(s.alignment&(s.alignment-1))!=0)return consolidation_status_v1::invalid_input;
        if(cursor>UINT64_MAX-(s.alignment-1))return consolidation_status_v1::overflow;
        const auto target=(cursor+s.alignment-1)&~(s.alignment-1);
        if(s.bytes>UINT64_MAX-target)return consolidation_status_v1::overflow;
        copies[out++]={s.content,s.source_offset,target,s.bytes};cursor=target+s.bytes;
    }
    *count=out;*output_bytes=cursor;return consolidation_status_v1::success;
}
}
