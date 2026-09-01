#include <CellShard/artifact/atom_store/publication_v1.hh>
#include <array>
#include <cassert>
using namespace cellshard::artifact::atom_store;
struct state { int step=0; bool conflict=false; };
bool stage(void*c,const content_digest_v1&,const std::byte*,std::size_t){return ++static_cast<state*>(c)->step==1;}
bool sync_object(void*c,const content_digest_v1&){return ++static_cast<state*>(c)->step==2;}
bool swap_root(void*c,const content_digest_v1&,const content_digest_v1&){auto*s=static_cast<state*>(c);++s->step;return !s->conflict&&s->step==3;}
bool sync_root(void*c){return ++static_cast<state*>(c)->step==4;}
root_generation_manifest_v1 manifest(std::uint64_t generation,std::byte root){root_generation_manifest_v1 m{};m.store_identity={1,2};m.generation=generation;m.structure_epoch=1;m.root_content.bytes[0]=root;if(generation>1)m.parent_root_content.bytes[0]=static_cast<std::byte>(static_cast<unsigned>(root)-1);return m;}
int main(){auto current=manifest(1,std::byte{1});auto next=manifest(2,std::byte{2});std::array<std::byte,1> image{};state s{};publication_backend_v1 b{&s,stage,sync_object,swap_root,sync_root};assert(publish_root_generation_v1(current,next,image.data(),image.size(),b)==publication_status_v1::success);assert(s.step==4);s={0,true};assert(publish_root_generation_v1(current,next,image.data(),image.size(),b)==publication_status_v1::root_conflict);assert(s.step==3);}
