#include <CellShard/artifact/atom_store/atom_compression_experiment_v1.hh>
#include <array>
#include <cassert>
using namespace cellshard::artifact::atom_store;
int main(){std::array<std::byte,64>a{};std::array<std::byte,64>b{};for(std::size_t i=0;i<b.size();++i)b[i]=static_cast<std::byte>(i);atom_block_view_v1 atoms[2]{{a.data(),a.size()},{b.data(),b.size()}};std::array<std::byte,128>scratch{};atom_compression_experiment_v1 result{};assert(run_atom_compression_experiment_v1(atoms,2,4,scratch.data(),scratch.size(),&result)==codec_status_v1::success);assert(result.raw_bytes==128&&result.atoms_using_rle==1&&result.atom_aware_bytes==74&&result.atom_aware_wins);atom_block_view_v1 one{b.data(),b.size()};assert(run_atom_compression_experiment_v1(&one,1,4,scratch.data(),scratch.size(),&result)==codec_status_v1::success);assert(!result.atom_aware_wins);}
