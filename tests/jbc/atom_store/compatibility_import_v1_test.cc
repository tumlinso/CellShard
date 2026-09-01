#include <CellShard/artifact/atom_store/compatibility_import_v1.hh>
#include <array>
#include <cassert>
using namespace cellshard::artifact::atom_store;
int main(){std::array<std::byte,8>p{{std::byte{'C'},std::byte{'S'},std::byte{'P'},std::byte{'A'},std::byte{'C'},std::byte{'K'},std::byte{'0'},std::byte{'1'}}};compatibility_import_v1 out{};assert(inspect_compatibility_import_v1(p.data(),p.size(),false,{1,2},{3,4},&out)==compatibility_import_status_v1::success);assert(out.family==compatibility_family_v1::cspack&&out.source_bytes==8);p[0]=std::byte{0x89};p[1]=std::byte{'H'};p[2]=std::byte{'D'};p[3]=std::byte{'F'};p[4]=std::byte{0x0d};p[5]=std::byte{0x0a};p[6]=std::byte{0x1a};p[7]=std::byte{0x0a};assert(inspect_compatibility_import_v1(p.data(),p.size(),false,{1,2},{3,4},&out)==compatibility_import_status_v1::csh5_confirmation_required);assert(inspect_compatibility_import_v1(p.data(),p.size(),true,{1,2},{3,4},&out)==compatibility_import_status_v1::success);}
