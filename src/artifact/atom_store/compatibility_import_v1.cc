#include <CellShard/artifact/atom_store/compatibility_import_v1.hh>
#include <array>
namespace cellshard::artifact::atom_store {
compatibility_import_status_v1 inspect_compatibility_import_v1(const std::byte *source,std::size_t bytes,bool confirmed,semantic_identity_v1 atom,action_identity_v1 action,compatibility_import_v1 *out) noexcept {
    if(source==nullptr||bytes<8||out==nullptr||!atom.valid()||!action.valid())return compatibility_import_status_v1::invalid_input;
    static constexpr std::array<std::byte,8> hdf{{std::byte{0x89},std::byte{'H'},std::byte{'D'},std::byte{'F'},std::byte{0x0d},std::byte{0x0a},std::byte{0x1a},std::byte{0x0a}}};
    static constexpr std::array<std::byte,8> pack{{std::byte{'C'},std::byte{'S'},std::byte{'P'},std::byte{'A'},std::byte{'C'},std::byte{'K'},std::byte{'0'},std::byte{'1'}}};
    static constexpr std::array<std::byte,8> exec1{{std::byte{'C'},std::byte{'P'},std::byte{'E'},std::byte{'X'},std::byte{'E'},std::byte{'C'},std::byte{'0'},std::byte{'1'}}};
    static constexpr std::array<std::byte,8> exec2{{std::byte{'C'},std::byte{'P'},std::byte{'E'},std::byte{'X'},std::byte{'E'},std::byte{'C'},std::byte{'0'},std::byte{'2'}}};
    auto match=[&](const auto&m){for(std::size_t i=0;i<8;++i)if(source[i]!=m[i])return false;return true;};compatibility_family_v1 family{};
    if(match(hdf)){if(!confirmed)return compatibility_import_status_v1::csh5_confirmation_required;family=compatibility_family_v1::csh5;}else if(match(pack))family=compatibility_family_v1::cspack;else if(match(exec1))family=compatibility_family_v1::cpexec01;else if(match(exec2))family=compatibility_family_v1::cpexec02;else return compatibility_import_status_v1::unrecognized;
    *out={atom,action,family,0,sha256_digest_v1(source,bytes),bytes};return compatibility_import_status_v1::success;
}
}
