#pragma once
#include <cstddef>
#include <cstdint>
namespace cellshard::artifact::atom_store {
enum class codec_status_v1 : std::uint32_t { success, invalid_input, insufficient_output, corrupt_input };
using codec_transform_fn_v1=codec_status_v1(*)(const std::byte*,std::size_t,std::byte*,std::size_t,std::size_t*) noexcept;
struct codec_provider_v1 { std::uint64_t codec_identity=0; codec_transform_fn_v1 encode=nullptr; codec_transform_fn_v1 decode=nullptr; };
class codec_registry_v1 { public: constexpr codec_registry_v1(codec_provider_v1*slots,std::size_t capacity)noexcept:slots_(slots),capacity_(capacity){} [[nodiscard]] bool register_provider(codec_provider_v1 provider)noexcept; [[nodiscard]] const codec_provider_v1*find(std::uint64_t identity)const noexcept; [[nodiscard]] std::size_t size()const noexcept{return size_;} private:codec_provider_v1*slots_=nullptr;std::size_t capacity_=0;std::size_t size_=0;};
inline constexpr std::uint64_t raw_codec_identity_v1=1;
inline constexpr std::uint64_t delta_u64_index_codec_identity_v1=2;
[[nodiscard]] codec_provider_v1 raw_codec_provider_v1() noexcept;
[[nodiscard]] codec_provider_v1 delta_u64_index_codec_provider_v1() noexcept;
}
