#pragma once

#include <CellShard/identity.hh>

#include <cstddef>
#include <cstdint>
#include <type_traits>

namespace cellshard::compiler::graph {

enum operation_provider_capability : std::uint64_t {
    supports_cpu = UINT64_C(1) << 0,
    supports_cuda = UINT64_C(1) << 1,
    supports_partial_lowering = UINT64_C(1) << 2,
    supports_resume = UINT64_C(1) << 3,
};

struct operation_provider_descriptor {
    producer_abi_id provider{};
    operator_class_id operation{};
    content_digest source_content{};
    std::uint64_t source_revision = 0;
    std::uint64_t capability_flags = 0;
    std::uint32_t input_arity = 0;
    std::uint32_t output_arity = 0;
};

[[nodiscard]] constexpr bool valid_operation_provider_descriptor(
    const operation_provider_descriptor &descriptor) noexcept {
    constexpr auto known = supports_cpu | supports_cuda
        | supports_partial_lowering | supports_resume;
    return descriptor.provider.valid() && descriptor.operation.valid()
        && valid_content_digest(descriptor.source_content)
        && descriptor.source_content.algorithm != digest_algorithm::none
        && descriptor.source_revision != 0 && descriptor.input_arity != 0
        && descriptor.output_arity != 0 && descriptor.capability_flags != 0
        && (descriptor.capability_flags & ~known) == 0
        && ((descriptor.capability_flags & (supports_cpu | supports_cuda)) != 0);
}

static_assert(std::is_trivially_copyable<operation_provider_descriptor>::value);

} // namespace cellshard::compiler::graph
