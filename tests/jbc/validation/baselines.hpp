#pragma once
#include "fixtures.hpp"
namespace cellshard::jbc::validation {
enum class baseline_kind : std::uint8_t { identity_csr, frequency_only };
inline void baseline_order(const std::uint64_t* frequency, std::uint32_t count,
                           baseline_kind kind, std::uint32_t* output) noexcept {
    for (std::uint32_t i = 0; i < count; ++i) output[i] = i;
    if (kind == baseline_kind::identity_csr || frequency == nullptr) return;
    for (std::uint32_t i = 1; i < count; ++i) {
        const std::uint32_t value = output[i]; std::uint32_t j = i;
        while (j != 0 && (frequency[value] > frequency[output[j - 1]] ||
               (frequency[value] == frequency[output[j - 1]] && value < output[j - 1]))) {
            output[j] = output[j - 1]; --j;
        }
        output[j] = value;
    }
}
}  // namespace cellshard::jbc::validation
