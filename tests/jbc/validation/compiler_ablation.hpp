#pragma once
#include "metrics.hpp"
#include <array>
namespace cellshard::jbc::validation {
enum class compiler_layer : std::uint8_t { composition = 1, grammar = 2, basis = 4, superatom = 8 };
struct compiler_ablation { std::uint8_t enabled_layers = 0; };
inline constexpr std::array<compiler_ablation, 6> compiler_ablation_matrix{{
    {0}, {15}, {14}, {13}, {11}, {7}}};
inline bool layer_enabled(compiler_ablation item, compiler_layer layer) noexcept {
    return (item.enabled_layers & static_cast<std::uint8_t>(layer)) != 0;
}
struct compiler_ablation_result { metric_record metric{}; std::uint64_t persistent_bytes = 0; std::uint64_t build_ns = 0; };
inline bool comparable(const compiler_ablation_result& left,
                       const compiler_ablation_result& right) noexcept {
    return complete_metric(left.metric) && complete_metric(right.metric) &&
           left.metric.fixture_id == right.metric.fixture_id &&
           left.metric.matched_null == right.metric.matched_null;
}
}  // namespace cellshard::jbc::validation
