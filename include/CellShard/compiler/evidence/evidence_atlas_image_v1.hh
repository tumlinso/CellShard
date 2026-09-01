#pragma once

#include <CellShard/compiler/evidence/evidence_atlas_v1.hh>

#include <cstdint>

namespace cellshard::compiler::evidence {

inline constexpr std::uint64_t evidence_atlas_image_header_bytes_v1 = 64;
inline constexpr std::uint64_t evidence_atlas_image_record_bytes_v1 = 80;

enum class evidence_atlas_image_code_v1 : std::uint32_t {
    success = 0,
    invalid_atlas,
    size_overflow,
    missing_buffer,
    insufficient_buffer,
    invalid_magic,
    unsupported_schema,
    invalid_header_size,
    invalid_total_size,
    record_limit_exceeded,
    checksum_mismatch,
    invalid_record,
    build_failure,
    allocation_failure,
};

struct evidence_atlas_image_result_v1 {
    evidence_atlas_image_code_v1 code = evidence_atlas_image_code_v1::success;
    std::uint64_t required_bytes = 0;
    std::uint64_t index = 0;
    [[nodiscard]] constexpr bool ok() const noexcept {
        return code == evidence_atlas_image_code_v1::success;
    }
};

[[nodiscard]] evidence_atlas_image_result_v1 evidence_atlas_image_requirements_v1(
    evidence_atlas_view_v1 atlas,
    std::uint64_t maximum_records) noexcept;

[[nodiscard]] evidence_atlas_image_result_v1 encode_evidence_atlas_v1(
    evidence_atlas_view_v1 atlas,
    void *destination,
    std::uint64_t destination_bytes,
    std::uint64_t maximum_records) noexcept;

[[nodiscard]] evidence_atlas_image_result_v1 decode_evidence_atlas_v1(
    const void *source,
    std::uint64_t source_bytes,
    std::uint64_t maximum_records,
    evidence_atlas_builder_v1 *output) noexcept;

} // namespace cellshard::compiler::evidence
