#pragma once

#include <CellShard/compiler/evidence/atom_evidence_record_v1.hh>

#include <cstddef>
#include <cstdint>
#include <type_traits>
#include <vector>

namespace cellshard::compiler::evidence {

struct evidence_atlas_source_v1 {
    const atom_evidence_record_v1 *records = nullptr;
    std::uint64_t record_count = 0;
    evidence_identity_v1 atlas_identity{};
    std::uint64_t atlas_generation = 0;
};

struct evidence_atlas_view_v1 {
    const atom_evidence_record_v1 *records = nullptr;
    std::uint64_t record_count = 0;
    evidence_identity_v1 atlas_identity{};
    std::uint64_t atlas_generation = 0;
};

struct evidence_atlas_requirements_v1 {
    std::uint64_t record_capacity = 0;
    std::uint64_t record_bytes = 0;
};

enum class evidence_atlas_build_code_v1 : std::uint32_t {
    ready = 0,
    built,
    invalid_atlas_identity,
    missing_atlas_generation,
    empty_source,
    missing_records,
    record_limit_exceeded,
    byte_overflow,
    invalid_record,
    unordered_or_duplicate_record,
    insufficient_capacity,
    allocation_failure,
};

struct evidence_atlas_build_result_v1 {
    evidence_atlas_build_code_v1 code = evidence_atlas_build_code_v1::ready;
    std::uint64_t index = 0;
    evidence_atlas_requirements_v1 requirements{};
    [[nodiscard]] constexpr bool ok() const noexcept {
        return code == evidence_atlas_build_code_v1::ready
            || code == evidence_atlas_build_code_v1::built;
    }
};

[[nodiscard]] evidence_atlas_build_result_v1 evidence_atlas_requirements(
    evidence_atlas_source_v1 source,
    std::uint64_t maximum_records) noexcept;

class evidence_atlas_builder_v1 {
public:
    [[nodiscard]] evidence_atlas_build_result_v1 fill(
        evidence_atlas_source_v1 source,
        evidence_atlas_requirements_v1 capacity,
        std::uint64_t maximum_records) noexcept;

    void reset() noexcept;
    [[nodiscard]] const evidence_atlas_view_v1 &view() const noexcept { return view_; }

private:
    void rebind() noexcept;
    evidence_atlas_view_v1 view_{};
    std::vector<atom_evidence_record_v1> records_{};
};

static_assert(offsetof(evidence_atlas_source_v1, records) == 0);
static_assert(offsetof(evidence_atlas_view_v1, records) == 0);
static_assert(std::is_standard_layout<evidence_atlas_view_v1>::value);

} // namespace cellshard::compiler::evidence
