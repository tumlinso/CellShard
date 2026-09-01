#include <CellShard/compiler/evidence/evidence_atlas_image_v1.hh>

#include <cstring>
#include <limits>
#include <new>
#include <stdexcept>
#include <vector>

namespace cellshard::compiler::evidence {
namespace {

constexpr unsigned char magic[8] = {'C', 'S', 'E', 'V', 'A', 'T', '0', '1'};

void put_u32(unsigned char *out, std::uint32_t value) noexcept {
    for (unsigned shift = 0; shift < 32; shift += 8)
        *out++ = static_cast<unsigned char>(value >> shift);
}
void put_u64(unsigned char *out, std::uint64_t value) noexcept {
    for (unsigned shift = 0; shift < 64; shift += 8)
        *out++ = static_cast<unsigned char>(value >> shift);
}
std::uint32_t get_u32(const unsigned char *in) noexcept {
    std::uint32_t value = 0;
    for (unsigned shift = 0; shift < 32; shift += 8)
        value |= static_cast<std::uint32_t>(*in++) << shift;
    return value;
}
std::uint64_t get_u64(const unsigned char *in) noexcept {
    std::uint64_t value = 0;
    for (unsigned shift = 0; shift < 64; shift += 8)
        value |= static_cast<std::uint64_t>(*in++) << shift;
    return value;
}
std::uint64_t checksum(const unsigned char *data, std::uint64_t size) noexcept {
    std::uint64_t value = 14695981039346656037ULL;
    for (std::uint64_t index = 0; index < size; ++index) {
        value ^= data[index];
        value *= 1099511628211ULL;
    }
    return value;
}
void put_identity(unsigned char *out, evidence_identity_v1 identity) noexcept {
    put_u64(out, identity.producer_namespace);
    put_u64(out + 8, identity.local_identity);
}
evidence_identity_v1 get_identity(const unsigned char *in) noexcept {
    return {get_u64(in), get_u64(in + 8)};
}

} // namespace

evidence_atlas_image_result_v1 evidence_atlas_image_requirements_v1(
    evidence_atlas_view_v1 atlas, std::uint64_t maximum_records) noexcept {
    const auto validation = evidence_atlas_requirements(
        {atlas.records, atlas.record_count, atlas.atlas_identity, atlas.atlas_generation},
        maximum_records);
    if (!validation.ok()) return {evidence_atlas_image_code_v1::invalid_atlas};
    if (atlas.record_count >
        (std::numeric_limits<std::uint64_t>::max()
         - evidence_atlas_image_header_bytes_v1)
            / evidence_atlas_image_record_bytes_v1)
        return {evidence_atlas_image_code_v1::size_overflow};
    return {evidence_atlas_image_code_v1::success,
            evidence_atlas_image_header_bytes_v1
                + atlas.record_count * evidence_atlas_image_record_bytes_v1};
}

evidence_atlas_image_result_v1 encode_evidence_atlas_v1(
    evidence_atlas_view_v1 atlas, void *destination,
    std::uint64_t destination_bytes, std::uint64_t maximum_records) noexcept {
    const auto requirement = evidence_atlas_image_requirements_v1(atlas, maximum_records);
    if (!requirement.ok()) return requirement;
    if (destination == nullptr) return {evidence_atlas_image_code_v1::missing_buffer,
                                        requirement.required_bytes};
    if (destination_bytes < requirement.required_bytes)
        return {evidence_atlas_image_code_v1::insufficient_buffer,
                requirement.required_bytes};
    auto *bytes = static_cast<unsigned char *>(destination);
    std::memcpy(bytes, magic, sizeof(magic));
    put_u32(bytes + 8, 1);
    put_u32(bytes + 12, evidence_atlas_image_header_bytes_v1);
    put_u64(bytes + 16, requirement.required_bytes);
    put_identity(bytes + 24, atlas.atlas_identity);
    put_u64(bytes + 40, atlas.atlas_generation);
    put_u64(bytes + 48, atlas.record_count);
    put_u64(bytes + 56, 0);
    auto *out = bytes + evidence_atlas_image_header_bytes_v1;
    for (std::uint64_t index = 0; index < atlas.record_count; ++index, out += 80) {
        const auto &record = atlas.records[index];
        put_u32(out, record.schema_version);
        put_u32(out + 4, record.record_bytes);
        put_identity(out + 8, record.evidence_identity);
        put_identity(out + 24, record.subject_atom_identity);
        put_identity(out + 40, record.source_identity);
        put_u64(out + 56, record.observation_generation);
        put_u64(out + 64, record.observation_count);
        put_u32(out + 72, static_cast<std::uint32_t>(record.kind));
        put_u32(out + 76, static_cast<std::uint32_t>(record.disposition));
    }
    put_u64(bytes + 56, checksum(bytes + 64, requirement.required_bytes - 64));
    return requirement;
}

evidence_atlas_image_result_v1 decode_evidence_atlas_v1(
    const void *source, std::uint64_t source_bytes,
    std::uint64_t maximum_records, evidence_atlas_builder_v1 *output) noexcept {
    if (output == nullptr) return {evidence_atlas_image_code_v1::build_failure};
    output->reset();
    if (source == nullptr) return {evidence_atlas_image_code_v1::missing_buffer};
    if (source_bytes < 64) return {evidence_atlas_image_code_v1::invalid_total_size};
    const auto *bytes = static_cast<const unsigned char *>(source);
    if (std::memcmp(bytes, magic, sizeof(magic)) != 0)
        return {evidence_atlas_image_code_v1::invalid_magic};
    if (get_u32(bytes + 8) != 1)
        return {evidence_atlas_image_code_v1::unsupported_schema};
    if (get_u32(bytes + 12) != 64)
        return {evidence_atlas_image_code_v1::invalid_header_size};
    const auto total_bytes = get_u64(bytes + 16);
    const auto count = get_u64(bytes + 48);
    if (count > maximum_records)
        return {evidence_atlas_image_code_v1::record_limit_exceeded};
    if (count > (std::numeric_limits<std::uint64_t>::max() - 64) / 80
        || total_bytes != 64 + count * 80 || total_bytes != source_bytes)
        return {evidence_atlas_image_code_v1::invalid_total_size};
    if (get_u64(bytes + 56) != checksum(bytes + 64, total_bytes - 64))
        return {evidence_atlas_image_code_v1::checksum_mismatch};
    std::vector<atom_evidence_record_v1> records;
    try {
        records.resize(count);
    } catch (const std::bad_alloc &) {
        return {evidence_atlas_image_code_v1::allocation_failure};
    } catch (const std::length_error &) {
        return {evidence_atlas_image_code_v1::allocation_failure};
    }
    const auto *in = bytes + 64;
    for (std::uint64_t index = 0; index < count; ++index, in += 80) {
        auto &record = records[index];
        record.schema_version = get_u32(in);
        record.record_bytes = get_u32(in + 4);
        record.evidence_identity = get_identity(in + 8);
        record.subject_atom_identity = get_identity(in + 24);
        record.source_identity = get_identity(in + 40);
        record.observation_generation = get_u64(in + 56);
        record.observation_count = get_u64(in + 64);
        record.kind = static_cast<evidence_kind>(get_u32(in + 72));
        record.disposition = static_cast<evidence_disposition_v1>(get_u32(in + 76));
        if (!validate_atom_evidence_record_v1(record).valid())
            return {evidence_atlas_image_code_v1::invalid_record, total_bytes, index};
    }
    const evidence_atlas_source_v1 atlas_source{
        records.data(), records.size(), get_identity(bytes + 24), get_u64(bytes + 40)};
    const auto requirements = evidence_atlas_requirements(atlas_source, maximum_records);
    if (!requirements.ok())
        return {evidence_atlas_image_code_v1::invalid_record, total_bytes,
                requirements.index};
    const auto build = output->fill(atlas_source, requirements.requirements, maximum_records);
    if (!build.ok()) return {evidence_atlas_image_code_v1::build_failure};
    return {evidence_atlas_image_code_v1::success, total_bytes, count};
}

} // namespace cellshard::compiler::evidence
