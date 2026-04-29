#pragma once

#include <cstdint>

namespace cellshard {

inline constexpr std::uint32_t dataset_assay_invalid_row = 0xffffffffu;
inline constexpr std::uint32_t dataset_assay_invalid_id = 0xffffffffu;

// These numeric values intentionally mirror cudaBioTypes' public enum order.
// CellShard keeps the POD ABI independent while Cellerator validates meaning
// against cudaBioTypes.
enum dataset_modality_kind : std::uint32_t {
    dataset_modality_scrna = 0u,
    dataset_modality_scatac = 1u,
    dataset_modality_cite_adt = 2u,
    dataset_modality_spatial = 3u,
    dataset_modality_fragments = 4u,
    dataset_modality_multimodal = 5u
};

enum dataset_observation_kind : std::uint32_t {
    dataset_observation_cell = 0u,
    dataset_observation_nucleus = 1u,
    dataset_observation_spot = 2u,
    dataset_observation_fragment_aggregate = 3u,
    dataset_observation_bulk_sample = 4u
};

enum dataset_feature_kind : std::uint32_t {
    dataset_feature_gene = 0u,
    dataset_feature_transcript = 1u,
    dataset_feature_exon = 2u,
    dataset_feature_peak = 3u,
    dataset_feature_fragment = 4u,
    dataset_feature_protein = 5u,
    dataset_feature_spatial_coordinate = 6u,
    dataset_feature_custom = 7u
};

enum dataset_value_semantics : std::uint32_t {
    dataset_values_raw_counts = 0u,
    dataset_values_binary_accessibility = 1u,
    dataset_values_normalized_counts = 2u,
    dataset_values_log_transformed = 3u,
    dataset_values_dense_projection = 4u
};

enum dataset_processing_state : std::uint32_t {
    dataset_processing_raw = 0u,
    dataset_processing_qc_filtered = 1u,
    dataset_processing_normalized = 2u,
    dataset_processing_log_transformed = 3u,
    dataset_processing_feature_selected = 4u,
    dataset_processing_integrated = 5u,
    dataset_processing_imputed = 6u
};

enum dataset_axis_kind : std::uint32_t {
    dataset_axis_observations = 0u,
    dataset_axis_features = 1u
};

enum dataset_pairing_kind : std::uint32_t {
    dataset_pairing_none = 0u,
    dataset_pairing_exact_observation = 1u,
    dataset_pairing_partial_observation = 2u,
    dataset_pairing_donor_level = 3u,
    dataset_pairing_sample_level = 4u
};

struct dataset_assay_semantics {
    std::uint32_t modality = dataset_modality_scrna;
    std::uint32_t observation_unit = dataset_observation_cell;
    std::uint32_t feature_type = dataset_feature_gene;
    std::uint32_t value_semantics = dataset_values_raw_counts;
    std::uint32_t processing_state = dataset_processing_raw;
    std::uint32_t row_axis = dataset_axis_observations;
    std::uint32_t col_axis = dataset_axis_features;
    std::uint32_t feature_namespace = 0u;
};

struct dataset_assay_row_map_view {
    std::uint32_t global_observation_count = 0u;
    std::uint32_t assay_row_count = 0u;
    const std::uint32_t *global_to_assay_row = nullptr;
    const std::uint32_t *assay_row_to_global = nullptr;
};

struct dataset_assay_view {
    const char *assay_id = nullptr;
    dataset_assay_semantics semantics{};
    std::uint64_t rows = 0u;
    std::uint64_t cols = 0u;
    std::uint64_t nnz = 0u;
    std::uint64_t feature_order_hash = 0u;
    dataset_assay_row_map_view row_map{};
};

struct dataset_pairing_view {
    std::uint32_t pairing = dataset_pairing_none;
    std::uint32_t assay_count = 0u;
    const dataset_assay_view *assays = nullptr;
};

struct dataset_assay_pack_manifest_view {
    const char *assay_id = nullptr;
    std::uint64_t shard_id = 0u;
    std::uint64_t global_row_begin = 0u;
    std::uint64_t global_row_end = 0u;
    std::uint64_t local_row_begin = 0u;
    std::uint64_t local_row_end = 0u;
    const char *path = nullptr;
};

inline bool dataset_assay_is_valid_row(const std::uint32_t row) {
    return row != dataset_assay_invalid_row;
}

inline bool dataset_assay_semantics_valid(const dataset_assay_semantics *semantics) {
    return semantics != nullptr && semantics->row_axis != semantics->col_axis;
}

inline std::uint32_t dataset_assay_row_for_global(const dataset_assay_row_map_view *map,
                                                  const std::uint32_t global_row) {
    if (map == nullptr || map->global_to_assay_row == nullptr || global_row >= map->global_observation_count) {
        return dataset_assay_invalid_row;
    }
    return map->global_to_assay_row[global_row];
}

inline bool dataset_validate_assay_row_map(const dataset_assay_row_map_view *map) {
    if (map == nullptr) return false;
    if (map->global_observation_count != 0u && map->global_to_assay_row == nullptr) return false;
    if (map->assay_row_count != 0u && map->assay_row_to_global == nullptr) return false;

    for (std::uint32_t global = 0u; global < map->global_observation_count; ++global) {
        const std::uint32_t assay_row = map->global_to_assay_row[global];
        if (!dataset_assay_is_valid_row(assay_row)) continue;
        if (assay_row >= map->assay_row_count) return false;
        if (map->assay_row_to_global[assay_row] != global) return false;
    }
    for (std::uint32_t assay_row = 0u; assay_row < map->assay_row_count; ++assay_row) {
        const std::uint32_t global = map->assay_row_to_global[assay_row];
        if (global >= map->global_observation_count) return false;
        if (map->global_to_assay_row[global] != assay_row) return false;
    }
    return true;
}

inline bool dataset_resolve_paired_rows(const dataset_assay_row_map_view *lhs,
                                        const dataset_assay_row_map_view *rhs,
                                        const std::uint32_t global_row,
                                        std::uint32_t *lhs_row,
                                        std::uint32_t *rhs_row) {
    if (lhs_row != nullptr) *lhs_row = dataset_assay_invalid_row;
    if (rhs_row != nullptr) *rhs_row = dataset_assay_invalid_row;
    if (lhs == nullptr || rhs == nullptr) return false;
    if (lhs->global_observation_count != rhs->global_observation_count) return false;
    if (global_row >= lhs->global_observation_count) return false;

    const std::uint32_t resolved_lhs = dataset_assay_row_for_global(lhs, global_row);
    const std::uint32_t resolved_rhs = dataset_assay_row_for_global(rhs, global_row);
    if (lhs_row != nullptr) *lhs_row = resolved_lhs;
    if (rhs_row != nullptr) *rhs_row = resolved_rhs;
    return dataset_assay_is_valid_row(resolved_lhs) || dataset_assay_is_valid_row(resolved_rhs);
}

inline bool dataset_validate_pairing_view(const dataset_pairing_view *pairing) {
    if (pairing == nullptr) return false;
    if (pairing->assay_count == 0u) return pairing->assays == nullptr;
    if (pairing->assays == nullptr) return false;
    if (pairing->pairing != dataset_pairing_exact_observation
        && pairing->pairing != dataset_pairing_partial_observation) {
        return false;
    }

    const std::uint32_t global_count = pairing->assays[0].row_map.global_observation_count;
    for (std::uint32_t i = 0u; i < pairing->assay_count; ++i) {
        const dataset_assay_view &assay = pairing->assays[i];
        if (!dataset_assay_semantics_valid(&assay.semantics)) return false;
        if (assay.row_map.global_observation_count != global_count) return false;
        if (!dataset_validate_assay_row_map(&assay.row_map)) return false;
        if (pairing->pairing == dataset_pairing_exact_observation
            && assay.row_map.assay_row_count != global_count) {
            return false;
        }
        if (pairing->pairing == dataset_pairing_exact_observation) {
            for (std::uint32_t global = 0u; global < global_count; ++global) {
                if (!dataset_assay_is_valid_row(assay.row_map.global_to_assay_row[global])) return false;
            }
        }
    }
    return true;
}

inline bool dataset_pack_manifest_range_valid(const dataset_assay_pack_manifest_view *pack) {
    return pack != nullptr
        && pack->assay_id != nullptr
        && pack->path != nullptr
        && pack->global_row_begin <= pack->global_row_end
        && pack->local_row_begin <= pack->local_row_end
        && (pack->global_row_end - pack->global_row_begin) == (pack->local_row_end - pack->local_row_begin);
}

} // namespace cellshard
