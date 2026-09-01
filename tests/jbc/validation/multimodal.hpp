#pragma once
#include "fixtures.hpp"
namespace cellshard::jbc::validation {
struct modality_identity {
    global_id modality_id = 0;
    global_id entity_domain_id = 0;
    global_id entity_order_id = 0;
    global_id feature_domain_id = 0;
    global_id feature_order_id = 0;
};
inline bool valid_multimodal_spine(const modality_identity* modalities,
                                   std::uint32_t count) noexcept {
    if (modalities == nullptr || count < 2) return false;
    const auto entity_domain = modalities[0].entity_domain_id;
    const auto entity_order = modalities[0].entity_order_id;
    for (std::uint32_t i = 0; i < count; ++i) {
        const auto& item = modalities[i];
        if (item.modality_id == 0 || item.entity_domain_id != entity_domain ||
            item.entity_order_id != entity_order || item.feature_domain_id == 0 || item.feature_order_id == 0) return false;
        if (i != 0 && (modalities[i - 1].modality_id >= item.modality_id ||
                       modalities[i - 1].feature_domain_id >= item.feature_domain_id)) return false;
    }
    return true;
}
}  // namespace cellshard::jbc::validation
