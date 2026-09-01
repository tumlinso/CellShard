#pragma once

#include <cstdint>
#include <type_traits>

#if defined(CELLSHARD_RUNTIME_V2_HAS_NUMABRAID)
#include <numabraid/numabraid.hh>
#endif

namespace cellshard::runtime_v2 {

struct numabraid_capabilities {
    bool package_available = false;
    bool topology_api = false;
    bool forwarding_api = false;
    bool nccl_provider = false;
    std::uint32_t version_major = 0;
    std::uint32_t version_minor = 0;
    std::uint32_t version_patch = 0;
};

[[nodiscard]] constexpr numabraid_capabilities
discover_numabraid_capabilities() noexcept {
#if defined(CELLSHARD_RUNTIME_V2_HAS_NUMABRAID)
    // numaBraid 0.0.0 intentionally publishes identity only. Do not infer
    // benchmark-internal topology or relay APIs as stable capabilities.
    return {true, false, false, false,
            static_cast<std::uint32_t>(numabraid::version_major),
            static_cast<std::uint32_t>(numabraid::version_minor),
            static_cast<std::uint32_t>(numabraid::version_patch)};
#else
    return {};
#endif
}

static_assert(std::is_trivially_copyable_v<numabraid_capabilities>);

} // namespace cellshard::runtime_v2
