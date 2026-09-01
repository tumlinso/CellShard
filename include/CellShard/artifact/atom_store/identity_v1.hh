#pragma once

#include <CellShard/artifact/atom_store/content_digest_v1.hh>

#include <cstdint>
#include <type_traits>

namespace cellshard::artifact::atom_store {

template<class Tag>
struct strong_identity_v1 {
    std::uint64_t high = 0;
    std::uint64_t low = 0;
    [[nodiscard]] constexpr bool valid() const noexcept {
        return high != 0 || low != 0;
    }
};

struct semantic_identity_tag_v1;
struct action_identity_tag_v1;
struct materialization_identity_tag_v1;
struct replica_identity_tag_v1;

using semantic_identity_v1 = strong_identity_v1<semantic_identity_tag_v1>;
using action_identity_v1 = strong_identity_v1<action_identity_tag_v1>;
using materialization_identity_v1
    = strong_identity_v1<materialization_identity_tag_v1>;
using replica_identity_v1 = strong_identity_v1<replica_identity_tag_v1>;

struct atom_identity_bundle_v1 {
    semantic_identity_v1 semantic{};
    content_digest_v1 content{};
    action_identity_v1 action{};
    materialization_identity_v1 materialization{};
    replica_identity_v1 replica{};
};

[[nodiscard]] constexpr bool valid_atom_identity_bundle_v1(
    const atom_identity_bundle_v1 &identity) noexcept {
    return identity.semantic.valid() && valid_content_digest_v1(identity.content)
        && identity.action.valid() && identity.materialization.valid()
        && identity.replica.valid();
}

template<class LeftTag, class RightTag>
bool operator==(strong_identity_v1<LeftTag>,
                strong_identity_v1<RightTag>) = delete;

template<class Tag>
[[nodiscard]] constexpr bool operator==(strong_identity_v1<Tag> left,
                                        strong_identity_v1<Tag> right) noexcept {
    return left.high == right.high && left.low == right.low;
}

static_assert(std::is_trivially_copyable<semantic_identity_v1>::value);
static_assert(std::is_trivially_copyable<atom_identity_bundle_v1>::value);

} // namespace cellshard::artifact::atom_store
