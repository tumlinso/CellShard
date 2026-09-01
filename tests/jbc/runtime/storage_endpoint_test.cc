#include <CellShard/runtime/v2/storage_endpoint.hh>

#include <cassert>

using namespace cellshard;
using namespace cellshard::runtime_v2;

int main() {
    storage_endpoint endpoint{
        source_provider_id{7}, source_location_id{9},
        storage_endpoint_kind::block_device, 1,
        {storage_access_mode::direct, 4096, 32, 1ULL << 20, true},
    };
    assert(valid_storage_endpoint(endpoint));
    endpoint.access.required_alignment = 3000;
    assert(!valid_storage_endpoint(endpoint));
    endpoint.access.required_alignment = 4096;
    endpoint.access.preferred_request_bytes = 4097;
    assert(!valid_storage_endpoint(endpoint));
    endpoint.access.preferred_request_bytes = 1ULL << 20;
    endpoint.access.read_only = false;
    assert(!valid_storage_endpoint(endpoint));
}
