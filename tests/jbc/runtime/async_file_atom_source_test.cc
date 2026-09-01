#include <CellShard/runtime/v2/async_file_atom_source.hh>

#include <array>
#include <cassert>
#include <chrono>
#include <thread>
#include <unistd.h>

using namespace cellshard;
using namespace cellshard::runtime_v2;

int main() {
    char path[] = "/tmp/cellshard-async-source-XXXXXX";
    const int descriptor = ::mkstemp(path);
    assert(descriptor >= 0);
    const std::array bytes{std::byte{1}, std::byte{2}, std::byte{3},
                           std::byte{4}, std::byte{5}, std::byte{6}};
    assert(::write(descriptor, bytes.data(), bytes.size())
           == static_cast<ssize_t>(bytes.size()));
    assert(::close(descriptor) == 0);

    async_file_atom_source provider;
    assert(open_async_file_atom_source(path, storage_object_id{3}, &provider)
           == status_code::success);
    std::array<std::byte, 4> destination{};
    const std::array ranges{
        atom_range{storage_object_id{3}, 1, 2, 0},
        atom_range{storage_object_id{3}, 4, 2, 2},
    };
    const atom_source_request request{{ranges.data(), ranges.size()},
                                      destination.data(), destination.size()};
    atom_request_token token{};
    const atom_source_ref source = provider.ref();
    assert(valid_atom_source(source));
    assert(source.ops->submit(source.context, request, &token)
           == status_code::success);
    atom_request_state state = atom_request_state::pending;
    for (int attempt = 0; attempt < 100 && state == atom_request_state::pending;
         ++attempt) {
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
        state = source.ops->query(source.context, token);
    }
    assert(state == atom_request_state::complete);
    assert(destination[0] == std::byte{2} && destination[1] == std::byte{3});
    assert(destination[2] == std::byte{5} && destination[3] == std::byte{6});
    assert(source.ops->query(source.context, atom_request_token{999})
           == atom_request_state::invalid);
    provider.reset();
    assert(::unlink(path) == 0);
}
