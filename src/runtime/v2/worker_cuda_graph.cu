#include <CellShard/runtime/v2/worker_cuda_graph.cuh>

#include <vector>

namespace cellshard::runtime_v2 {

struct worker_cuda_graph::impl {
    int device_id = -1;
    cudaGraph_t graph = nullptr;
    cudaGraphExec_t executable = nullptr;
    std::vector<cudaGraphNode_t> nodes;
    std::vector<std::uint32_t> command_indices;
    std::vector<cudaMemcpyKind> kinds;
    std::vector<std::uint64_t> byte_limits;

    ~impl() noexcept {
        int previous = -1;
        (void)cudaGetDevice(&previous);
        if (device_id >= 0) {
            (void)cudaSetDevice(device_id);
        }
        if (executable != nullptr) {
            (void)cudaGraphExecDestroy(executable);
        }
        if (graph != nullptr) {
            (void)cudaGraphDestroy(graph);
        }
        if (previous >= 0) {
            (void)cudaSetDevice(previous);
        }
    }
};

namespace {
bool valid_binding(const cuda_graph_copy_binding &binding,
                   std::size_t command_count) noexcept {
    return binding.command_index < command_count && binding.source != nullptr
        && binding.destination != nullptr && binding.bytes != 0
        && binding.kind != cudaMemcpyDefault;
}
} // namespace

worker_cuda_graph::worker_cuda_graph() noexcept = default;
worker_cuda_graph::~worker_cuda_graph() noexcept { reset(); }

status_code worker_cuda_graph::prepare(
    int device_id, const runtime_command_program &program,
    array_view<cuda_graph_copy_binding> bindings) noexcept {
    if (impl_ || device_id < 0 || !valid_runtime_command_program(program)
        || bindings.empty() || bindings.size != program.commands.size) {
        return status_code::invalid_input;
    }
    try {
        auto candidate = std::make_unique<impl>();
        candidate->device_id = device_id;
        candidate->nodes.resize(bindings.size, nullptr);
        candidate->command_indices.resize(bindings.size);
        candidate->kinds.resize(bindings.size);
        candidate->byte_limits.resize(bindings.size);
        int previous = -1;
        if (cudaGetDevice(&previous) != cudaSuccess
            || cudaSetDevice(device_id) != cudaSuccess
            || cudaGraphCreate(&candidate->graph, 0) != cudaSuccess) {
            (void)cudaSetDevice(previous);
            return status_code::cuda_failure;
        }
        for (std::size_t i = 0; i < bindings.size; ++i) {
            const auto &binding = bindings[i];
            if (!valid_binding(binding, program.commands.size)
                || binding.command_index != i
                || program.commands[i].logical_node == 0) {
                (void)cudaSetDevice(previous);
                return status_code::invalid_input;
            }
            const auto &command = program.commands[i];
            std::vector<cudaGraphNode_t> dependencies;
            dependencies.reserve(command.dependency_count);
            for (std::uint32_t d = 0; d < command.dependency_count; ++d) {
                dependencies.push_back(candidate->nodes[
                    program.dependencies[command.dependency_begin + d]]);
            }
            if (cudaGraphAddMemcpyNode1D(
                    &candidate->nodes[i], candidate->graph,
                    dependencies.data(), dependencies.size(),
                    binding.destination, binding.source,
                    static_cast<std::size_t>(binding.bytes), binding.kind)
                != cudaSuccess) {
                (void)cudaSetDevice(previous);
                return status_code::cuda_failure;
            }
            candidate->command_indices[i] = binding.command_index;
            candidate->kinds[i] = binding.kind;
            candidate->byte_limits[i] = binding.bytes;
        }
        if (cudaGraphInstantiate(&candidate->executable, candidate->graph,
                                 nullptr, nullptr, 0) != cudaSuccess
            || cudaSetDevice(previous) != cudaSuccess) {
            return status_code::cuda_failure;
        }
        impl_ = std::move(candidate);
    } catch (...) {
        return status_code::allocation_failure;
    }
    return status_code::success;
}

status_code worker_cuda_graph::launch(
    array_view<cuda_graph_copy_binding> bindings,
    cudaStream_t stream) noexcept {
    if (!impl_ || bindings.size != impl_->nodes.size() || stream == nullptr) {
        return status_code::invalid_input;
    }
    int previous = -1;
    if (cudaGetDevice(&previous) != cudaSuccess
        || cudaSetDevice(impl_->device_id) != cudaSuccess) {
        return status_code::cuda_failure;
    }
    for (std::size_t i = 0; i < bindings.size; ++i) {
        const auto &binding = bindings[i];
        if (binding.command_index != impl_->command_indices[i]
            || binding.source == nullptr || binding.destination == nullptr
            || binding.bytes == 0 || binding.bytes > impl_->byte_limits[i]
            || binding.kind != impl_->kinds[i]
            || cudaGraphExecMemcpyNodeSetParams1D(
                   impl_->executable, impl_->nodes[i], binding.destination,
                   binding.source, static_cast<std::size_t>(binding.bytes),
                   binding.kind) != cudaSuccess) {
            (void)cudaSetDevice(previous);
            return status_code::invalid_input;
        }
    }
    const cudaError_t launch_status = cudaGraphLaunch(impl_->executable, stream);
    const cudaError_t restore_status = cudaSetDevice(previous);
    return launch_status == cudaSuccess && restore_status == cudaSuccess
        ? status_code::success
        : status_code::cuda_failure;
}

bool worker_cuda_graph::valid() const noexcept {
    return impl_ && impl_->graph != nullptr && impl_->executable != nullptr;
}

void worker_cuda_graph::reset() noexcept {
    impl_.reset();
}

} // namespace cellshard::runtime_v2
