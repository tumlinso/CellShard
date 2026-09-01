#include <CellShard/runtime/v2/logical_node.hh>

#include <cuda_runtime_api.h>
#include <numa.h>
#include <sched.h>
#include <sys/wait.h>
#include <unistd.h>

#include <array>
#include <cerrno>
#include <cstdlib>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <string>
#include <vector>

namespace {

struct gpu_record {
    int device = -1;
    int numa_node = -1;
    char pci_bus_id[32]{};
};

struct worker_record {
    int requested_node = -1;
    int observed_cpu = -1;
    int observed_node = -1;
    int selected_device = -1;
    int selected_device_node = -1;
    int cuda_status = -1;
};

int pci_numa_node(const char *cuda_pci) {
    std::string pci(cuda_pci == nullptr ? "" : cuda_pci);
    if (pci.size() == 16 && pci.rfind("00000000:", 0) == 0) {
        pci.erase(0, 4);
    }
    std::ifstream input("/sys/bus/pci/devices/" + pci + "/numa_node");
    int node = -1;
    input >> node;
    return node;
}

int cpu_count_on_node(int node) {
    bitmask *mask = numa_allocate_cpumask();
    if (mask == nullptr || numa_node_to_cpus(node, mask) != 0) {
        if (mask != nullptr) {
            numa_free_cpumask(mask);
        }
        return 0;
    }
    int count = 0;
    for (unsigned int cpu = 0; cpu < mask->size; ++cpu) {
        count += numa_bitmask_isbitset(mask, cpu) != 0;
    }
    numa_free_cpumask(mask);
    return count;
}

cellshard::content_digest cpu_set_identity(int node) {
    bitmask *mask = numa_allocate_cpumask();
    std::uint64_t hash = 1469598103934665603ULL;
    if (mask != nullptr && numa_node_to_cpus(node, mask) == 0) {
        for (unsigned int cpu = 0; cpu < mask->size; ++cpu) {
            const std::uint8_t bit =
                static_cast<std::uint8_t>(numa_bitmask_isbitset(mask, cpu) != 0);
            hash ^= bit;
            hash *= 1099511628211ULL;
        }
    }
    if (mask != nullptr) {
        numa_free_cpumask(mask);
    }
    cellshard::content_digest digest{};
    digest.algorithm = cellshard::digest_algorithm::legacy_fnv1a64;
    digest.used_bytes = 8;
    for (std::size_t i = 0; i < 8; ++i) {
        digest.bytes[i] = std::byte((hash >> (i * 8U)) & 0xffU);
    }
    return digest;
}

worker_record run_worker(int node, int device) {
    worker_record result{};
    result.requested_node = node;
    if (numa_run_on_node(node) != 0) {
        return result;
    }
    result.observed_cpu = sched_getcpu();
    result.observed_node = result.observed_cpu >= 0
        ? numa_node_of_cpu(result.observed_cpu)
        : -1;
    result.selected_device = device;
    result.selected_device_node = node;
    result.cuda_status = static_cast<int>(cudaSetDevice(device));
    if (result.cuda_status == static_cast<int>(cudaSuccess)) {
        result.cuda_status = static_cast<int>(cudaFree(nullptr));
    }
    return result;
}

} // namespace

int main(int argc, char **argv) {
    if (argc == 5 && std::string(argv[1]) == "--worker") {
        const int node = std::atoi(argv[2]);
        const int device = std::atoi(argv[3]);
        const int output_fd = std::atoi(argv[4]);
        const worker_record record = run_worker(node, device);
        return write(output_fd, &record, sizeof(record))
                == static_cast<ssize_t>(sizeof(record))
            ? 0
            : 1;
    }
    if (numa_available() < 0 || numa_max_node() != 1) {
        std::fprintf(stderr, "campaign requires exactly two NUMA nodes\n");
        return 2;
    }
    int device_count = 0;
    if (cudaGetDeviceCount(&device_count) != cudaSuccess || device_count != 4) {
        std::fprintf(stderr, "campaign requires exactly four visible CUDA devices\n");
        return 3;
    }
    std::vector<gpu_record> gpus(static_cast<std::size_t>(device_count));
    for (int device = 0; device < device_count; ++device) {
        gpus[device].device = device;
        if (cudaDeviceGetPCIBusId(gpus[device].pci_bus_id,
                                  sizeof(gpus[device].pci_bus_id), device)
            != cudaSuccess) {
            return 4;
        }
        gpus[device].numa_node = pci_numa_node(gpus[device].pci_bus_id);
        if (gpus[device].numa_node < 0 || gpus[device].numa_node > 1) {
            return 5;
        }
    }

    std::array<cellshard::runtime_v2::logical_node, 2> nodes{};
    for (int node = 0; node < 2; ++node) {
        const int cpu_count = cpu_count_on_node(node);
        if (cpu_count <= 0) {
            return 6;
        }
        nodes[node] = {static_cast<std::uint32_t>(node + 1),
                       {static_cast<std::uint32_t>(node),
                        static_cast<std::uint32_t>(cpu_count), 1,
                        cpu_set_identity(node)}};
    }
    if (!cellshard::runtime_v2::valid_logical_nodes(
            {nodes.data(), nodes.size()})) {
        return 11;
    }

    std::array<worker_record, 2> workers{};
    for (int node = 0; node < 2; ++node) {
        int selected_device = -1;
        for (const gpu_record &gpu : gpus) {
            if (gpu.numa_node == node) {
                selected_device = gpu.device;
                break;
            }
        }
        if (selected_device < 0) {
            return 12;
        }
        int pipe_descriptors[2]{};
        if (pipe(pipe_descriptors) != 0) {
            return 7;
        }
        const pid_t child = fork();
        if (child < 0) {
            return 8;
        }
        if (child == 0) {
            close(pipe_descriptors[0]);
            char node_argument[16]{};
            char device_argument[16]{};
            char descriptor_argument[16]{};
            std::snprintf(node_argument, sizeof(node_argument), "%d", node);
            std::snprintf(device_argument, sizeof(device_argument), "%d",
                          selected_device);
            std::snprintf(descriptor_argument, sizeof(descriptor_argument), "%d",
                          pipe_descriptors[1]);
            execl(argv[0], argv[0], "--worker", node_argument, device_argument,
                  descriptor_argument, static_cast<char *>(nullptr));
            _exit(127);
        }
        close(pipe_descriptors[1]);
        const ssize_t received =
            read(pipe_descriptors[0], &workers[node], sizeof(worker_record));
        close(pipe_descriptors[0]);
        int child_status = 0;
        if (waitpid(child, &child_status, 0) != child
            || !WIFEXITED(child_status) || WEXITSTATUS(child_status) != 0
            || received != static_cast<ssize_t>(sizeof(worker_record))) {
            return 9;
        }
    }

    bool valid = true;
    for (int node = 0; node < 2; ++node) {
        valid &= workers[node].requested_node == node;
        valid &= workers[node].observed_node == node;
        valid &= workers[node].selected_device_node == node;
        valid &= workers[node].cuda_status == static_cast<int>(cudaSuccess);
    }

    std::printf("{\"schema\":\"CS-JBC-RUNTIME-CAMPAIGN/1\",");
    std::printf("\"process_model\":\"one-worker-process-per-numa-node\",");
    std::printf("\"numa_nodes\":2,\"visible_cuda_devices\":4,");
    std::printf("\"gpus\":[");
    for (int device = 0; device < device_count; ++device) {
        std::printf("%s{\"device\":%d,\"pci\":\"%s\",\"numa_node\":%d}",
                    device == 0 ? "" : ",", gpus[device].device,
                    gpus[device].pci_bus_id, gpus[device].numa_node);
    }
    std::printf("],\"workers\":[");
    for (int node = 0; node < 2; ++node) {
        std::printf("%s{\"requested_node\":%d,\"observed_cpu\":%d,"
                    "\"observed_node\":%d,\"selected_device\":%d}",
                    node == 0 ? "" : ",", workers[node].requested_node,
                    workers[node].observed_cpu, workers[node].observed_node,
                    workers[node].selected_device);
    }
    std::printf("],\"valid\":%s}\n", valid ? "true" : "false");
    return valid ? 0 : 10;
}
