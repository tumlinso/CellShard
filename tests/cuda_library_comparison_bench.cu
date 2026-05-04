#include <CellShard/export/dataset_export.hh>

#include "../src/bucket/major_nnz_raw.cuh"
#include "../src/convert/compressed_from_coo_raw.cuh"
#include "../src/convert/compressed_transpose_raw.cuh"
#include "../src/convert/coo_from_compressed_raw.cuh"
#include "../src/repack/kernels/sharded_blocked_ell.cuh"

#include <cub/cub.cuh>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>

#include <fcntl.h>
#include <sys/file.h>
#include <unistd.h>

namespace cs = ::cellshard;
namespace cse = ::cellshard::exporting;

namespace {

struct lock_guard_file {
    int fd = -1;

    explicit lock_guard_file(const char *path) {
        fd = ::open(path, O_CREAT | O_RDWR, 0666);
        if (fd < 0) throw std::runtime_error("could not open benchmark mutex");
        if (::flock(fd, LOCK_EX) != 0) throw std::runtime_error("could not lock benchmark mutex");
    }

    ~lock_guard_file() {
        if (fd >= 0) {
            (void) ::flock(fd, LOCK_UN);
            (void) ::close(fd);
        }
    }
};

template<typename T>
struct device_buffer {
    T *ptr = nullptr;
    std::size_t count = 0;

    device_buffer() = default;
    explicit device_buffer(std::size_t n) { reset(n); }
    device_buffer(const device_buffer &) = delete;
    device_buffer &operator=(const device_buffer &) = delete;

    ~device_buffer() { if (ptr != nullptr) cudaFree(ptr); }

    void reset(std::size_t n) {
        if (ptr != nullptr) cudaFree(ptr);
        ptr = nullptr;
        count = n;
        cudaError_t err = cudaMalloc((void **) &ptr, (n == 0 ? 1 : n) * sizeof(T));
        if (err != cudaSuccess) throw std::runtime_error(cudaGetErrorString(err));
    }
};

struct raw_device_buffer {
    void *ptr = nullptr;
    std::size_t bytes = 0;

    raw_device_buffer() = default;
    explicit raw_device_buffer(std::size_t n) { reset(n); }
    raw_device_buffer(const raw_device_buffer &) = delete;
    raw_device_buffer &operator=(const raw_device_buffer &) = delete;

    ~raw_device_buffer() { if (ptr != nullptr) cudaFree(ptr); }

    void reset(std::size_t n) {
        if (ptr != nullptr) cudaFree(ptr);
        ptr = nullptr;
        bytes = n;
        cudaError_t err = cudaMalloc(&ptr, n == 0 ? 1 : n);
        if (err != cudaSuccess) throw std::runtime_error(cudaGetErrorString(err));
    }
};

struct entry {
    unsigned int row;
    unsigned int col;
    unsigned short value_key;
};

struct scenario {
    std::string id;
    unsigned int rows;
    unsigned int cols;
    bool coo_sorted_by_row = false;
    std::vector<entry> coo;
    std::vector<unsigned int> csr_ptr;
    std::vector<unsigned int> csr_col;
    std::vector<__half> csr_val;
};

struct metric {
    std::string operation;
    std::string scenario_id;
    double custom_mean_ms = 0.0;
    double custom_median_ms = 0.0;
    double custom_cv = 0.0;
    double library_mean_ms = 0.0;
    double library_median_ms = 0.0;
    double library_cv = 0.0;
    double speedup = 0.0;
    double custom_delta_percent = 0.0;
    bool correctness = false;
    bool timing_valid = true;
};

static void cuda_check(cudaError_t err, const char *label) {
    if (err != cudaSuccess) {
        std::ostringstream os;
        os << label << ": " << cudaGetErrorString(err);
        throw std::runtime_error(os.str());
    }
}

static void require_ok(int ok, const char *label) {
    if (!ok) throw std::runtime_error(label);
}

template<typename T>
static void upload(device_buffer<T> &dst, const std::vector<T> &src, cudaStream_t stream) {
    dst.reset(src.size());
    if (!src.empty()) {
        cuda_check(cudaMemcpyAsync(dst.ptr, src.data(), src.size() * sizeof(T), cudaMemcpyHostToDevice, stream), "upload");
    }
}

template<typename T>
static std::vector<T> download(const T *src, std::size_t n, cudaStream_t stream) {
    std::vector<T> out(n);
    if (n != 0) {
        cuda_check(cudaMemcpyAsync(out.data(), src, n * sizeof(T), cudaMemcpyDeviceToHost, stream), "download");
    }
    cuda_check(cudaStreamSynchronize(stream), "download sync");
    return out;
}

static unsigned short value_key(unsigned int row, unsigned int col) {
    return (unsigned short) (((row * 131u + col * 17u) % 1023u) + 1u);
}

static __half value_half(unsigned short key) {
    return __float2half((float) key);
}

static unsigned short value_key_from_real(double value) {
    if (!std::isfinite(value)) value = 0.0;
    const double scaled = std::fabs(value) * 1024.0;
    const unsigned long rounded = (unsigned long) std::llround(scaled);
    return (unsigned short) std::min<unsigned long>(65535ul, std::max<unsigned long>(1ul, rounded));
}

static unsigned short half_key(__half v) {
    return (unsigned short) std::lround(__half2float(v));
}

static scenario make_scenario(const std::string &id, unsigned int rows, unsigned int cols, unsigned int per_row, int mode) {
    scenario s;
    s.id = id;
    s.rows = rows;
    s.cols = cols;
    s.csr_ptr.assign((std::size_t) rows + 1u, 0u);

    std::vector<entry> sorted;
    sorted.reserve((std::size_t) rows * per_row);
    for (unsigned int r = 0; r < rows; ++r) {
        unsigned int k = per_row;
        if (mode == 1 && (r % 19u) == 0u) k = per_row * 8u;
        if (mode == 1 && (r % 23u) == 1u) k = 1u;
        if (k > cols) k = cols;
        for (unsigned int j = 0; j < k; ++j) {
            unsigned int c = 0;
            if (mode == 1) {
                c = (j < (k / 2u + 1u)) ? ((j * 3u + r) % std::max(1u, cols / 16u)) : ((r * 977u + j * 37u) % cols);
            } else if (mode == 2) {
                c = (r * 1103515245u + j * 2654435761u + 17u) % cols;
            } else {
                c = (r * 17u + j * 104729u + j * j * 13u) % cols;
            }
            sorted.push_back(entry{r, c, value_key(r, c)});
        }
    }
    std::sort(sorted.begin(), sorted.end(), [](const entry &a, const entry &b) {
        return std::tie(a.row, a.col, a.value_key) < std::tie(b.row, b.col, b.value_key);
    });
    sorted.erase(std::unique(sorted.begin(), sorted.end(), [](const entry &a, const entry &b) {
        return a.row == b.row && a.col == b.col;
    }), sorted.end());

    for (const entry &e : sorted) ++s.csr_ptr[(std::size_t) e.row + 1u];
    for (unsigned int r = 0; r < rows; ++r) s.csr_ptr[(std::size_t) r + 1u] += s.csr_ptr[r];
    s.csr_col.reserve(sorted.size());
    s.csr_val.reserve(sorted.size());
    for (const entry &e : sorted) {
        s.csr_col.push_back(e.col);
        s.csr_val.push_back(value_half(e.value_key));
    }

    s.coo = sorted;
    std::stable_sort(s.coo.begin(), s.coo.end(), [](const entry &a, const entry &b) {
        const unsigned int ha = (a.row * 73856093u) ^ (a.col * 19349663u);
        const unsigned int hb = (b.row * 73856093u) ^ (b.col * 19349663u);
        return ha < hb;
    });
    s.coo_sorted_by_row = false;
    return s;
}

static std::string scenario_id_from_path(const std::filesystem::path &path) {
    std::string id = path.stem().string();
    for (char &c : id) {
        const bool ok = (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || (c >= '0' && c <= '9');
        if (!ok) c = '_';
    }
    return "real_mtx_" + id;
}

static scenario load_matrix_market_scenario(const std::filesystem::path &path, const std::string &id) {
    std::ifstream in(path);
    if (!in) throw std::runtime_error("could not open Matrix Market input: " + path.string());

    std::string line;
    if (!std::getline(in, line) || line.rfind("%%MatrixMarket", 0) != 0) {
        throw std::runtime_error("Matrix Market banner missing: " + path.string());
    }
    do {
        if (!std::getline(in, line)) throw std::runtime_error("Matrix Market size line missing: " + path.string());
    } while (!line.empty() && line[0] == '%');

    std::istringstream dims(line);
    unsigned long long rows64 = 0;
    unsigned long long cols64 = 0;
    unsigned long long nnz64 = 0;
    if (!(dims >> rows64 >> cols64 >> nnz64)) throw std::runtime_error("Matrix Market size line is invalid: " + path.string());
    if (rows64 > 0xffffffffull || cols64 > 0xffffffffull || nnz64 > 0xffffffffull) {
        throw std::runtime_error("Matrix Market dimensions exceed CellShard benchmark index width");
    }

    scenario s;
    s.id = id.empty() ? scenario_id_from_path(path) : id;
    s.rows = (unsigned int) rows64;
    s.cols = (unsigned int) cols64;
    s.coo.reserve((std::size_t) nnz64);

    unsigned long long r = 0;
    unsigned long long c = 0;
    double v = 0.0;
    while (in >> r >> c >> v) {
        if (r == 0ull || c == 0ull || r > rows64 || c > cols64) {
            throw std::runtime_error("Matrix Market coordinate out of range: " + path.string());
        }
        if (v == 0.0) continue;
        s.coo.push_back(entry{(unsigned int) (r - 1ull), (unsigned int) (c - 1ull), value_key_from_real(v)});
    }

    std::sort(s.coo.begin(), s.coo.end(), [](const entry &a, const entry &b) {
        return std::tie(a.row, a.col, a.value_key) < std::tie(b.row, b.col, b.value_key);
    });
    s.coo.erase(std::unique(s.coo.begin(), s.coo.end(), [](const entry &a, const entry &b) {
        return a.row == b.row && a.col == b.col;
    }), s.coo.end());

    s.csr_ptr.assign((std::size_t) s.rows + 1u, 0u);
    for (const entry &e : s.coo) ++s.csr_ptr[(std::size_t) e.row + 1u];
    for (unsigned int row = 0; row < s.rows; ++row) s.csr_ptr[(std::size_t) row + 1u] += s.csr_ptr[row];
    s.csr_col.reserve(s.coo.size());
    s.csr_val.reserve(s.coo.size());
    for (const entry &e : s.coo) {
        s.csr_col.push_back(e.col);
        s.csr_val.push_back(value_half(e.value_key));
    }
    s.coo_sorted_by_row = true;
    return s;
}

static scenario scenario_from_csr_export(const cse::csr_matrix_export &csr, const std::string &id) {
    if (csr.rows > 0xffffffffull || csr.cols > 0xffffffffull || csr.data.size() > 0xffffffffull) {
        throw std::runtime_error("CSR export exceeds CellShard benchmark index width");
    }
    scenario s;
    s.id = id;
    s.rows = (unsigned int) csr.rows;
    s.cols = (unsigned int) csr.cols;
    s.csr_ptr.resize(csr.indptr.size());
    s.csr_col.resize(csr.indices.size());
    s.csr_val.resize(csr.data.size());
    s.coo.reserve(csr.data.size());

    for (std::size_t i = 0; i < csr.indptr.size(); ++i) {
        if (csr.indptr[i] < 0 || (unsigned long long) csr.indptr[i] > 0xffffffffull) {
            throw std::runtime_error("CSR export indptr exceeds CellShard benchmark index width");
        }
        s.csr_ptr[i] = (unsigned int) csr.indptr[i];
    }
    for (unsigned int row = 0; row < s.rows; ++row) {
        const unsigned int begin = s.csr_ptr[row];
        const unsigned int end = s.csr_ptr[(std::size_t) row + 1u];
        if (end < begin || end > csr.indices.size()) throw std::runtime_error("CSR export row pointer is invalid");
        for (unsigned int p = begin; p < end; ++p) {
            const unsigned int col = (unsigned int) csr.indices[p];
            const unsigned short key = value_key_from_real(csr.data[p]);
            s.csr_col[p] = col;
            s.csr_val[p] = value_half(key);
            s.coo.push_back(entry{row, col, key});
        }
    }
    s.coo_sorted_by_row = true;
    return s;
}

static std::string csh5_scenario_id_from_path(const std::filesystem::path &path, std::size_t row_limit) {
    std::string id = path.stem().string();
    for (char &c : id) {
        const bool ok = (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || (c >= '0' && c <= '9');
        if (!ok) c = '_';
    }
    std::ostringstream os;
    os << "native_csh5_" << id;
    if (row_limit != 0u) os << "_rows" << row_limit;
    return os.str();
}

static scenario load_csh5_scenario(const std::filesystem::path &path, std::size_t row_limit) {
    cse::csr_matrix_export csr;
    std::string error;
    if (row_limit == 0u) {
        if (!cse::load_dataset_as_csr(path.c_str(), &csr, &error)) {
            throw std::runtime_error("failed to load native csh5 as CSR: " + error);
        }
    } else {
        std::vector<std::uint64_t> rows(row_limit);
        for (std::size_t i = 0; i < row_limit; ++i) rows[i] = (std::uint64_t) i;
        if (!cse::load_dataset_rows_as_csr(path.c_str(), rows.data(), rows.size(), &csr, &error)) {
            throw std::runtime_error("failed to load native csh5 row subset as CSR: " + error);
        }
    }
    return scenario_from_csr_export(csr, csh5_scenario_id_from_path(path, row_limit));
}

static std::vector<entry> entries_from_csr(unsigned int rows,
                                           const std::vector<unsigned int> &ptr,
                                           const std::vector<unsigned int> &minor,
                                           const std::vector<__half> &val) {
    std::vector<entry> out;
    out.reserve(val.size());
    for (unsigned int r = 0; r < rows; ++r) {
        for (unsigned int p = ptr[r]; p < ptr[(std::size_t) r + 1u]; ++p) {
            out.push_back(entry{r, minor[p], half_key(val[p])});
        }
    }
    std::sort(out.begin(), out.end(), [](const entry &a, const entry &b) {
        return std::tie(a.row, a.col, a.value_key) < std::tie(b.row, b.col, b.value_key);
    });
    return out;
}

static std::vector<entry> entries_from_coo(const std::vector<unsigned int> &row,
                                           const std::vector<unsigned int> &col,
                                           const std::vector<__half> &val) {
    std::vector<entry> out;
    out.reserve(val.size());
    for (std::size_t i = 0; i < val.size(); ++i) out.push_back(entry{row[i], col[i], half_key(val[i])});
    std::sort(out.begin(), out.end(), [](const entry &a, const entry &b) {
        return std::tie(a.row, a.col, a.value_key) < std::tie(b.row, b.col, b.value_key);
    });
    return out;
}

static bool same_entries(const std::vector<entry> &a, const std::vector<entry> &b) {
    if (a.size() != b.size()) return false;
    for (std::size_t i = 0; i < a.size(); ++i) {
        if (a[i].row != b[i].row || a[i].col != b[i].col || a[i].value_key != b[i].value_key) return false;
    }
    return true;
}

static std::vector<float> time_ms(const std::function<void()> &fn, int warmup, int repeats, cudaStream_t stream) {
    cudaEvent_t start = nullptr;
    cudaEvent_t stop = nullptr;
    cuda_check(cudaEventCreate(&start), "event create start");
    cuda_check(cudaEventCreate(&stop), "event create stop");
    for (int i = 0; i < warmup; ++i) {
        fn();
        cuda_check(cudaStreamSynchronize(stream), "warmup sync");
    }
    std::vector<float> samples;
    samples.reserve((std::size_t) repeats);
    for (int i = 0; i < repeats; ++i) {
        cuda_check(cudaEventRecord(start, stream), "event record start");
        fn();
        cuda_check(cudaEventRecord(stop, stream), "event record stop");
        cuda_check(cudaEventSynchronize(stop), "event sync stop");
        float ms = 0.0f;
        cuda_check(cudaEventElapsedTime(&ms, start, stop), "event elapsed");
        samples.push_back(ms);
    }
    cuda_check(cudaEventDestroy(start), "event destroy start");
    cuda_check(cudaEventDestroy(stop), "event destroy stop");
    return samples;
}

static double mean(const std::vector<float> &v) {
    if (v.empty()) return 0.0;
    return std::accumulate(v.begin(), v.end(), 0.0) / (double) v.size();
}

static double median(std::vector<float> v) {
    if (v.empty()) return 0.0;
    std::sort(v.begin(), v.end());
    const std::size_t mid = v.size() / 2u;
    return (v.size() & 1u) ? v[mid] : ((double) v[mid - 1u] + (double) v[mid]) * 0.5;
}

static double cv(const std::vector<float> &v) {
    const double m = mean(v);
    if (m == 0.0 || v.size() < 2u) return 0.0;
    double accum = 0.0;
    for (float x : v) accum += ((double) x - m) * ((double) x - m);
    return std::sqrt(accum / (double) (v.size() - 1u)) / m;
}

static metric make_metric(const std::string &op,
                          const std::string &sid,
                          const std::vector<float> &custom,
                          const std::vector<float> &library,
                          bool correctness) {
    metric m;
    m.operation = op;
    m.scenario_id = sid;
    m.custom_mean_ms = mean(custom);
    m.custom_median_ms = median(custom);
    m.custom_cv = cv(custom);
    m.library_mean_ms = mean(library);
    m.library_median_ms = median(library);
    m.library_cv = cv(library);
    m.speedup = m.library_median_ms > 0.0 ? m.custom_median_ms / m.library_median_ms : 0.0;
    m.custom_delta_percent = m.library_median_ms > 0.0 ? ((m.custom_median_ms / m.library_median_ms) - 1.0) * 100.0 : 0.0;
    m.correctness = correctness;
    m.timing_valid = m.custom_cv <= 0.25 && m.library_cv <= 0.25;
    return m;
}

static std::size_t scan_bytes_for(unsigned int n, cudaStream_t stream) {
    std::size_t bytes = 0;
    cub::DeviceScan::ExclusiveSum(nullptr, bytes, (unsigned int *) nullptr, (unsigned int *) nullptr, n, stream);
    return bytes;
}

__global__ static void count_nonzero_half_atomic(const __half *values, unsigned int n, unsigned int *out) {
    const unsigned int tid = (unsigned int) (blockIdx.x * blockDim.x + threadIdx.x);
    const unsigned int stride = (unsigned int) (gridDim.x * blockDim.x);
    unsigned int local = 0u;
    for (unsigned int i = tid; i < n; i += stride) {
        local += __half2float(values[i]) != 0.0f ? 1u : 0u;
    }
    if (local != 0u) atomicAdd(out, local);
}

struct nonzero_half {
    __host__ __device__ unsigned int operator()(const __half &v) const {
        return __half2float(v) != 0.0f ? 1u : 0u;
    }
};

static metric bench_coo_to_compressed(const scenario &s, int warmup, int repeats, cudaStream_t stream) {
    const unsigned int nnz = (unsigned int) s.coo.size();
    std::vector<unsigned int> h_row(nnz), h_col(nnz);
    std::vector<__half> h_val(nnz);
    for (unsigned int i = 0; i < nnz; ++i) {
        h_row[i] = s.coo[i].row;
        h_col[i] = s.coo[i].col;
        h_val[i] = value_half(s.coo[i].value_key);
    }

    device_buffer<unsigned int> d_row, d_col;
    device_buffer<__half> d_val;
    upload(d_row, h_row, stream);
    upload(d_col, h_col, stream);
    upload(d_val, h_val, stream);

    device_buffer<unsigned int> custom_ptr((std::size_t) s.rows + 1u), custom_heads(s.rows), custom_minor(nnz);
    device_buffer<__half> custom_val(nnz);
    raw_device_buffer custom_scan(scan_bytes_for(s.rows + 1u, stream));
    device_buffer<unsigned int> lib_ptr((std::size_t) s.rows + 1u), lib_sort_row(nnz), lib_minor(nnz), lib_perm(nnz);
    device_buffer<__half> lib_val(nnz);
    std::size_t sort_bytes = 0;
    require_ok(cs::convert::compressed_from_coo_library_workspace_bytes(s.rows, s.cols, nnz, lib_sort_row.ptr, lib_minor.ptr, stream, &sort_bytes),
               "coo_to_compressed library workspace");
    raw_device_buffer lib_tmp(sort_bytes);

    auto custom_op = [&]() {
        if (s.coo_sorted_by_row) {
            require_ok(cs::convert::build_compressed_from_sorted_coo_custom_raw(s.rows, nnz, d_row.ptr, d_col.ptr, d_val.ptr,
                                                                               custom_ptr.ptr, custom_minor.ptr,
                                                                               custom_val.ptr, stream),
                       "coo_to_compressed sorted custom");
        } else {
            require_ok(cs::convert::build_compressed_from_coo_custom_raw(s.rows, nnz, d_row.ptr, d_col.ptr, d_val.ptr,
                                                                         custom_ptr.ptr, custom_heads.ptr, custom_minor.ptr,
                                                                         custom_val.ptr, custom_scan.ptr, custom_scan.bytes, stream),
                       "coo_to_compressed custom");
        }
    };
    auto library_op = [&]() {
        require_ok(cs::convert::build_compressed_from_coo_library_raw(s.rows, s.cols, nnz, d_row.ptr, d_col.ptr, d_val.ptr,
                                                                      lib_ptr.ptr, lib_sort_row.ptr, lib_minor.ptr,
                                                                      lib_val.ptr, lib_perm.ptr, lib_tmp.ptr, lib_tmp.bytes, stream),
                   "coo_to_compressed library");
    };

    custom_op();
    library_op();
    const auto cptr = download(custom_ptr.ptr, (std::size_t) s.rows + 1u, stream);
    const auto cminor = download(custom_minor.ptr, nnz, stream);
    const auto cval = download(custom_val.ptr, nnz, stream);
    const auto lptr = download(lib_ptr.ptr, (std::size_t) s.rows + 1u, stream);
    const auto lminor = download(lib_minor.ptr, nnz, stream);
    const auto lval = download(lib_val.ptr, nnz, stream);
    const bool ok = same_entries(entries_from_csr(s.rows, cptr, cminor, cval), entries_from_csr(s.rows, lptr, lminor, lval)) &&
                    same_entries(entries_from_csr(s.rows, lptr, lminor, lval), entries_from_csr(s.rows, s.csr_ptr, s.csr_col, s.csr_val));

    return make_metric("coo_to_compressed", s.id, time_ms(custom_op, warmup, repeats, stream), time_ms(library_op, warmup, repeats, stream), ok);
}

static metric bench_compressed_to_coo(const scenario &s, int warmup, int repeats, cudaStream_t stream) {
    const unsigned int nnz = (unsigned int) s.csr_col.size();
    device_buffer<unsigned int> d_ptr, d_col;
    device_buffer<__half> d_val;
    upload(d_ptr, s.csr_ptr, stream);
    upload(d_col, s.csr_col, stream);
    upload(d_val, s.csr_val, stream);

    device_buffer<unsigned int> custom_row(nnz), custom_col(nnz), lib_row(nnz), lib_col(nnz);
    device_buffer<__half> custom_val(nnz), lib_val(nnz);

    auto custom_op = [&]() {
        require_ok(cs::convert::build_coo_from_compressed_custom_raw(s.rows, nnz, d_ptr.ptr, d_col.ptr, d_val.ptr,
                                                                     custom_row.ptr, custom_col.ptr, custom_val.ptr, stream),
                   "compressed_to_coo custom");
    };
    auto library_op = [&]() {
        require_ok(cs::convert::build_coo_from_compressed_library_raw(s.rows, nnz, d_ptr.ptr, d_col.ptr, d_val.ptr,
                                                                      lib_row.ptr, lib_col.ptr, lib_val.ptr, stream),
                   "compressed_to_coo library");
    };

    custom_op();
    library_op();
    const bool ok = same_entries(entries_from_coo(download(custom_row.ptr, nnz, stream), download(custom_col.ptr, nnz, stream), download(custom_val.ptr, nnz, stream)),
                                 entries_from_coo(download(lib_row.ptr, nnz, stream), download(lib_col.ptr, nnz, stream), download(lib_val.ptr, nnz, stream)));
    return make_metric("compressed_to_coo", s.id, time_ms(custom_op, warmup, repeats, stream), time_ms(library_op, warmup, repeats, stream), ok);
}

static metric bench_compressed_transpose(const scenario &s, int warmup, int repeats, cudaStream_t stream) {
    const unsigned int nnz = (unsigned int) s.csr_col.size();
    device_buffer<unsigned int> d_ptr, d_col;
    device_buffer<__half> d_val;
    upload(d_ptr, s.csr_ptr, stream);
    upload(d_col, s.csr_col, stream);
    upload(d_val, s.csr_val, stream);

    device_buffer<unsigned int> custom_ptr((std::size_t) s.cols + 1u), custom_heads(s.cols), custom_minor(nnz);
    device_buffer<__half> custom_val(nnz);
    raw_device_buffer custom_scan(scan_bytes_for(s.cols + 1u, stream));
    device_buffer<unsigned int> lib_ptr((std::size_t) s.cols + 1u), lib_minor(nnz);
    device_buffer<__half> lib_val(nnz);
    std::size_t lib_bytes = 0;
    require_ok(cs::convert::compressed_transpose_library_workspace_bytes(s.rows, s.cols, nnz, d_ptr.ptr, d_col.ptr, d_val.ptr,
                                                                         lib_ptr.ptr, lib_minor.ptr, lib_val.ptr, stream, &lib_bytes),
               "transpose library workspace");
    raw_device_buffer lib_tmp(lib_bytes);

    auto custom_op = [&]() {
        require_ok(cs::convert::build_compressed_transpose_custom_raw(s.rows, s.cols, nnz, d_ptr.ptr, d_col.ptr, d_val.ptr,
                                                                      custom_ptr.ptr, custom_heads.ptr, custom_minor.ptr,
                                                                      custom_val.ptr, custom_scan.ptr, custom_scan.bytes, stream),
                   "transpose custom");
    };
    auto library_op = [&]() {
        require_ok(cs::convert::build_compressed_transpose_library_raw(s.rows, s.cols, nnz, d_ptr.ptr, d_col.ptr, d_val.ptr,
                                                                       lib_ptr.ptr, lib_minor.ptr, lib_val.ptr,
                                                                       lib_tmp.ptr, lib_tmp.bytes, stream),
                   "transpose library");
    };

    custom_op();
    library_op();
    const bool ok = same_entries(entries_from_csr(s.cols, download(custom_ptr.ptr, (std::size_t) s.cols + 1u, stream),
                                                  download(custom_minor.ptr, nnz, stream), download(custom_val.ptr, nnz, stream)),
                                 entries_from_csr(s.cols, download(lib_ptr.ptr, (std::size_t) s.cols + 1u, stream),
                                                  download(lib_minor.ptr, nnz, stream), download(lib_val.ptr, nnz, stream)));
    return make_metric("compressed_transpose", s.id, time_ms(custom_op, warmup, repeats, stream), time_ms(library_op, warmup, repeats, stream), ok);
}

static metric bench_major_nnz_bucket(const scenario &s, int warmup, int repeats, cudaStream_t stream) {
    const unsigned int rows = s.rows;
    device_buffer<unsigned int> d_ptr;
    upload(d_ptr, s.csr_ptr, stream);
    device_buffer<unsigned int> c_count(rows), c_sorted(rows), c_order_in(rows), c_order_out(rows), c_buckets(9);
    device_buffer<unsigned int> l_count(rows), l_sorted(rows), l_order_in(rows), l_order_out(rows), l_buckets(9);
    std::size_t sort_bytes = 0;
    require_ok(cs::bucket::major_nnz_bucket_sort_scratch_bytes(rows, &sort_bytes), "bucket sort workspace");
    raw_device_buffer c_tmp(sort_bytes), l_tmp(sort_bytes);

    auto custom_op = [&]() {
        require_ok(cs::bucket::build_major_nnz_bucket_plan_custom_raw(d_ptr.ptr, rows, c_count.ptr, c_sorted.ptr, c_order_in.ptr,
                                                                      c_order_out.ptr, c_buckets.ptr, 8u, c_tmp.ptr, c_tmp.bytes, stream),
                   "major_nnz custom");
    };
    auto library_op = [&]() {
        require_ok(cs::bucket::build_major_nnz_bucket_plan_library_raw(d_ptr.ptr, rows, l_count.ptr, l_sorted.ptr, l_order_in.ptr,
                                                                       l_order_out.ptr, l_buckets.ptr, 8u, l_tmp.ptr, l_tmp.bytes, stream),
                   "major_nnz library");
    };

    custom_op();
    library_op();
    const bool ok = download(c_sorted.ptr, rows, stream) == download(l_sorted.ptr, rows, stream) &&
                    download(c_order_out.ptr, rows, stream) == download(l_order_out.ptr, rows, stream);
    return make_metric("major_nnz_bucket", s.id, time_ms(custom_op, warmup, repeats, stream), time_ms(library_op, warmup, repeats, stream), ok);
}

static metric bench_block_key_runs(const scenario &s, int warmup, int repeats, cudaStream_t stream) {
    const unsigned int nnz = (unsigned int) s.csr_col.size();
    const unsigned int block_size = 32u;
    const unsigned int row_blocks = (s.rows + block_size - 1u) / block_size;
    std::vector<unsigned long long> keys;
    keys.reserve(nnz);
    for (const entry &e : s.coo) {
        keys.push_back(((unsigned long long) (e.row / block_size) << 32u) | (unsigned long long) (e.col / block_size));
    }
    std::sort(keys.begin(), keys.end());
    keys.erase(std::unique(keys.begin(), keys.end()), keys.end());
    std::vector<unsigned long long> expanded_keys;
    expanded_keys.reserve(nnz);
    for (unsigned long long key : keys) {
        expanded_keys.push_back(key);
        if ((key & 3ull) == 0ull) expanded_keys.push_back(key);
    }
    std::sort(expanded_keys.begin(), expanded_keys.end());
    const unsigned int key_count = (unsigned int) expanded_keys.size();
    device_buffer<unsigned long long> d_keys;
    upload(d_keys, expanded_keys, stream);
    device_buffer<unsigned int> head_flags(key_count), row_counts(row_blocks);
    device_buffer<unsigned long long> unique_keys(key_count);
    device_buffer<unsigned int> run_lengths(key_count), num_runs(1);
    std::size_t rle_bytes = 0;
    cub::DeviceRunLengthEncode::Encode(nullptr, rle_bytes, d_keys.ptr, unique_keys.ptr, run_lengths.ptr, num_runs.ptr, key_count, stream);
    raw_device_buffer rle_tmp(rle_bytes);
    const dim3 block(256, 1, 1);
    const dim3 grid(std::min<unsigned int>(4096u, (key_count + 255u) / 256u), 1, 1);

    auto custom_op = [&]() {
        cuda_check(cudaMemsetAsync(row_counts.ptr, 0, (std::size_t) row_blocks * sizeof(unsigned int), stream), "memset row counts");
        cs::repack::kernels::mark_block_group_heads_and_count<<<grid, block, 0, stream>>>(
            key_count,
            reinterpret_cast<const cs::types::u64 *>(d_keys.ptr),
            row_blocks,
            head_flags.ptr,
            row_counts.ptr);
        cuda_check(cudaGetLastError(), "block run custom");
    };
    auto library_op = [&]() {
        cuda_check(cub::DeviceRunLengthEncode::Encode(rle_tmp.ptr, rle_tmp.bytes, d_keys.ptr, unique_keys.ptr, run_lengths.ptr,
                                                      num_runs.ptr, key_count, stream),
                   "block run library rle");
    };

    custom_op();
    library_op();
    const auto runs_host = download(num_runs.ptr, 1u, stream);
    const bool ok = runs_host[0] == keys.size();
    return make_metric("block_key_runs", s.id, time_ms(custom_op, warmup, repeats, stream), time_ms(library_op, warmup, repeats, stream), ok);
}

static metric bench_blocked_nonzero_count(const scenario &s, int warmup, int repeats, cudaStream_t stream) {
    const unsigned int slots = std::max<unsigned int>(1u, s.rows * 8u);
    std::vector<__half> values(slots);
    unsigned int expected = 0u;
    for (unsigned int i = 0; i < slots; ++i) {
        const bool live = (i % 11u) != 0u && ((i * 17u + s.rows) % 29u) != 0u;
        values[i] = live ? __float2half((float) ((i % 31u) + 1u)) : __float2half(0.0f);
        expected += live ? 1u : 0u;
    }
    device_buffer<__half> d_values;
    upload(d_values, values, stream);
    device_buffer<unsigned int> custom_out(1), library_out(1);
    std::size_t reduce_bytes = 0;
    {
        cub::TransformInputIterator<unsigned int, nonzero_half, __half *> flags(d_values.ptr, nonzero_half{});
        cuda_check(cub::DeviceReduce::Sum(nullptr, reduce_bytes, flags, library_out.ptr, slots, stream), "nonzero reduce workspace");
    }
    raw_device_buffer reduce_tmp(reduce_bytes);
    const dim3 block(256, 1, 1);
    const dim3 grid(std::min<unsigned int>(4096u, (slots + 255u) / 256u), 1, 1);

    auto custom_op = [&]() {
        cuda_check(cudaMemsetAsync(custom_out.ptr, 0, sizeof(unsigned int), stream), "memset nonzero count");
        count_nonzero_half_atomic<<<grid, block, 0, stream>>>(d_values.ptr, slots, custom_out.ptr);
        cuda_check(cudaGetLastError(), "nonzero custom");
    };
    auto library_op = [&]() {
        cub::TransformInputIterator<unsigned int, nonzero_half, __half *> flags(d_values.ptr, nonzero_half{});
        cuda_check(cub::DeviceReduce::Sum(reduce_tmp.ptr, reduce_tmp.bytes, flags, library_out.ptr, slots, stream), "nonzero library reduce");
    };

    custom_op();
    library_op();
    const bool ok = download(custom_out.ptr, 1u, stream)[0] == expected && download(library_out.ptr, 1u, stream)[0] == expected;
    return make_metric("blocked_nonzero_count", s.id, time_ms(custom_op, warmup, repeats, stream), time_ms(library_op, warmup, repeats, stream), ok);
}

static void write_outputs(const std::filesystem::path &dir,
                          const std::vector<metric> &metrics,
                          int warmup,
                          int repeats,
                          const std::string &mutex_path,
                          const cudaDeviceProp &prop) {
    std::filesystem::create_directories(dir);
    std::filesystem::create_directories(dir / "impl_a");
    std::filesystem::create_directories(dir / "impl_b");
    {
        std::ofstream f(dir / "compare_config.json");
        f << "{\n"
          << "  \"comparison_id\": \"cellshard-custom-vs-cusparse-cub\",\n"
          << "  \"impl_a_name\": \"custom\",\n"
          << "  \"impl_b_name\": \"library\",\n"
          << "  \"scenario_id\": \"synthetic_sparse_suite\",\n"
          << "  \"warmup\": " << warmup << ",\n"
          << "  \"repeats\": " << repeats << ",\n"
          << "  \"profile_friendly\": true,\n"
          << "  \"mutex_path\": \"" << mutex_path << "\"\n"
          << "}\n";
    }
    for (const char *impl : {"impl_a", "impl_b"}) {
        const bool custom = std::strcmp(impl, "impl_a") == 0;
        {
            std::ofstream f(dir / impl / "run_config.json");
            f << "{\n"
              << "  \"implementation\": \"" << (custom ? "custom" : "library") << "\",\n"
              << "  \"scenario_id\": \"synthetic_sparse_suite\",\n"
              << "  \"warmup\": " << warmup << ",\n"
              << "  \"repeats\": " << repeats << "\n"
              << "}\n";
        }
        {
            std::ofstream f(dir / impl / "results.json");
            f << "{\n"
              << "  \"implementation\": \"" << (custom ? "custom" : "library") << "\",\n"
              << "  \"results\": [\n";
            for (std::size_t i = 0; i < metrics.size(); ++i) {
                const metric &m = metrics[i];
                f << "    {\"operation\":\"" << m.operation << "\",\"scenario\":\"" << m.scenario_id
                  << "\",\"median_ms\":" << (custom ? m.custom_median_ms : m.library_median_ms)
                  << ",\"mean_ms\":" << (custom ? m.custom_mean_ms : m.library_mean_ms)
                  << ",\"cv\":" << (custom ? m.custom_cv : m.library_cv)
                  << ",\"correctness\":" << (m.correctness ? "true" : "false") << "}";
                if (i + 1u != metrics.size()) f << ",";
                f << "\n";
            }
            f << "  ]\n}\n";
        }
    }
    {
        std::ofstream f(dir / "summary.json");
        f << "{\n"
          << "  \"impl_a_name\": \"custom\",\n"
          << "  \"impl_b_name\": \"library\",\n"
          << "  \"device\": \"" << prop.name << "\",\n"
          << "  \"cuda_compute_capability\": \"" << prop.major << "." << prop.minor << "\",\n"
          << "  \"correctness_passed\": " << (std::all_of(metrics.begin(), metrics.end(), [](const metric &m) { return m.correctness; }) ? "true" : "false") << ",\n"
          << "  \"results\": [\n";
        for (std::size_t i = 0; i < metrics.size(); ++i) {
            const metric &m = metrics[i];
            f << "    {\"operation\":\"" << m.operation << "\",\"scenario\":\"" << m.scenario_id
              << "\",\"custom_median_ms\":" << m.custom_median_ms
              << ",\"custom_mean_ms\":" << m.custom_mean_ms
              << ",\"library_median_ms\":" << m.library_median_ms
              << ",\"library_mean_ms\":" << m.library_mean_ms
              << ",\"speedup_factor\":" << m.speedup
              << ",\"custom_delta_percent\":" << m.custom_delta_percent
              << ",\"correctness\":" << (m.correctness ? "true" : "false")
              << ",\"timing_valid\":" << (m.timing_valid ? "true" : "false") << "}";
            if (i + 1u != metrics.size()) f << ",";
            f << "\n";
        }
        f << "  ]\n}\n";
    }
    {
        std::ofstream f(dir / "summary.txt");
        f << "CellShard CUDA custom vs cuSPARSE/CUB comparison\n";
        f << "Device: " << prop.name << " sm_" << prop.major << prop.minor << "\n";
        f << "Warmup: " << warmup << " Repeats: " << repeats << " Mutex: " << mutex_path << "\n\n";
        f << std::left << std::setw(26) << "operation" << std::setw(28) << "scenario"
          << std::right << std::setw(14) << "custom ms" << std::setw(14) << "library ms"
          << std::setw(12) << "speedup" << std::setw(14) << "custom delta" << std::setw(8) << "ok" << "\n";
        for (const metric &m : metrics) {
            f << std::left << std::setw(26) << m.operation << std::setw(28) << m.scenario_id
              << std::right << std::setw(14) << std::fixed << std::setprecision(4) << m.custom_median_ms
              << std::setw(14) << m.library_median_ms
              << std::setw(12) << m.speedup
              << std::setw(13) << m.custom_delta_percent << "%"
              << std::setw(8) << (m.correctness ? "yes" : "no") << "\n";
        }
    }
}

} // namespace

int main(int argc, char **argv) {
    try {
        std::filesystem::path output_dir = "cuda-library-comparison";
        std::vector<std::filesystem::path> matrix_market_inputs;
        std::vector<std::filesystem::path> csh5_inputs;
        std::size_t csh5_rows = 0u;
        int synthetic = 1;
        int warmup = 3;
        int repeats = 20;
        for (int i = 1; i < argc; ++i) {
            if (std::strcmp(argv[i], "--output-dir") == 0 && i + 1 < argc) output_dir = argv[++i];
            else if (std::strcmp(argv[i], "--warmup") == 0 && i + 1 < argc) warmup = std::atoi(argv[++i]);
            else if (std::strcmp(argv[i], "--repeats") == 0 && i + 1 < argc) repeats = std::atoi(argv[++i]);
            else if (std::strcmp(argv[i], "--matrix-market") == 0 && i + 1 < argc) matrix_market_inputs.push_back(argv[++i]);
            else if (std::strcmp(argv[i], "--csh5") == 0 && i + 1 < argc) csh5_inputs.push_back(argv[++i]);
            else if (std::strcmp(argv[i], "--csh5-rows") == 0 && i + 1 < argc) csh5_rows = (std::size_t) std::strtoull(argv[++i], nullptr, 10);
            else if (std::strcmp(argv[i], "--real-only") == 0) synthetic = 0;
        }
        if (repeats < 3) repeats = 3;
        if (warmup < 0) warmup = 0;

        const char *mutex_env = std::getenv("COMPARE_BENCHMARK_MUTEX_PATH");
        const std::string mutex_path = mutex_env != nullptr && mutex_env[0] != '\0' ? mutex_env : "/tmp/compare_benchmarks.lock";
        lock_guard_file lock(mutex_path.c_str());

        int device = 0;
        cuda_check(cudaSetDevice(device), "set device");
        cudaDeviceProp prop{};
        cuda_check(cudaGetDeviceProperties(&prop, device), "device properties");
        cudaStream_t stream = nullptr;
        cuda_check(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), "stream create");

        std::vector<scenario> scenarios;
        if (synthetic) {
            scenarios.push_back(make_scenario("small_sanity", 8u, 7u, 3u, 0));
            scenarios.push_back(make_scenario("uniform_sparse_rows", 8192u, 4096u, 8u, 0));
            scenarios.push_back(make_scenario("skewed_hot_rows_cols", 8192u, 4096u, 8u, 1));
            scenarios.push_back(make_scenario("single_cell_tall_sparse", 50000u, 2000u, 4u, 2));
            scenarios.push_back(make_scenario("rect_transpose_wide", 4096u, 32768u, 8u, 0));
            scenarios.push_back(make_scenario("rect_transpose_tall", 32768u, 4096u, 4u, 2));
        }
        for (const std::filesystem::path &path : matrix_market_inputs) {
            scenarios.push_back(load_matrix_market_scenario(path, ""));
        }
        for (const std::filesystem::path &path : csh5_inputs) {
            scenarios.push_back(load_csh5_scenario(path, csh5_rows));
        }
        if (scenarios.empty()) {
            throw std::runtime_error("no benchmark scenarios selected");
        }

        std::vector<metric> metrics;
        for (const scenario &s : scenarios) {
            std::cerr << "running " << s.id << " nnz=" << s.csr_col.size() << "\n";
            metrics.push_back(bench_coo_to_compressed(s, warmup, repeats, stream));
            metrics.push_back(bench_compressed_to_coo(s, warmup, repeats, stream));
            metrics.push_back(bench_compressed_transpose(s, warmup, repeats, stream));
            metrics.push_back(bench_major_nnz_bucket(s, warmup, repeats, stream));
            metrics.push_back(bench_block_key_runs(s, warmup, repeats, stream));
            metrics.push_back(bench_blocked_nonzero_count(s, warmup, repeats, stream));
        }

        cuda_check(cudaStreamDestroy(stream), "stream destroy");
        write_outputs(output_dir, metrics, warmup, repeats, mutex_path, prop);
        const bool ok = std::all_of(metrics.begin(), metrics.end(), [](const metric &m) { return m.correctness; });
        return ok ? 0 : 2;
    } catch (const std::exception &ex) {
        std::cerr << "benchmark failed: " << ex.what() << "\n";
        return 1;
    }
}
