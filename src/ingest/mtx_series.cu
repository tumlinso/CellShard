#include <CellShard/ingest/mtx_series.cuh>
#include <CellShard/io/csh5/api.cuh>

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <sys/stat.h>
#include <thread>
#include <unordered_map>
#include <utility>
#include <vector>

namespace fs = std::filesystem;

namespace cellshard {
namespace ingest {
namespace mtx_series {
namespace {

namespace cs = ::cellshard;

struct timer {
    std::chrono::steady_clock::time_point start = std::chrono::steady_clock::now();
    double seconds() const {
        return std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
    }
};

struct text_column {
    std::vector<std::uint32_t> offsets{0u};
    std::string data;

    void reserve(std::size_t count, std::size_t bytes) {
        offsets.reserve(count + 1u);
        data.reserve(bytes);
    }
    void append(std::string_view value) {
        if (data.size() + value.size() > std::numeric_limits<std::uint32_t>::max()) {
            throw std::runtime_error("CellShard text column exceeds u32 byte limit");
        }
        data.append(value.data(), value.size());
        offsets.push_back((std::uint32_t) data.size());
    }
    cs::dataset_text_column_view view() const {
        cs::dataset_text_column_view v{};
        v.count = offsets.empty() ? 0u : (std::uint32_t) offsets.size() - 1u;
        v.bytes = (std::uint32_t) data.size();
        v.offsets = offsets.data();
        v.data = data.data();
        return v;
    }
    void release() {
        std::vector<std::uint32_t>().swap(offsets);
        std::string().swap(data);
    }
};

struct matrix_header {
    std::uint64_t features = 0, cells = 0, nnz = 0;
};

struct feature_record {
    std::string id, name, type;
    std::vector<std::string> metadata;
};

struct dataset_plan {
    std::string id, matrix_path, barcode_path, feature_path, metadata_path;
    matrix_header header;
    std::vector<std::uint32_t> row_nnz;
    std::vector<std::uint64_t> part_begin, part_end, part_nnz;
    std::uint64_t global_row_begin = 0, global_part_begin = 0;
    bool row_cache_hit = false;
};

struct observation_columns {
    std::vector<std::vector<std::string>> values;
    std::vector<std::string> source_dataset;
    std::vector<char> seen;
};

struct fast_input {
    explicit fast_input(std::FILE *fp) : fp_(fp) {}
    bool read_u64(std::uint64_t *out) {
        int c = 0;
        do {
            c = get();
            if (c == EOF) return false;
        } while (c <= ' ');
        std::uint64_t value = 0;
        while (c > ' ') {
            value = value * 10u + (std::uint64_t) (c - '0');
            c = get();
        }
        *out = value;
        return true;
    }

private:
    int get() {
        if (pos_ == end_) {
            end_ = std::fread(buffer_, 1, sizeof(buffer_), fp_);
            pos_ = 0;
            if (end_ == 0) return EOF;
        }
        return (unsigned char) buffer_[pos_++];
    }

    std::FILE *fp_ = nullptr;
    char buffer_[1 << 20]{};
    std::size_t pos_ = 0, end_ = 0;
};

static std::vector<std::string> split_char(const std::string &s, char sep, bool keep_empty = false) {
    std::vector<std::string> out;
    std::string cur;
    for (char c : s) {
        if (c == sep) {
            if (keep_empty || !cur.empty()) out.push_back(cur);
            cur.clear();
        } else {
            cur.push_back(c);
        }
    }
    if (keep_empty || !cur.empty()) out.push_back(cur);
    return out;
}

static std::vector<std::string> parse_csv_line(const std::string &line) {
    std::vector<std::string> fields;
    std::string cur;
    bool quoted = false;
    for (std::size_t i = 0; i < line.size(); ++i) {
        const char c = line[i];
        if (quoted) {
            if (c == '"' && i + 1u < line.size() && line[i + 1u] == '"') {
                cur.push_back('"');
                ++i;
            } else if (c == '"') {
                quoted = false;
            } else {
                cur.push_back(c);
            }
        } else if (c == ',') {
            fields.push_back(cur);
            cur.clear();
        } else if (c == '"') {
            quoted = true;
        } else {
            cur.push_back(c);
        }
    }
    fields.push_back(cur);
    return fields;
}

static std::size_t column_index(const std::vector<std::string> &header, const std::string &name) {
    for (std::size_t i = 0; i < header.size(); ++i) {
        if (header[i] == name) return i;
    }
    throw std::runtime_error("missing metadata column: " + name);
}

static std::vector<std::string> load_lines(const std::string &path) {
    std::ifstream in(path);
    std::string line;
    std::vector<std::string> rows;
    if (!in) throw std::runtime_error("cannot open: " + path);
    while (std::getline(in, line)) {
        if (!line.empty() && line.back() == '\r') line.pop_back();
        if (!line.empty()) rows.push_back(line);
    }
    return rows;
}

static matrix_header read_matrix_header_file(const std::string &path) {
    std::ifstream in(path);
    std::string line;
    if (!std::getline(in, line) || line.rfind("%%MatrixMarket matrix coordinate integer", 0) != 0) {
        throw std::runtime_error("bad MatrixMarket banner: " + path);
    }
    while (std::getline(in, line)) {
        if (!line.empty() && line[0] == '%') continue;
        std::istringstream ss(line);
        matrix_header h;
        ss >> h.features >> h.cells >> h.nnz;
        if (!ss || h.features == 0 || h.cells == 0) throw std::runtime_error("bad MatrixMarket header: " + path);
        return h;
    }
    throw std::runtime_error("missing MatrixMarket dimensions: " + path);
}

static matrix_header skip_matrix_header(std::FILE *fp, const std::string &path) {
    char line[4096];
    if (!std::fgets(line, sizeof(line), fp) || std::strncmp(line, "%%MatrixMarket matrix coordinate integer", 40) != 0) {
        throw std::runtime_error("bad MatrixMarket banner: " + path);
    }
    while (std::fgets(line, sizeof(line), fp)) {
        if (line[0] == '%') continue;
        std::istringstream ss(line);
        matrix_header h;
        ss >> h.features >> h.cells >> h.nnz;
        if (!ss) throw std::runtime_error("bad MatrixMarket dimensions: " + path);
        return h;
    }
    throw std::runtime_error("missing MatrixMarket dimensions: " + path);
}

static int natural_dataset_suffix(const std::string &name) {
    const std::size_t pos = name.find_last_not_of("0123456789");
    if (pos == std::string::npos || pos + 1u >= name.size()) return -1;
    return std::atoi(name.c_str() + pos + 1u);
}

static std::vector<dataset_plan> discover_root_datasets(const options &opts) {
    std::vector<std::string> ids;
    std::vector<dataset_plan> plans;
    if (opts.root.empty()) return plans;
    if (!fs::exists(opts.root)) throw std::runtime_error("input root does not exist: " + opts.root);
    if (!opts.dataset_ids.empty()) {
        ids = opts.dataset_ids;
    } else {
        for (const auto &entry : fs::directory_iterator(opts.root)) {
            if (!entry.is_directory()) continue;
            fs::path base = entry.path();
            if (!opts.source_subdir.empty()) base /= opts.source_subdir;
            if (fs::exists(base / opts.matrix_filename)) ids.push_back(entry.path().filename().string());
        }
        std::sort(ids.begin(), ids.end(), [](const std::string &a, const std::string &b) {
            const int ai = natural_dataset_suffix(a);
            const int bi = natural_dataset_suffix(b);
            if (ai >= 0 && bi >= 0 && ai != bi) return ai < bi;
            return a < b;
        });
    }
    plans.reserve(ids.size());
    for (const std::string &id : ids) {
        fs::path base = fs::path(opts.root) / id;
        if (!opts.source_subdir.empty()) base /= opts.source_subdir;
        dataset_plan d;
        d.id = id;
        d.matrix_path = (base / opts.matrix_filename).string();
        d.barcode_path = (base / opts.barcode_filename).string();
        d.feature_path = (base / opts.feature_filename).string();
        d.metadata_path = opts.cell_metadata_path;
        plans.push_back(std::move(d));
    }
    return plans;
}

static std::vector<dataset_plan> load_manifest_datasets(const options &opts) {
    std::vector<dataset_plan> plans;
    if (opts.manifest_path.empty()) return plans;
    std::ifstream in(opts.manifest_path);
    std::string line;
    if (!in) throw std::runtime_error("cannot open manifest: " + opts.manifest_path);
    if (!std::getline(in, line)) return plans;
    if (!line.empty() && line.back() == '\r') line.pop_back();
    const auto header = split_char(line, '\t', true);
    const std::size_t id_i = column_index(header, "dataset_id");
    const std::size_t matrix_i = column_index(header, "matrix_path");
    const std::size_t feature_i = column_index(header, "feature_path");
    const std::size_t barcode_i = column_index(header, "barcode_path");
    std::size_t metadata_i = std::numeric_limits<std::size_t>::max();
    for (std::size_t i = 0; i < header.size(); ++i) {
        if (header[i] == "metadata_path") metadata_i = i;
    }
    while (std::getline(in, line)) {
        if (!line.empty() && line.back() == '\r') line.pop_back();
        if (line.empty()) continue;
        const auto fields = split_char(line, '\t', true);
        const std::size_t max_i = std::max({id_i, matrix_i, feature_i, barcode_i});
        if (fields.size() <= max_i) throw std::runtime_error("short manifest row: " + line);
        if (!opts.dataset_ids.empty()
            && std::find(opts.dataset_ids.begin(), opts.dataset_ids.end(), fields[id_i]) == opts.dataset_ids.end()) {
            continue;
        }
        dataset_plan d;
        d.id = fields[id_i];
        d.matrix_path = fields[matrix_i];
        d.feature_path = fields[feature_i];
        d.barcode_path = fields[barcode_i];
        d.metadata_path = metadata_i < fields.size() && !fields[metadata_i].empty() ? fields[metadata_i] : opts.cell_metadata_path;
        plans.push_back(std::move(d));
    }
    return plans;
}

static std::string row_cache_path(const dataset_plan &d, const std::string &cache_root) {
    struct stat st {};
    if (cache_root.empty()) return std::string();
    if (::stat(d.matrix_path.c_str(), &st) != 0) throw std::runtime_error("stat failed: " + d.matrix_path);
    std::ostringstream os;
    os << d.id << "."
       << (unsigned long long) std::hash<std::string>{}(d.matrix_path) << "."
       << (unsigned long long) st.st_size << "."
       << (long long) st.st_mtim.tv_sec << "."
       << (long long) st.st_mtim.tv_nsec << "."
       << d.header.features << "."
       << d.header.cells << "."
       << d.header.nnz << ".features_by_cells.row_nnz.u32";
    return (fs::path(cache_root) / "row_nnz" / os.str()).string();
}

static bool load_row_cache(const dataset_plan &d, const std::string &cache_root, std::vector<std::uint32_t> *out) {
    const std::string path = row_cache_path(d, cache_root);
    if (path.empty()) return false;
    std::ifstream in(path, std::ios::binary);
    char magic[8]{};
    std::uint64_t features = 0, cells = 0, nnz = 0;
    if (!in) return false;
    in.read(magic, sizeof(magic));
    in.read(reinterpret_cast<char *>(&features), sizeof(features));
    in.read(reinterpret_cast<char *>(&cells), sizeof(cells));
    in.read(reinterpret_cast<char *>(&nnz), sizeof(nnz));
    if (std::memcmp(magic, "RNNZV003", 8) != 0 || features != d.header.features || cells != d.header.cells || nnz != d.header.nnz) return false;
    out->assign((std::size_t) d.header.cells, 0u);
    in.read(reinterpret_cast<char *>(out->data()), (std::streamsize) (out->size() * sizeof(std::uint32_t)));
    return (bool) in;
}

static void store_row_cache(const dataset_plan &d, const std::string &cache_root, const std::vector<std::uint32_t> &counts) {
    const std::string path = row_cache_path(d, cache_root);
    if (path.empty()) return;
    fs::create_directories(fs::path(path).parent_path());
    std::ofstream out(path, std::ios::binary | std::ios::trunc);
    if (!out) throw std::runtime_error("cannot write row-nnz cache: " + path);
    const char magic[8] = {'R','N','N','Z','V','0','0','3'};
    out.write(magic, sizeof(magic));
    out.write(reinterpret_cast<const char *>(&d.header.features), sizeof(d.header.features));
    out.write(reinterpret_cast<const char *>(&d.header.cells), sizeof(d.header.cells));
    out.write(reinterpret_cast<const char *>(&d.header.nnz), sizeof(d.header.nnz));
    out.write(reinterpret_cast<const char *>(counts.data()), (std::streamsize) (counts.size() * sizeof(std::uint32_t)));
}

static std::vector<std::uint32_t> scan_transposed_row_counts(dataset_plan *d, const options &opts, bool *hit_out) {
    std::vector<std::uint32_t> counts;
    if (load_row_cache(*d, opts.cache_root, &counts)) {
        *hit_out = true;
        return counts;
    }
    *hit_out = false;
    std::FILE *fp = std::fopen(d->matrix_path.c_str(), "rb");
    if (!fp) throw std::runtime_error("cannot open matrix: " + d->matrix_path);
    const matrix_header h = skip_matrix_header(fp, d->matrix_path);
    if (h.features != d->header.features || h.cells != d->header.cells || h.nnz != d->header.nnz) {
        std::fclose(fp);
        throw std::runtime_error(d->id + ": header mismatch during row-count scan");
    }
    counts.assign((std::size_t) d->header.cells, 0u);
    fast_input input(fp);
    std::uint64_t prev_cell = 0;
    for (std::uint64_t i = 0; i < d->header.nnz; ++i) {
        std::uint64_t feature = 0, cell = 0, value = 0;
        if (!input.read_u64(&feature) || !input.read_u64(&cell) || !input.read_u64(&value)) {
            std::fclose(fp);
            throw std::runtime_error(d->id + ": truncated MatrixMarket stream");
        }
        if (feature == 0u || feature > d->header.features || cell == 0u || cell > d->header.cells || cell < prev_cell) {
            std::fclose(fp);
            throw std::runtime_error(d->id + ": invalid or non-cell-sorted MatrixMarket entry");
        }
        prev_cell = cell;
        ++counts[(std::size_t) cell - 1u];
    }
    std::fclose(fp);
    store_row_cache(*d, opts.cache_root, counts);
    return counts;
}

static void plan_partitions(dataset_plan *d, std::uint64_t max_part_nnz) {
    std::uint64_t acc = 0, start = 0;
    d->part_begin.clear();
    d->part_end.clear();
    d->part_nnz.clear();
    for (std::uint64_t row = 0; row < d->header.cells; ++row) {
        const std::uint64_t next = d->row_nnz[(std::size_t) row];
        if (row > start && acc != 0u && max_part_nnz != 0u && acc + next > max_part_nnz) {
            d->part_begin.push_back(start);
            d->part_end.push_back(row);
            d->part_nnz.push_back(acc);
            start = row;
            acc = 0;
        }
        acc += next;
    }
    d->part_begin.push_back(start);
    d->part_end.push_back(d->header.cells);
    d->part_nnz.push_back(acc);
}

static unsigned int effective_cpu_workers(const options &opts, std::size_t work_items) {
    if (work_items == 0u) return 1u;
    unsigned int workers = opts.cpu_workers > 0
        ? (unsigned int) opts.cpu_workers
        : std::max(1u, std::thread::hardware_concurrency() > 1u ? std::thread::hardware_concurrency() - 1u : 1u);
    return std::max(1u, std::min<unsigned int>(workers, (unsigned int) work_items));
}

static std::vector<int> effective_gpu_ids(const options &opts) {
    if (!opts.gpu_ids.empty()) return opts.gpu_ids;
    int device_count = 0;
    std::vector<int> ids;
    if (cudaGetDeviceCount(&device_count) != cudaSuccess || device_count <= 0) {
        cudaGetLastError();
        ids.push_back(0);
        return ids;
    }
    ids.reserve((std::size_t) device_count);
    for (int i = 0; i < device_count; ++i) ids.push_back(i);
    return ids;
}

static std::vector<dataset_plan> load_and_plan_datasets(const options &opts, stats *st) {
    if (opts.orientation != matrix_orientation::features_by_cells) {
        throw std::runtime_error("only features_by_cells MatrixMarket series are supported");
    }
    timer t;
    std::vector<dataset_plan> datasets = !opts.manifest_path.empty() ? load_manifest_datasets(opts) : discover_root_datasets(opts);
    if (datasets.empty()) throw std::runtime_error("no MatrixMarket series datasets found");

    for (dataset_plan &d : datasets) {
        d.header = read_matrix_header_file(d.matrix_path);
        const auto barcodes = load_lines(d.barcode_path);
        if (barcodes.size() != d.header.cells) throw std::runtime_error(d.id + ": barcode count/header column mismatch");
        if (!fs::exists(d.feature_path)) throw std::runtime_error(d.id + ": missing feature file");
    }

    std::atomic<std::size_t> next{0u};
    std::mutex error_mutex, log_mutex;
    std::exception_ptr error;
    const unsigned int workers = effective_cpu_workers(opts, datasets.size());
    std::vector<std::thread> threads;
    for (unsigned int worker = 0; worker < workers; ++worker) {
        threads.emplace_back([&, worker]() {
            while (true) {
                const std::size_t i = next.fetch_add(1u);
                if (i >= datasets.size()) break;
                try {
                    timer part_timer;
                    bool cache_hit = false;
                    datasets[i].row_nnz = scan_transposed_row_counts(&datasets[i], opts, &cache_hit);
                    datasets[i].row_cache_hit = cache_hit;
                    plan_partitions(&datasets[i], opts.max_part_nnz);
                    {
                        std::lock_guard<std::mutex> lock(log_mutex);
                        std::cerr << "planned " << datasets[i].id
                                  << " cells=" << datasets[i].header.cells
                                  << " nnz=" << datasets[i].header.nnz
                                  << " parts=" << datasets[i].part_nnz.size()
                                  << " row_cache_hit=" << (cache_hit ? 1 : 0)
                                  << " worker=" << worker
                                  << " seconds=" << part_timer.seconds() << "\n";
                    }
                } catch (...) {
                    std::lock_guard<std::mutex> lock(error_mutex);
                    if (!error) error = std::current_exception();
                }
            }
        });
    }
    for (std::thread &thread : threads) thread.join();
    if (error) std::rethrow_exception(error);

    std::uint64_t global_row = 0, global_part = 0;
    for (dataset_plan &d : datasets) {
        d.global_row_begin = global_row;
        d.global_part_begin = global_part;
        global_row += d.header.cells;
        global_part += d.part_nnz.size();
        if (st != nullptr) {
            st->row_cache_hits += d.row_cache_hit ? 1u : 0u;
            st->row_cache_misses += d.row_cache_hit ? 0u : 1u;
        }
    }
    if (st != nullptr) st->planning_seconds = t.seconds();
    return datasets;
}

static std::vector<feature_record> load_features_with_metadata(const std::vector<dataset_plan> &datasets,
                                                               const options &opts) {
    if (datasets.empty()) throw std::runtime_error("no datasets for feature metadata");
    const auto first_lines = load_lines(datasets.front().feature_path);
    std::vector<feature_record> features;
    features.reserve(first_lines.size());
    for (const std::string &line : first_lines) {
        const auto fields = split_char(line, '\t', true);
        feature_record f;
        f.id = fields.empty() ? line : fields[0];
        f.name = fields.size() > 1u && !fields[1].empty() ? fields[1] : f.id;
        f.type = fields.size() > 2u ? fields[2] : "";
        f.metadata.assign(opts.feature_columns.size(), "");
        features.push_back(std::move(f));
    }
    if (features.size() != datasets.front().header.features) throw std::runtime_error("feature count mismatch in first dataset");
    for (std::size_t i = 1; i < datasets.size(); ++i) {
        const auto other = load_lines(datasets[i].feature_path);
        if (other.size() != features.size()) throw std::runtime_error(datasets[i].id + ": feature count differs");
        for (std::size_t j = 0; j < other.size(); ++j) {
            const auto fields = split_char(other[j], '\t', true);
            const std::string id = fields.empty() ? other[j] : fields[0];
            if (id != features[j].id) throw std::runtime_error(datasets[i].id + ": feature order differs at row " + std::to_string(j));
        }
    }
    if (opts.feature_metadata_path.empty() || opts.feature_columns.empty()) return features;

    std::ifstream in(opts.feature_metadata_path);
    std::string line;
    if (!in) {
        if (opts.allow_missing_metadata) return features;
        throw std::runtime_error("cannot open feature metadata: " + opts.feature_metadata_path);
    }
    std::getline(in, line);
    const auto header = parse_csv_line(line);
    const std::size_t id_i = column_index(header, opts.feature_id_column);
    std::vector<std::size_t> col_i;
    col_i.reserve(opts.feature_columns.size());
    for (const std::string &name : opts.feature_columns) col_i.push_back(column_index(header, name));
    const std::size_t max_i = *std::max_element(col_i.begin(), col_i.end());
    std::unordered_map<std::string, std::vector<std::string>> by_id;
    while (std::getline(in, line)) {
        const auto fields = parse_csv_line(line);
        if (fields.size() <= std::max(id_i, max_i)) continue;
        std::vector<std::string> values;
        values.reserve(col_i.size());
        for (std::size_t idx : col_i) values.push_back(fields[idx]);
        by_id[fields[id_i]] = std::move(values);
    }
    std::uint64_t missing = 0u;
    for (feature_record &f : features) {
        const auto hit = by_id.find(f.id);
        if (hit == by_id.end()) {
            ++missing;
            continue;
        }
        f.metadata = hit->second;
    }
    std::cerr << "loaded feature metadata rows=" << features.size() << " missing_feature_metadata=" << missing << "\n";
    return features;
}

static void append_string_vector(text_column *col, const std::vector<std::string> &values) {
    std::size_t bytes = 0;
    for (const std::string &v : values) bytes += v.size();
    col->reserve(values.size(), bytes);
    for (const std::string &v : values) col->append(v);
}

static observation_columns build_observation_metadata(const options &opts,
                                                      const std::vector<dataset_plan> &datasets,
                                                      text_column *global_barcodes,
                                                      std::vector<std::uint32_t> *cell_dataset_ids,
                                                      std::vector<std::uint64_t> *cell_local_indices) {
    std::unordered_map<std::string, std::uint64_t> row_by_barcode;
    std::uint64_t total_rows = 0;
    for (const auto &d : datasets) total_rows += d.header.cells;
    row_by_barcode.reserve((std::size_t) total_rows * 13u / 10u);
    global_barcodes->reserve((std::size_t) total_rows, (std::size_t) total_rows * 34u);
    cell_dataset_ids->reserve((std::size_t) total_rows);
    cell_local_indices->reserve((std::size_t) total_rows);

    observation_columns out;
    out.values.assign(opts.observation_columns.size(), std::vector<std::string>((std::size_t) total_rows));
    out.source_dataset.resize((std::size_t) total_rows);
    out.seen.assign((std::size_t) total_rows, opts.cell_metadata_path.empty() || opts.observation_columns.empty() ? 1 : 0);

    for (std::size_t dataset_i = 0; dataset_i < datasets.size(); ++dataset_i) {
        const auto barcodes = load_lines(datasets[dataset_i].barcode_path);
        if (barcodes.size() != datasets[dataset_i].header.cells) throw std::runtime_error(datasets[dataset_i].id + ": barcode count changed");
        for (std::size_t local = 0; local < barcodes.size(); ++local) {
            const std::uint64_t global = datasets[dataset_i].global_row_begin + local;
            if (!row_by_barcode.emplace(barcodes[local], global).second) {
                throw std::runtime_error("duplicate barcode/cell_id: " + barcodes[local]);
            }
            global_barcodes->append(barcodes[local]);
            cell_dataset_ids->push_back((std::uint32_t) dataset_i);
            cell_local_indices->push_back((std::uint64_t) local);
            out.source_dataset[(std::size_t) global] = datasets[dataset_i].id;
        }
    }
    if (opts.cell_metadata_path.empty() || opts.observation_columns.empty()) return out;

    std::ifstream in(opts.cell_metadata_path);
    std::string line;
    if (!in) {
        if (opts.allow_missing_metadata) return out;
        throw std::runtime_error("cannot open cell metadata: " + opts.cell_metadata_path);
    }
    std::getline(in, line);
    const auto header = parse_csv_line(line);
    const std::size_t cell_i = column_index(header, opts.cell_id_column);
    std::vector<std::size_t> col_i;
    col_i.reserve(opts.observation_columns.size());
    for (const std::string &name : opts.observation_columns) col_i.push_back(column_index(header, name));
    const std::size_t max_i = std::max(cell_i, *std::max_element(col_i.begin(), col_i.end()));

    const bool use_prefilter = !opts.observation_prefilter_column.empty() && datasets.size() < 64u;
    std::uint64_t scanned = 0, matched = 0;
    timer t;
    while (std::getline(in, line)) {
        ++scanned;
        if (use_prefilter) {
            bool hit_dataset = false;
            for (const auto &d : datasets) {
                if (line.find(d.id) != std::string::npos) {
                    hit_dataset = true;
                    break;
                }
            }
            if (!hit_dataset) continue;
        }
        const auto fields = parse_csv_line(line);
        if (fields.size() <= max_i) throw std::runtime_error("short cell metadata row");
        const auto hit = row_by_barcode.find(fields[cell_i]);
        if (hit == row_by_barcode.end()) continue;
        const std::size_t row = (std::size_t) hit->second;
        if (out.seen[row] != 0) throw std::runtime_error("duplicate metadata row for selected cell_id: " + fields[cell_i]);
        for (std::size_t col = 0; col < col_i.size(); ++col) out.values[col][row] = fields[col_i[col]];
        out.seen[row] = 1;
        ++matched;
    }
    if (matched != total_rows && !opts.allow_missing_metadata) {
        throw std::runtime_error("cell metadata coverage mismatch: matched " + std::to_string(matched) + " of " + std::to_string(total_rows));
    }
    std::cerr << "loaded observation metadata scanned=" << scanned
              << " matched=" << matched
              << " seconds=" << t.seconds() << "\n";
    return out;
}

static std::uint64_t sliced_total_slots(const std::vector<std::uint32_t> &row_nnz,
                                        std::uint64_t begin,
                                        std::uint64_t end,
                                        std::uint32_t slice_rows,
                                        std::vector<std::uint32_t> *offsets,
                                        std::vector<std::uint32_t> *widths) {
    offsets->clear();
    widths->clear();
    offsets->push_back(0u);
    std::uint64_t slots = 0;
    for (std::uint64_t s = begin; s < end; s += slice_rows) {
        const std::uint64_t e = std::min<std::uint64_t>(end, s + slice_rows);
        std::uint32_t width = 0;
        for (std::uint64_t r = s; r < e; ++r) width = std::max(width, row_nnz[(std::size_t) r]);
        widths->push_back(width);
        offsets->push_back((std::uint32_t) (e - begin));
        slots += (e - s) * (std::uint64_t) width;
    }
    return slots;
}

static std::uint64_t sliced_bytes_from_aux(std::uint32_t slice_count, std::uint64_t slots) {
    const std::uint64_t offsets_bytes = (std::uint64_t) (slice_count + 1u) * sizeof(std::uint32_t);
    const std::uint64_t widths_bytes = (std::uint64_t) slice_count * sizeof(std::uint32_t);
    const std::uint64_t col_bytes = slots * sizeof(cs::types::idx_t);
    const std::uint64_t val_bytes = slots * sizeof(real::storage_t);
    return sizeof(cs::sparse::sliced_ell) + offsets_bytes + widths_bytes + col_bytes + val_bytes;
}

static void allocate_sliced_part(cs::sparse::sliced_ell *sliced,
                                 const dataset_plan &d,
                                 std::uint64_t local_part,
                                 std::uint32_t slice_rows,
                                 std::uint64_t feature_count,
                                 std::vector<std::uint32_t> *offsets,
                                 std::vector<std::uint32_t> *widths,
                                 std::vector<std::uint64_t> *slice_bases,
                                 std::vector<std::uint32_t> *write_counts) {
    const std::uint64_t begin = d.part_begin[(std::size_t) local_part];
    const std::uint64_t end = d.part_end[(std::size_t) local_part];
    const std::uint64_t rows = end - begin;
    if (rows > std::numeric_limits<std::uint32_t>::max() || feature_count > std::numeric_limits<std::uint32_t>::max()) {
        throw std::runtime_error(d.id + ": partition exceeds CellShard u32 limits");
    }
    sliced_total_slots(d.row_nnz, begin, end, slice_rows, offsets, widths);
    cs::sparse::init(sliced, (cs::types::dim_t) rows, (cs::types::dim_t) feature_count, (cs::types::nnz_t) d.part_nnz[(std::size_t) local_part]);
    if (!cs::sparse::allocate(sliced, (cs::types::u32) widths->size(), offsets->data(), widths->data())) {
        throw std::runtime_error(d.id + ": failed to allocate sliced part");
    }
    slice_bases->assign(widths->size(), 0u);
    std::uint64_t base = 0;
    for (std::size_t i = 0; i < widths->size(); ++i) {
        (*slice_bases)[i] = base;
        base += (std::uint64_t) ((*offsets)[i + 1u] - (*offsets)[i]) * (*widths)[i];
    }
    write_counts->assign((std::size_t) rows, 0u);
}

static void append_dataset_parts_streaming(const options &opts,
                                           const dataset_plan &d,
                                           std::uint64_t feature_count,
                                           const std::vector<int> &gpu_ids,
                                           unsigned int worker_id,
                                           std::mutex *write_mutex,
                                           std::vector<std::uint32_t> *part_bucket_counts,
                                           std::vector<std::uint64_t> *part_bucketed_bytes,
                                           std::vector<std::uint64_t> *part_execution_bytes,
                                           std::uint64_t *canonical_bytes_sum,
                                           std::uint64_t *bucketed_bytes_sum) {
    if (!gpu_ids.empty()) (void) cudaSetDevice(gpu_ids[worker_id % gpu_ids.size()]);
    cs::sparse::sliced_ell sliced;
    std::vector<std::uint32_t> offsets, widths, write_counts;
    std::vector<std::uint64_t> slice_bases;
    std::uint64_t local_part = 0, written_in_part = 0;
    timer part_timer;

    auto allocate_current = [&]() {
        cs::sparse::init(&sliced);
        allocate_sliced_part(&sliced, d, local_part, opts.slice_rows, feature_count, &offsets, &widths, &slice_bases, &write_counts);
        written_in_part = 0;
        part_timer = timer{};
    };
    auto finalize_current = [&]() {
        const std::uint64_t expected = d.part_nnz[(std::size_t) local_part];
        const std::uint64_t global_part = d.global_part_begin + local_part;
        if (written_in_part != expected) {
            cs::sparse::clear(&sliced);
            throw std::runtime_error(d.id + ": partition fill nnz mismatch");
        }
        std::uint64_t bucketed_bytes = 0;
        std::uint32_t bucket_count = 1;
        cs::bucketed_sliced_ell_partition bucketed;
        cs::init(&bucketed);
        const std::uint64_t canonical_bytes = (std::uint64_t) cs::sparse::bytes(&sliced);
        if (!cs::choose_bucket_count_for_sliced_ell_partition(&sliced, &bucket_count, &bucketed_bytes)
            || !cs::build_bucketed_sliced_ell_partition(&bucketed, &sliced, bucket_count, &bucketed_bytes)) {
            cs::clear(&bucketed);
            cs::sparse::clear(&sliced);
            throw std::runtime_error(d.id + ": optimized sliced partition build failed");
        }
        {
            std::lock_guard<std::mutex> lock(*write_mutex);
            if (!cs::append_sliced_ell_partition_h5(opts.output_path.c_str(), (unsigned long) global_part, &bucketed)) {
                cs::clear(&bucketed);
                cs::sparse::clear(&sliced);
                throw std::runtime_error(d.id + ": optimized sliced partition append failed");
            }
            std::cerr << "appended part=" << global_part
                      << " dataset=" << d.id
                      << " local_part=" << local_part
                      << " worker=" << worker_id
                      << " gpu=" << (!gpu_ids.empty() ? gpu_ids[worker_id % gpu_ids.size()] : 0)
                      << " rows=" << sliced.rows
                      << " nnz=" << sliced.nnz
                      << " slots=" << cs::sparse::total_slots(&sliced)
                      << " chosen_buckets=" << bucket_count
                      << " segments=" << bucketed.segment_count
                      << " canonical_bytes=" << canonical_bytes
                      << " bucketed_bytes=" << bucketed_bytes
                      << " bucketed_to_canonical=" << (canonical_bytes != 0 ? (double) bucketed_bytes / (double) canonical_bytes : 0.0)
                      << " seconds=" << part_timer.seconds() << "\n";
        }
        (*part_bucket_counts)[(std::size_t) global_part] = bucketed.segment_count;
        (*part_bucketed_bytes)[(std::size_t) global_part] = bucketed_bytes;
        (*part_execution_bytes)[(std::size_t) global_part] = bucketed_bytes;
        *canonical_bytes_sum += canonical_bytes;
        *bucketed_bytes_sum += bucketed_bytes;
        cs::clear(&bucketed);
        cs::sparse::clear(&sliced);
        ++local_part;
    };

    if (d.part_nnz.empty()) return;
    allocate_current();
    std::FILE *fp = std::fopen(d.matrix_path.c_str(), "rb");
    if (!fp) throw std::runtime_error("cannot open matrix: " + d.matrix_path);
    (void) skip_matrix_header(fp, d.matrix_path);
    fast_input input(fp);
    for (std::uint64_t i = 0; i < d.header.nnz; ++i) {
        std::uint64_t feature = 0, cell = 0, value = 0;
        if (!input.read_u64(&feature) || !input.read_u64(&cell) || !input.read_u64(&value)) {
            std::fclose(fp);
            cs::sparse::clear(&sliced);
            throw std::runtime_error(d.id + ": truncated stream during fill");
        }
        const std::uint64_t local_cell = cell - 1u;
        while (local_part < d.part_nnz.size() && local_cell >= d.part_end[(std::size_t) local_part]) {
            finalize_current();
            if (local_part < d.part_nnz.size()) allocate_current();
        }
        if (local_part >= d.part_nnz.size()) {
            std::fclose(fp);
            throw std::runtime_error(d.id + ": entry exceeded planned rows");
        }
        const std::uint64_t begin = d.part_begin[(std::size_t) local_part];
        const std::uint32_t row = (std::uint32_t) (local_cell - begin);
        const std::uint32_t slice = row / opts.slice_rows;
        const std::uint32_t slice_row_begin = offsets[(std::size_t) slice];
        const std::uint32_t row_in_slice = row - slice_row_begin;
        const std::uint32_t slot = write_counts[(std::size_t) row]++;
        if (slot >= widths[(std::size_t) slice]) {
            std::fclose(fp);
            cs::sparse::clear(&sliced);
            throw std::runtime_error(d.id + ": row write exceeded planned slice width");
        }
        const std::uint64_t idx = slice_bases[(std::size_t) slice] + (std::uint64_t) row_in_slice * widths[(std::size_t) slice] + slot;
        sliced.col_idx[idx] = (cs::types::idx_t) (feature - 1u);
        sliced.val[idx] = __float2half((float) value);
        ++written_in_part;
    }
    std::fclose(fp);
    while (local_part < d.part_nnz.size()) {
        finalize_current();
        if (local_part < d.part_nnz.size()) allocate_current();
    }
}

static void append_feature_metadata(const options &opts,
                                    const std::vector<feature_record> &features) {
    if (opts.feature_columns.empty()) return;
    std::vector<text_column> columns(opts.feature_columns.size());
    std::vector<cs::dataset_observation_metadata_column_view> views(opts.feature_columns.size());
    for (std::size_t col = 0; col < opts.feature_columns.size(); ++col) {
        std::vector<std::string> values;
        values.reserve(features.size());
        for (const feature_record &f : features) values.push_back(col < f.metadata.size() ? f.metadata[col] : "");
        append_string_vector(&columns[col], values);
        views[col] = {opts.feature_columns[col].c_str(), cs::dataset_observation_metadata_type_text, columns[col].view(), nullptr, nullptr};
    }
    const cs::dataset_feature_metadata_view feature_metadata{(std::uint64_t) features.size(), (std::uint32_t) views.size(), views.data()};
    if (!cs::append_dataset_feature_metadata_h5(opts.output_path.c_str(), &feature_metadata)) {
        throw std::runtime_error("append_dataset_feature_metadata_h5 failed");
    }
}

static void run_payload_workers(const options &opts,
                                const std::vector<dataset_plan> &datasets,
                                std::uint64_t feature_count,
                                std::vector<std::uint32_t> *part_bucket_counts,
                                std::vector<std::uint64_t> *part_bucketed_bytes,
                                std::vector<std::uint64_t> *part_execution_bytes,
                                stats *st) {
    timer t;
    const auto gpus = effective_gpu_ids(opts);
    const unsigned int workers = effective_cpu_workers(opts, datasets.size());
    std::atomic<std::size_t> next{0u};
    std::mutex write_mutex, error_mutex;
    std::exception_ptr error;
    std::vector<std::uint64_t> canonical_by_worker(workers, 0u), bucketed_by_worker(workers, 0u);
    std::vector<std::thread> threads;
    std::cerr << "starting payload workers cpu_workers=" << workers << " gpu_count=" << gpus.size() << "\n";
    for (unsigned int worker = 0; worker < workers; ++worker) {
        threads.emplace_back([&, worker]() {
            while (true) {
                const std::size_t i = next.fetch_add(1u);
                if (i >= datasets.size()) break;
                try {
                    append_dataset_parts_streaming(opts,
                                                   datasets[i],
                                                   feature_count,
                                                   gpus,
                                                   worker,
                                                   &write_mutex,
                                                   part_bucket_counts,
                                                   part_bucketed_bytes,
                                                   part_execution_bytes,
                                                   &canonical_by_worker[worker],
                                                   &bucketed_by_worker[worker]);
                } catch (...) {
                    std::lock_guard<std::mutex> lock(error_mutex);
                    if (!error) error = std::current_exception();
                }
            }
        });
    }
    for (std::thread &thread : threads) thread.join();
    if (error) std::rethrow_exception(error);
    if (st != nullptr) {
        st->payload_seconds = t.seconds();
        for (std::uint64_t value : canonical_by_worker) st->canonical_sliced_bytes += value;
        for (std::uint64_t value : bucketed_by_worker) st->bucketed_sliced_bytes += value;
    }
}

static void maybe_move_tmp_output(const std::string &tmp, const std::string &final_path) {
    if (tmp.empty() || tmp == final_path) return;
    std::error_code ec;
    fs::rename(tmp, final_path, ec);
    if (ec) {
        fs::remove(final_path, ec);
        ec.clear();
        fs::rename(tmp, final_path, ec);
    }
    if (ec) throw std::runtime_error("failed to move temporary output to final path: " + ec.message());
}

} // namespace

int convert_to_optimized_sliced_ell_csh5(const options &input_opts, stats *out_stats) {
    options opts = input_opts;
    stats local_stats{};
    stats *st = out_stats != nullptr ? out_stats : &local_stats;
    *st = stats{};
    timer total_timer;

    const std::string final_output = opts.output_path;
    if (!opts.prewarm_only && opts.output_path.empty()) throw std::runtime_error("output_path is required");
    if (!opts.prewarm_only && opts.atomic_output) {
        opts.output_path = !opts.tmp_output_path.empty() ? opts.tmp_output_path : final_output + ".tmp";
    }
    if (opts.slice_rows == 0u) throw std::runtime_error("slice_rows must be positive");
    if (opts.max_part_nnz == 0u) throw std::runtime_error("max_part_nnz must be positive");
    if (!opts.prewarm_only) {
        const fs::path parent = fs::path(opts.output_path).parent_path();
        if (!parent.empty()) fs::create_directories(parent);
        if (opts.output_path != final_output) fs::remove(opts.output_path);
    }

    auto datasets = load_and_plan_datasets(opts, st);
    auto features = load_features_with_metadata(datasets, opts);
    const std::uint64_t feature_count = features.size();
    std::uint64_t total_rows = 0, total_nnz = 0, part_count = 0;
    for (const auto &d : datasets) {
        if (d.header.features != feature_count) throw std::runtime_error(d.id + ": feature count mismatch");
        total_rows += d.header.cells;
        total_nnz += d.header.nnz;
        part_count += d.part_nnz.size();
    }
    st->datasets = datasets.size();
    st->rows = total_rows;
    st->cols = feature_count;
    st->nnz = total_nnz;
    st->partitions = part_count;
    if (opts.prewarm_only) {
        st->total_seconds = total_timer.seconds();
        std::cerr << "prewarm complete datasets=" << st->datasets << " rows=" << total_rows << " nnz=" << total_nnz << "\n";
        return 1;
    }

    timer metadata_timer;
    text_column dataset_ids, matrix_paths, feature_paths, barcode_paths, metadata_paths;
    text_column global_barcodes, feature_ids, feature_names, feature_types;
    std::vector<std::uint32_t> dataset_formats, cell_dataset_ids, feature_dataset_ids;
    std::vector<std::uint64_t> dataset_row_begin, dataset_row_end, dataset_rows, dataset_cols, dataset_nnz;
    std::vector<std::uint64_t> cell_local_indices, feature_local_indices, dataset_feature_offsets;
    std::vector<std::uint32_t> dataset_feature_to_global;
    std::vector<std::uint64_t> part_rows, part_nnz, part_aux, part_row_offsets, shard_offsets;
    std::vector<std::uint32_t> part_axes, part_dataset_ids, part_codec_ids;
    std::vector<std::uint32_t> part_slice_counts, part_slice_rows, part_formats, part_bucket_counts;
    std::vector<std::uint64_t> part_sliced_bytes, part_bucketed_bytes, part_execution_bytes;

    part_row_offsets.push_back(0u);
    for (std::size_t dataset_i = 0; dataset_i < datasets.size(); ++dataset_i) {
        const auto &d = datasets[dataset_i];
        dataset_ids.append(d.id);
        matrix_paths.append(d.matrix_path);
        feature_paths.append(d.feature_path);
        barcode_paths.append(d.barcode_path);
        metadata_paths.append(d.metadata_path);
        dataset_formats.push_back(1u);
        dataset_row_begin.push_back(d.global_row_begin);
        dataset_row_end.push_back(d.global_row_begin + d.header.cells);
        dataset_rows.push_back(d.header.cells);
        dataset_cols.push_back(feature_count);
        dataset_nnz.push_back(d.header.nnz);
        for (std::uint64_t p = 0; p < d.part_nnz.size(); ++p) {
            std::vector<std::uint32_t> offsets, widths;
            const std::uint64_t slots = sliced_total_slots(d.row_nnz, d.part_begin[(std::size_t) p], d.part_end[(std::size_t) p], opts.slice_rows, &offsets, &widths);
            if (slots > std::numeric_limits<std::uint32_t>::max()) throw std::runtime_error(d.id + ": planned slots exceed u32");
            const std::uint64_t rows = d.part_end[(std::size_t) p] - d.part_begin[(std::size_t) p];
            part_rows.push_back(rows);
            part_nnz.push_back(d.part_nnz[(std::size_t) p]);
            part_axes.push_back(0u);
            part_aux.push_back(cs::sparse::pack_sliced_ell_aux((cs::types::u32) widths.size(), (cs::types::u32) slots));
            part_row_offsets.push_back(part_row_offsets.back() + rows);
            part_dataset_ids.push_back((std::uint32_t) dataset_i);
            part_codec_ids.push_back(0u);
            part_slice_counts.push_back((std::uint32_t) widths.size());
            part_slice_rows.push_back(opts.slice_rows);
            part_formats.push_back(cs::dataset_execution_format_bucketed_sliced_ell);
            part_bucket_counts.push_back(0u);
            part_sliced_bytes.push_back(sliced_bytes_from_aux((std::uint32_t) widths.size(), slots));
            part_bucketed_bytes.push_back(0u);
            part_execution_bytes.push_back(0u);
        }
    }

    for (std::size_t i = 0; i < features.size(); ++i) {
        feature_ids.append(features[i].id);
        feature_names.append(features[i].name.empty() ? features[i].id : features[i].name);
        feature_types.append(features[i].type);
        feature_dataset_ids.push_back(0u);
        feature_local_indices.push_back((std::uint64_t) i);
    }
    dataset_feature_offsets.push_back(0u);
    for (std::size_t dataset_i = 0; dataset_i < datasets.size(); ++dataset_i) {
        for (std::size_t feature_i = 0; feature_i < features.size(); ++feature_i) dataset_feature_to_global.push_back((std::uint32_t) feature_i);
        dataset_feature_offsets.push_back((std::uint64_t) dataset_feature_to_global.size());
    }

    auto obs = build_observation_metadata(opts, datasets, &global_barcodes, &cell_dataset_ids, &cell_local_indices);

    shard_offsets.push_back(0u);
    std::vector<std::uint32_t> shard_part_begin{0u}, shard_part_end;
    std::uint64_t shard_bytes = 0;
    for (std::uint32_t p = 0; p < part_rows.size(); ++p) {
        const bool should_cut = p > shard_part_begin.back() && shard_bytes != 0u && shard_bytes + part_sliced_bytes[p] > opts.target_shard_bytes;
        if (should_cut) {
            shard_part_end.push_back(p);
            shard_offsets.push_back(part_row_offsets[p]);
            shard_part_begin.push_back(p);
            shard_bytes = 0;
        }
        shard_bytes += part_sliced_bytes[p];
    }
    shard_part_end.push_back((std::uint32_t) part_rows.size());
    shard_offsets.push_back(total_rows);
    st->shards = shard_part_begin.size();

    cs::dataset_codec_descriptor codec{};
    codec.codec_id = 0u;
    codec.family = cs::dataset_codec_family_sliced_ell;
    codec.value_code = (std::uint32_t) real::code_of<real::storage_t>::code;
    codec.bits = (std::uint32_t) (sizeof(real::storage_t) * 8u);

    cs::dataset_layout_view layout{};
    layout.rows = total_rows;
    layout.cols = feature_count;
    layout.nnz = total_nnz;
    layout.num_partitions = part_rows.size();
    layout.num_shards = shard_part_begin.size();
    layout.partition_rows = part_rows.data();
    layout.partition_nnz = part_nnz.data();
    layout.partition_axes = part_axes.data();
    layout.partition_aux = part_aux.data();
    layout.partition_row_offsets = part_row_offsets.data();
    layout.partition_dataset_ids = part_dataset_ids.data();
    layout.partition_codec_ids = part_codec_ids.data();
    layout.shard_offsets = shard_offsets.data();
    layout.codecs = &codec;
    layout.num_codecs = 1u;

    cs::dataset_dataset_table_view dataset_view{};
    dataset_view.count = (std::uint32_t) datasets.size();
    dataset_view.dataset_ids = dataset_ids.view();
    dataset_view.matrix_paths = matrix_paths.view();
    dataset_view.feature_paths = feature_paths.view();
    dataset_view.barcode_paths = barcode_paths.view();
    dataset_view.metadata_paths = metadata_paths.view();
    dataset_view.formats = dataset_formats.data();
    dataset_view.row_begin = dataset_row_begin.data();
    dataset_view.row_end = dataset_row_end.data();
    dataset_view.rows = dataset_rows.data();
    dataset_view.cols = dataset_cols.data();
    dataset_view.nnz = dataset_nnz.data();

    cs::dataset_provenance_view provenance{};
    provenance.global_barcodes = global_barcodes.view();
    provenance.cell_dataset_ids = cell_dataset_ids.data();
    provenance.cell_local_indices = cell_local_indices.data();
    provenance.feature_ids = feature_ids.view();
    provenance.feature_names = feature_names.view();
    provenance.feature_types = feature_types.view();
    provenance.feature_dataset_ids = feature_dataset_ids.data();
    provenance.feature_local_indices = feature_local_indices.data();
    provenance.dataset_feature_offsets = dataset_feature_offsets.data();
    provenance.dataset_feature_to_global = dataset_feature_to_global.data();

    std::cerr << "creating CellShard HDF5 output=" << opts.output_path
              << " rows=" << total_rows
              << " cols=" << feature_count
              << " nnz=" << total_nnz
              << " partitions=" << part_rows.size()
              << " shards=" << shard_part_begin.size() << "\n";
    if (!cs::create_dataset_sliced_ell_h5(opts.output_path.c_str(), &layout, &dataset_view, &provenance)) {
        throw std::runtime_error("create_dataset_sliced_ell_h5 failed");
    }
    global_barcodes.release();
    feature_ids.release();
    feature_names.release();
    feature_types.release();
    dataset_ids.release();
    matrix_paths.release();
    feature_paths.release();
    barcode_paths.release();
    metadata_paths.release();

    std::vector<text_column> obs_text(opts.observation_columns.size() + 1u);
    std::vector<cs::dataset_observation_metadata_column_view> obs_views(opts.observation_columns.size() + 1u);
    for (std::size_t col = 0; col < opts.observation_columns.size(); ++col) {
        append_string_vector(&obs_text[col], obs.values[col]);
        obs_views[col] = {opts.observation_columns[col].c_str(), cs::dataset_observation_metadata_type_text, obs_text[col].view(), nullptr, nullptr};
    }
    append_string_vector(&obs_text.back(), obs.source_dataset);
    obs_views.back() = {"source_dataset_id", cs::dataset_observation_metadata_type_text, obs_text.back().view(), nullptr, nullptr};
    const cs::dataset_observation_metadata_view obs_metadata{total_rows, (std::uint32_t) obs_views.size(), obs_views.data()};
    if (!cs::append_dataset_observation_metadata_h5(opts.output_path.c_str(), &obs_metadata)) {
        throw std::runtime_error("append_dataset_observation_metadata_h5 failed");
    }
    append_feature_metadata(opts, features);

    text_column attr_keys, attr_values;
    const std::vector<std::pair<const char *, const char *>> attrs{
        {"assay", "scRNA-seq"},
        {"modality", "raw count MatrixMarket series"},
        {"input_orientation", "features x cells MatrixMarket"},
        {"artifact_orientation", "cells x features"},
        {"feature_namespace", "feature IDs from features.tsv"},
        {"processing_state", "raw integer counts; no normalization, log transform, filtering, imputation, or modality merge"},
        {"local_barcode_index", "stored as provenance cell_local_indices"},
        {"payload_layout", "optimized_bucketed_sliced_ell"},
    };
    for (const auto &kv : attrs) {
        attr_keys.append(kv.first);
        attr_values.append(kv.second);
    }
    const cs::dataset_user_attribute_view attributes{(std::uint32_t) attrs.size(), attr_keys.view(), attr_values.view()};
    if (!cs::append_dataset_user_attributes_h5(opts.output_path.c_str(), &attributes)) {
        throw std::runtime_error("append_dataset_user_attributes_h5 failed");
    }
    st->metadata_seconds = metadata_timer.seconds();

    run_payload_workers(opts, datasets, feature_count, &part_bucket_counts, &part_bucketed_bytes, &part_execution_bytes, st);

    std::vector<std::uint32_t> zero_part_u32(part_rows.size(), 0u), zero_shard_u32(shard_part_begin.size(), 0u);
    std::vector<float> zero_part_f32(part_rows.size(), 0.0f), zero_shard_f32(shard_part_begin.size(), 0.0f);
    std::vector<std::uint64_t> zero_part_u64(part_rows.size(), 0u), zero_shard_u64(shard_part_begin.size(), 0u);
    std::vector<std::uint32_t> shard_formats(shard_part_begin.size(), cs::dataset_execution_format_bucketed_sliced_ell);
    std::vector<std::uint32_t> shard_bucketed_partition_counts(shard_part_begin.size(), 0u);
    std::vector<std::uint32_t> shard_sliced_counts(shard_part_begin.size(), 0u), shard_sliced_rows(shard_part_begin.size(), opts.slice_rows);
    std::vector<std::uint64_t> shard_execution_bytes(shard_part_begin.size(), 0u), shard_bucketed_bytes(shard_part_begin.size(), 0u);
    for (std::size_t shard = 0; shard < shard_part_begin.size(); ++shard) {
        for (std::uint32_t p = shard_part_begin[shard]; p < shard_part_end[shard]; ++p) {
            shard_bucketed_partition_counts[shard] += 1u;
            shard_sliced_counts[shard] += part_slice_counts[p];
            shard_execution_bytes[shard] += part_execution_bytes[p];
            shard_bucketed_bytes[shard] += part_bucketed_bytes[p];
        }
    }
    cs::dataset_execution_view execution{};
    execution.partition_count = (std::uint32_t) part_rows.size();
    execution.partition_execution_formats = part_formats.data();
    execution.partition_blocked_ell_block_sizes = zero_part_u32.data();
    execution.partition_blocked_ell_bucket_counts = zero_part_u32.data();
    execution.partition_blocked_ell_fill_ratios = zero_part_f32.data();
    execution.partition_execution_bytes = part_execution_bytes.data();
    execution.partition_blocked_ell_bytes = zero_part_u64.data();
    execution.partition_bucketed_blocked_ell_bytes = zero_part_u64.data();
    execution.partition_sliced_ell_slice_counts = part_slice_counts.data();
    execution.partition_sliced_ell_slice_rows = part_slice_rows.data();
    execution.partition_sliced_ell_bytes = part_sliced_bytes.data();
    execution.partition_bucketed_sliced_ell_bytes = part_bucketed_bytes.data();
    execution.shard_count = (std::uint32_t) shard_part_begin.size();
    execution.shard_execution_formats = shard_formats.data();
    execution.shard_blocked_ell_block_sizes = zero_shard_u32.data();
    execution.shard_bucketed_partition_counts = shard_bucketed_partition_counts.data();
    execution.shard_bucketed_segment_counts = zero_shard_u32.data();
    execution.shard_blocked_ell_fill_ratios = zero_shard_f32.data();
    execution.shard_execution_bytes = shard_execution_bytes.data();
    execution.shard_bucketed_blocked_ell_bytes = zero_shard_u64.data();
    execution.shard_sliced_ell_slice_counts = shard_sliced_counts.data();
    execution.shard_sliced_ell_slice_rows = shard_sliced_rows.data();
    execution.shard_bucketed_sliced_ell_bytes = shard_bucketed_bytes.data();
    execution.shard_preferred_pair_ids = zero_shard_u32.data();
    execution.shard_owner_node_ids = zero_shard_u32.data();
    execution.shard_owner_rank_ids = zero_shard_u32.data();
    execution.preferred_base_format = cs::dataset_execution_format_bucketed_sliced_ell;
    if (!cs::append_dataset_execution_h5(opts.output_path.c_str(), &execution)) {
        throw std::runtime_error("append_dataset_execution_h5 failed");
    }

    cs::dataset_runtime_service_view runtime_service{};
    cs::init(&runtime_service);
    runtime_service.service_mode = cs::dataset_runtime_service_mode_local_cache;
    runtime_service.live_write_mode = cs::dataset_live_write_mode_read_only;
    runtime_service.prefer_pack_delivery = 1u;
    runtime_service.canonical_generation = 1u;
    runtime_service.execution_plan_generation = 1u;
    if (!cs::append_dataset_runtime_service_h5(opts.output_path.c_str(), &runtime_service)) {
        throw std::runtime_error("append_dataset_runtime_service_h5 failed");
    }

    if (opts.atomic_output) maybe_move_tmp_output(opts.output_path, final_output);
    st->total_seconds = total_timer.seconds();
    std::cerr << "completed output=" << final_output
              << " rows=" << total_rows
              << " cols=" << feature_count
              << " nnz=" << total_nnz
              << " seconds=" << st->total_seconds << "\n";
    return 1;
}

} // namespace mtx_series
} // namespace ingest
} // namespace cellshard
