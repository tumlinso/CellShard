#include "../../../include/CellShard/io/cshard.hh"

#include "../../../include/CellShard/formats/blocked_ell.cuh"
#include "../../../include/CellShard/formats/sliced_ell.cuh"
#include "../../../include/CellShard/io/csh5/api.cuh"
#include "../../../include/CellShard/runtime/host/sharded_host.cuh"
#include "../../../include/CellShard/runtime/storage/disk.cuh"

#include <algorithm>
#include <array>
#include <cstring>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <unordered_map>

namespace cellshard::cshard {

namespace {

using spec::section_entry;
namespace fs = std::filesystem;

struct payload_section {
    section_entry entry{};
    std::vector<std::uint8_t> bytes;
};

struct file_plan {
    spec::header header{};
    std::vector<spec::metadata_record> metadata;
    std::vector<section_entry> sections;
    std::vector<spec::assay_descriptor> assays;
    std::vector<spec::pairing_descriptor> pairings;
    std::vector<spec::matrix_descriptor> matrices;
    std::vector<spec::table_column_descriptor> obs_columns;
    std::vector<spec::table_column_descriptor> var_columns;
    std::vector<payload_section> payloads;
};

[[noreturn]] void throw_format(const std::string &message) {
    throw std::runtime_error("invalid .cshard file: " + message);
}

void set_error(std::string *error, const std::string &message) {
    if (error != nullptr) *error = message;
}

std::uint64_t align_up(std::uint64_t value, std::uint64_t alignment) {
    if (alignment == 0u) return value;
    const std::uint64_t mask = alignment - 1u;
    return (value + mask) & ~mask;
}

std::uint64_t fnv1a64(const void *data, std::size_t bytes, std::uint64_t seed = 1469598103934665603ull) {
    const auto *ptr = static_cast<const std::uint8_t *>(data);
    std::uint64_t hash = seed;
    for (std::size_t i = 0u; i < bytes; ++i) {
        hash ^= ptr[i];
        hash *= 1099511628211ull;
    }
    return hash;
}

std::uint64_t feature_hash_from_table(const table_view &var) {
    const table_view::column *selected = nullptr;
    for (const table_view::column &column : var.columns) {
        if (column.name == "id" || column.name == "feature_id") {
            selected = &column;
            break;
        }
        if (selected == nullptr && (column.name == "name" || column.name == "_index")) selected = &column;
    }
    if (selected == nullptr || selected->text_values.empty()) return 0u;
    std::uint64_t hash = 1469598103934665603ull;
    for (const std::string &value : selected->text_values) {
        hash = fnv1a64(value.data(), value.size(), hash);
        const char zero = '\0';
        hash = fnv1a64(&zero, 1u, hash);
    }
    return hash;
}

template<typename T>
std::vector<std::uint8_t> bytes_from_array(const T *ptr, std::size_t count) {
    std::vector<std::uint8_t> out(count * sizeof(T));
    if (!out.empty()) std::memcpy(out.data(), ptr, out.size());
    return out;
}

template<typename T>
bool read_exact_at(const std::string &path, std::uint64_t offset, T *out, std::size_t count = 1u) {
    std::ifstream in(path, std::ios::binary);
    if (!in || out == nullptr) return false;
    in.seekg((std::streamoff) offset);
    if (!in) return false;
    in.read(reinterpret_cast<char *>(out), (std::streamsize) (count * sizeof(T)));
    return (bool) in;
}

std::vector<std::uint8_t> read_bytes_at(const std::string &path, std::uint64_t offset, std::uint64_t bytes) {
    std::vector<std::uint8_t> out((std::size_t) bytes);
    std::ifstream in(path, std::ios::binary);
    if (!in) throw_format("failed to open file for payload read");
    in.seekg((std::streamoff) offset);
    if (!in) throw_format("failed to seek to payload");
    if (!out.empty()) in.read(reinterpret_cast<char *>(out.data()), (std::streamsize) out.size());
    if (!in && !out.empty()) throw_format("short payload read");
    return out;
}

template<typename T>
std::vector<T> read_array_section(const std::string &path, const section_entry &section) {
    if (section.bytes % sizeof(T) != 0u) throw_format("section byte count is not element-aligned");
    std::vector<T> out((std::size_t) (section.bytes / sizeof(T)));
    if (!out.empty() && !read_exact_at(path, section.offset, out.data(), out.size())) {
        throw_format("failed to read array section");
    }
    return out;
}

template<typename T>
void write_pod(std::ofstream &out, const T &value) {
    out.write(reinterpret_cast<const char *>(&value), sizeof(T));
}

void write_zeroes(std::ofstream &out, std::uint64_t count) {
    static const std::array<char, 4096> zeroes{};
    while (count != 0u) {
        const std::uint64_t chunk = std::min<std::uint64_t>(count, zeroes.size());
        out.write(zeroes.data(), (std::streamsize) chunk);
        count -= chunk;
    }
}

void pad_to(std::ofstream &out, std::uint64_t target) {
    const std::uint64_t current = (std::uint64_t) out.tellp();
    if (current > target) throw std::runtime_error("internal cshard writer offset regression");
    write_zeroes(out, target - current);
}

void copy_fixed(char *dst, std::size_t dst_size, const std::string &value) {
    if (dst == nullptr || dst_size == 0u) return;
    std::memset(dst, 0, dst_size);
    std::memcpy(dst, value.data(), std::min<std::size_t>(value.size(), dst_size - 1u));
}

void add_metadata(file_plan *plan, const std::string &key, const std::string &value) {
    spec::metadata_record record{};
    copy_fixed(record.key, sizeof(record.key), key);
    copy_fixed(record.value, sizeof(record.value), value);
    plan->metadata.push_back(record);
}

std::uint32_t add_payload(file_plan *plan,
                          spec::section_kind kind,
                          spec::dtype dtype,
                          std::vector<std::uint8_t> bytes,
                          std::uint64_t element_count) {
    payload_section section{};
    section.entry.kind = kind;
    section.entry.id = (std::uint32_t) plan->payloads.size() + 1u;
    section.entry.bytes = (std::uint64_t) bytes.size();
    section.entry.element_count = element_count;
    section.entry.dtype = dtype;
    section.entry.alignment = (std::uint32_t) spec::payload_alignment;
    section.bytes = std::move(bytes);
    plan->payloads.push_back(std::move(section));
    return plan->payloads.back().entry.id;
}

std::vector<std::uint8_t> encode_text_offsets(const std::vector<std::string> &values) {
    std::vector<std::uint64_t> offsets(values.size() + 1u, 0u);
    std::uint64_t cursor = 0u;
    for (std::size_t i = 0u; i < values.size(); ++i) {
        offsets[i] = cursor;
        cursor += (std::uint64_t) values[i].size() + 1u;
    }
    offsets[values.size()] = cursor;
    return bytes_from_array(offsets.data(), offsets.size());
}

std::vector<std::uint8_t> encode_text_bytes(const std::vector<std::string> &values) {
    std::vector<std::uint8_t> bytes;
    std::uint64_t total = 0u;
    for (const std::string &value : values) total += (std::uint64_t) value.size() + 1u;
    bytes.reserve((std::size_t) total);
    for (const std::string &value : values) {
        bytes.insert(bytes.end(), value.begin(), value.end());
        bytes.push_back(0u);
    }
    return bytes;
}

void add_table_columns(file_plan *plan,
                       const table_view &table,
                       bool var_table,
                       std::vector<spec::table_column_descriptor> *out) {
    if (out == nullptr) return;
    out->clear();
    out->reserve(table.columns.size());
    for (const table_view::column &column : table.columns) {
        spec::table_column_descriptor desc{};
        copy_fixed(desc.name, sizeof(desc.name), column.name);
        desc.column_type = column.type;
        desc.row_count = (std::uint32_t) table.rows;
        if (column.type == spec::table_column_text) {
            desc.data_dtype = spec::dtype_u8;
            desc.offsets_section_id = add_payload(plan,
                                                  spec::section_kind_table_string_offsets,
                                                  spec::dtype_u64,
                                                  encode_text_offsets(column.text_values),
                                                  column.text_values.size() + 1u);
            desc.data_section_id = add_payload(plan,
                                               spec::section_kind_table_string_bytes,
                                               spec::dtype_u8,
                                               encode_text_bytes(column.text_values),
                                               0u);
        } else if (column.type == spec::table_column_float32) {
            desc.data_dtype = spec::dtype_f32;
            desc.data_section_id = add_payload(plan,
                                               spec::section_kind_table_float32_values,
                                               spec::dtype_f32,
                                               bytes_from_array(column.float32_values.data(), column.float32_values.size()),
                                               column.float32_values.size());
        } else if (column.type == spec::table_column_uint8) {
            desc.data_dtype = spec::dtype_u8;
            desc.data_section_id = add_payload(plan,
                                               spec::section_kind_table_uint8_values,
                                               spec::dtype_u8,
                                               bytes_from_array(column.uint8_values.data(), column.uint8_values.size()),
                                               column.uint8_values.size());
        }
        (void) var_table;
        out->push_back(desc);
    }
}

std::pair<std::uint32_t, std::uint32_t> append_table_columns(file_plan *plan,
                                                             const table_view &table,
                                                             std::vector<spec::table_column_descriptor> *out) {
    if (out == nullptr) return {0u, 0u};
    const std::uint32_t begin = (std::uint32_t) out->size();
    for (const table_view::column &column : table.columns) {
        spec::table_column_descriptor desc{};
        copy_fixed(desc.name, sizeof(desc.name), column.name);
        desc.column_type = column.type;
        desc.row_count = (std::uint32_t) table.rows;
        if (column.type == spec::table_column_text) {
            desc.data_dtype = spec::dtype_u8;
            desc.offsets_section_id = add_payload(plan,
                                                  spec::section_kind_table_string_offsets,
                                                  spec::dtype_u64,
                                                  encode_text_offsets(column.text_values),
                                                  column.text_values.size() + 1u);
            desc.data_section_id = add_payload(plan,
                                               spec::section_kind_table_string_bytes,
                                               spec::dtype_u8,
                                               encode_text_bytes(column.text_values),
                                               0u);
        } else if (column.type == spec::table_column_float32) {
            desc.data_dtype = spec::dtype_f32;
            desc.data_section_id = add_payload(plan,
                                               spec::section_kind_table_float32_values,
                                               spec::dtype_f32,
                                               bytes_from_array(column.float32_values.data(), column.float32_values.size()),
                                               column.float32_values.size());
        } else if (column.type == spec::table_column_uint8) {
            desc.data_dtype = spec::dtype_u8;
            desc.data_section_id = add_payload(plan,
                                               spec::section_kind_table_uint8_values,
                                               spec::dtype_u8,
                                               bytes_from_array(column.uint8_values.data(), column.uint8_values.size()),
                                               column.uint8_values.size());
        }
        out->push_back(desc);
    }
    return {begin, (std::uint32_t) table.columns.size()};
}

const section_entry *find_section(const std::vector<cshard_file::section_record> &sections, std::uint32_t id) {
    for (const auto &section : sections) {
        if (section.entry.id == id) return &section.entry;
    }
    return nullptr;
}

void validate_section_bounds(const spec::header &header, const section_entry &section) {
    if (section.alignment != 0u && (section.offset % section.alignment) != 0u) throw_format("section alignment mismatch");
    if (section.offset > header.file_size || section.bytes > header.file_size - section.offset) {
        throw_format("section extends past end of file");
    }
}

std::string matrix_layout_to_string(std::uint32_t layout) {
    return layout_name(layout);
}

table_view decode_table(const std::string &path,
                        const std::vector<cshard_file::section_record> &sections,
                        const std::vector<spec::table_column_descriptor> &columns,
                        std::uint64_t expected_rows) {
    table_view out;
    out.rows = expected_rows;
    out.columns.reserve(columns.size());
    for (const spec::table_column_descriptor &desc : columns) {
        table_view::column column;
        column.name = desc.name;
        column.type = desc.column_type;
        if (desc.row_count != expected_rows) throw_format("table column row count mismatch");
        if (desc.column_type == spec::table_column_text) {
            const section_entry *offsets_section = find_section(sections, desc.offsets_section_id);
            const section_entry *data_section = find_section(sections, desc.data_section_id);
            if (offsets_section == nullptr || data_section == nullptr) throw_format("text table column references a missing section");
            std::vector<std::uint64_t> offsets = read_array_section<std::uint64_t>(path, *offsets_section);
            std::vector<std::uint8_t> bytes = read_bytes_at(path, data_section->offset, data_section->bytes);
            if (offsets.size() != expected_rows + 1u) throw_format("text table offsets count mismatch");
            column.text_values.reserve((std::size_t) expected_rows);
            for (std::uint64_t row = 0u; row < expected_rows; ++row) {
                const std::uint64_t begin = offsets[(std::size_t) row];
                const std::uint64_t end = offsets[(std::size_t) row + 1u];
                if (end < begin || end > bytes.size()) throw_format("text table offset out of bounds");
                const std::uint64_t len = end > begin && bytes[(std::size_t) end - 1u] == 0u
                    ? end - begin - 1u
                    : end - begin;
                column.text_values.emplace_back(reinterpret_cast<const char *>(bytes.data() + begin), (std::size_t) len);
            }
        } else if (desc.column_type == spec::table_column_float32) {
            const section_entry *data_section = find_section(sections, desc.data_section_id);
            if (data_section == nullptr) throw_format("float table column references a missing section");
            column.float32_values = read_array_section<float>(path, *data_section);
            if (column.float32_values.size() != expected_rows) throw_format("float table column row count mismatch");
        } else if (desc.column_type == spec::table_column_uint8) {
            const section_entry *data_section = find_section(sections, desc.data_section_id);
            if (data_section == nullptr) throw_format("uint8 table column references a missing section");
            column.uint8_values = read_array_section<std::uint8_t>(path, *data_section);
            if (column.uint8_values.size() != expected_rows) throw_format("uint8 table column row count mismatch");
        } else {
            throw_format("unknown table column type");
        }
        out.columns.push_back(std::move(column));
    }
    return out;
}

table_view decode_table_range(const std::string &path,
                              const std::vector<cshard_file::section_record> &sections,
                              const std::vector<spec::table_column_descriptor> &columns,
                              std::uint32_t begin,
                              std::uint32_t count,
                              std::uint64_t expected_rows) {
    if (begin > columns.size() || count > columns.size() - begin) throw_format("table column descriptor range is out of bounds");
    std::vector<spec::table_column_descriptor> selected;
    selected.reserve(count);
    for (std::uint32_t i = 0u; i < count; ++i) selected.push_back(columns[(std::size_t) begin + i]);
    return decode_table(path, sections, selected, expected_rows);
}

void finalize_offsets(file_plan *plan) {
    std::uint64_t cursor = sizeof(spec::header);
    plan->header.magic = spec::magic;
    plan->header.version_major = spec::version_major;
    plan->header.version_minor = spec::version_minor;
    plan->header.endian_tag = spec::endian_tag;
    plan->header.header_bytes = sizeof(spec::header);
    plan->header.metadata_count = (std::uint32_t) plan->metadata.size();
    plan->header.metadata_offset = cursor;
    cursor += (std::uint64_t) plan->metadata.size() * sizeof(spec::metadata_record);
    cursor = align_up(cursor, 8u);

    plan->header.assay_directory_offset = cursor;
    plan->header.assay_directory_count = (std::uint32_t) plan->assays.size();
    cursor += (std::uint64_t) plan->assays.size() * sizeof(spec::assay_descriptor);
    cursor = align_up(cursor, 8u);

    plan->header.pairing_directory_offset = cursor;
    if (!plan->pairings.empty()) plan->header.pairing_kind = plan->pairings[0].pairing_kind;
    cursor += (std::uint64_t) plan->pairings.size() * sizeof(spec::pairing_descriptor);
    cursor = align_up(cursor, 8u);

    plan->header.matrix_directory_offset = cursor;
    plan->header.matrix_directory_count = (std::uint32_t) plan->matrices.size();
    cursor += (std::uint64_t) plan->matrices.size() * sizeof(spec::matrix_descriptor);
    cursor = align_up(cursor, 8u);

    plan->header.obs_table_directory_offset = cursor;
    plan->header.obs_table_column_count = (std::uint32_t) plan->obs_columns.size();
    cursor += (std::uint64_t) plan->obs_columns.size() * sizeof(spec::table_column_descriptor);
    cursor = align_up(cursor, 8u);

    plan->header.var_table_directory_offset = cursor;
    plan->header.var_table_column_count = (std::uint32_t) plan->var_columns.size();
    cursor += (std::uint64_t) plan->var_columns.size() * sizeof(spec::table_column_descriptor);
    cursor = align_up(cursor, 8u);

    plan->sections.clear();
    plan->sections.reserve(plan->payloads.size() + 5u);
    for (payload_section &payload : plan->payloads) plan->sections.push_back(payload.entry);

    plan->header.section_directory_offset = cursor;
    plan->header.section_directory_count = (std::uint32_t) plan->sections.size();
    cursor += (std::uint64_t) plan->sections.size() * sizeof(section_entry);
    cursor = align_up(cursor, spec::payload_alignment);
    plan->header.payload_offset = cursor;

    for (std::size_t i = 0u; i < plan->payloads.size(); ++i) {
        cursor = align_up(cursor, spec::payload_alignment);
        plan->payloads[i].entry.offset = cursor;
        plan->sections[i].offset = cursor;
        cursor += (std::uint64_t) plan->payloads[i].bytes.size();
    }
    plan->header.payload_bytes = cursor - plan->header.payload_offset;
    plan->header.file_size = cursor;
}

bool write_plan(const std::string &path, file_plan *plan, std::string *error) {
    try {
        finalize_offsets(plan);
        std::ofstream out(path, std::ios::binary | std::ios::trunc);
        if (!out) {
            set_error(error, "failed to create output .cshard");
            return false;
        }
        write_pod(out, plan->header);
        for (const auto &record : plan->metadata) write_pod(out, record);
        pad_to(out, plan->header.assay_directory_offset);
        for (const auto &assay : plan->assays) write_pod(out, assay);
        pad_to(out, plan->header.pairing_directory_offset);
        for (const auto &pairing : plan->pairings) write_pod(out, pairing);
        pad_to(out, plan->header.matrix_directory_offset);
        for (const auto &matrix : plan->matrices) write_pod(out, matrix);
        pad_to(out, plan->header.obs_table_directory_offset);
        for (const auto &column : plan->obs_columns) write_pod(out, column);
        pad_to(out, plan->header.var_table_directory_offset);
        for (const auto &column : plan->var_columns) write_pod(out, column);
        pad_to(out, plan->header.section_directory_offset);
        for (const auto &section : plan->sections) write_pod(out, section);
        for (const payload_section &payload : plan->payloads) {
            pad_to(out, payload.entry.offset);
            if (!payload.bytes.empty()) {
                out.write(reinterpret_cast<const char *>(payload.bytes.data()), (std::streamsize) payload.bytes.size());
            }
        }
        return (bool) out;
    } catch (const std::exception &exc) {
        set_error(error, exc.what());
        return false;
    }
}

void validate_table_for_write(const table_view &table, std::uint64_t expected_rows, const char *label) {
    if (table.rows != expected_rows) throw std::runtime_error(std::string(label) + " table row count does not match matrix shape");
    if (table.rows > std::numeric_limits<std::uint32_t>::max()) {
        throw std::runtime_error(std::string(label) + " table row count exceeds cshard v1 descriptor range");
    }
    for (const table_view::column &column : table.columns) {
        if (column.type == spec::table_column_text && column.text_values.size() != expected_rows) {
            throw std::runtime_error(std::string(label) + " text column length mismatch");
        }
        if (column.type == spec::table_column_float32 && column.float32_values.size() != expected_rows) {
            throw std::runtime_error(std::string(label) + " float32 column length mismatch");
        }
        if (column.type == spec::table_column_uint8 && column.uint8_values.size() != expected_rows) {
            throw std::runtime_error(std::string(label) + " uint8 column length mismatch");
        }
        if (column.type != spec::table_column_text
            && column.type != spec::table_column_float32
            && column.type != spec::table_column_uint8) {
            throw std::runtime_error(std::string(label) + " column has unsupported type");
        }
    }
}

void validate_csr_for_write(const exporting::csr_matrix_export &csr) {
    if (csr.rows > std::numeric_limits<std::uint32_t>::max()
        || csr.cols > std::numeric_limits<std::uint32_t>::max()) {
        throw std::runtime_error("CSR shape exceeds cshard v1 descriptor range");
    }
    if (csr.indptr.size() != (std::size_t) csr.rows + 1u || csr.indices.size() != csr.data.size()) {
        throw std::runtime_error("invalid CSR input");
    }
    if (csr.indptr.empty() || csr.indptr[0] != 0) throw std::runtime_error("CSR row pointer must start at zero");
    for (std::size_t i = 1u; i < csr.indptr.size(); ++i) {
        if (csr.indptr[i] < csr.indptr[i - 1]) throw std::runtime_error("CSR row pointer must be monotonic");
    }
    if ((std::uint64_t) csr.indptr.back() != csr.data.size()) throw std::runtime_error("CSR row pointer terminal value mismatch");
    if (csr.data.size() > std::numeric_limits<std::uint32_t>::max()) {
        throw std::runtime_error("CSR nnz exceeds cshard v1 descriptor range");
    }
}

file_plan base_plan(std::uint64_t rows,
                    std::uint64_t cols,
                    std::uint64_t nnz,
                    std::uint32_t canonical_layout,
                    const table_view &obs,
                    const table_view &var,
                    const writer_options &options) {
    file_plan plan;
    validate_table_for_write(obs, rows, "obs");
    validate_table_for_write(var, cols, "var");
    plan.header.rows = rows;
    plan.header.cols = cols;
    plan.header.nnz = nnz;
    plan.header.canonical_layout = canonical_layout;
    plan.header.feature_order_hash = options.feature_order_hash != 0u ? options.feature_order_hash : feature_hash_from_table(var);
    if (plan.header.feature_order_hash == 0u) plan.header.feature_order_hash = fnv1a64(&cols, sizeof(cols));
    add_metadata(&plan, "format_role", "experimental_standby_archive");
    add_metadata(&plan, "schema", "cshard_v1");
    add_metadata(&plan, "matrix_layout", layout_name(canonical_layout));
    add_metadata(&plan, "feature_order_hash", std::to_string(plan.header.feature_order_hash));
    add_table_columns(&plan, obs, false, &plan.obs_columns);
    add_table_columns(&plan, var, true, &plan.var_columns);
    return plan;
}

std::uint32_t add_csr_matrix(file_plan *plan,
                             const exporting::csr_matrix_export &csr,
                             std::uint32_t matrix_id,
                             std::uint32_t partition_id,
                             std::uint64_t row_begin) {
    std::vector<std::uint64_t> row_ptr(csr.indptr.size());
    std::vector<std::uint32_t> col_idx(csr.indices.size());
    std::vector<real::storage_t> values(csr.data.size());
    for (std::size_t i = 0u; i < csr.indptr.size(); ++i) row_ptr[i] = (std::uint64_t) csr.indptr[i];
    for (std::size_t i = 0u; i < csr.indices.size(); ++i) {
        if (csr.indices[i] < 0 || (std::uint64_t) csr.indices[i] >= csr.cols) throw std::runtime_error("CSR column index out of range");
        col_idx[i] = (std::uint32_t) csr.indices[i];
        values[i] = __float2half(csr.data[i]);
    }
    spec::matrix_descriptor desc{};
    desc.matrix_id = matrix_id;
    desc.partition_id = partition_id;
    desc.layout = spec::matrix_layout_csr;
    desc.value_dtype = spec::dtype_f16;
    desc.index_dtype = spec::dtype_u32;
    desc.row_begin = row_begin;
    desc.rows = csr.rows;
    desc.cols = csr.cols;
    desc.nnz = csr.data.size();
    desc.section_a_id = add_payload(plan,
                                    spec::section_kind_csr_row_ptr,
                                    spec::dtype_u64,
                                    bytes_from_array(row_ptr.data(), row_ptr.size()),
                                    row_ptr.size());
    desc.section_b_id = add_payload(plan,
                                    spec::section_kind_csr_col_idx,
                                    spec::dtype_u32,
                                    bytes_from_array(col_idx.data(), col_idx.size()),
                                    col_idx.size());
    desc.section_c_id = add_payload(plan,
                                    spec::section_kind_csr_values,
                                    spec::dtype_f16,
                                    bytes_from_array(values.data(), values.size()),
                                    values.size());
    plan->matrices.push_back(desc);
    return (std::uint32_t) plan->matrices.size() - 1u;
}

table_view table_from_summary_obs(const exporting::dataset_summary &summary,
                                  const std::vector<exporting::observation_metadata_column> &obs_columns) {
    table_view out;
    out.rows = summary.rows;
    table_view::column index;
    index.name = "_index";
    index.type = spec::table_column_text;
    index.text_values = summary.obs_names;
    if (index.text_values.size() < summary.rows) {
        index.text_values.resize((std::size_t) summary.rows);
        for (std::uint64_t row = 0u; row < summary.rows; ++row) {
            if (index.text_values[(std::size_t) row].empty()) index.text_values[(std::size_t) row] = "cell" + std::to_string(row);
        }
    }
    out.columns.push_back(std::move(index));
    for (const auto &src : obs_columns) {
        table_view::column column;
        column.name = src.name;
        column.type = src.type;
        column.text_values = src.text_values;
        column.float32_values = src.float32_values;
        column.uint8_values = src.uint8_values;
        out.columns.push_back(std::move(column));
    }
    return out;
}

table_view table_from_summary_var(const exporting::dataset_summary &summary,
                                  const std::vector<exporting::annotation_column> &var_columns) {
    table_view out;
    out.rows = summary.cols;
    const std::array<std::pair<const char *, const std::vector<std::string> *>, 3u> basics = {{
        {"id", &summary.var_ids},
        {"name", &summary.var_names},
        {"type", &summary.var_types}
    }};
    for (const auto &basic : basics) {
        table_view::column column;
        column.name = basic.first;
        column.type = spec::table_column_text;
        column.text_values = *basic.second;
        if (column.text_values.size() < summary.cols) {
            column.text_values.resize((std::size_t) summary.cols);
            for (std::uint64_t col = 0u; col < summary.cols; ++col) {
                if (column.text_values[(std::size_t) col].empty()) column.text_values[(std::size_t) col] = "feature" + std::to_string(col);
            }
        }
        out.columns.push_back(std::move(column));
    }
    for (const auto &src : var_columns) {
        if (src.name == "id" || src.name == "name" || src.name == "type") continue;
        table_view::column column;
        column.name = src.name;
        column.type = src.type;
        column.text_values = src.text_values;
        column.float32_values = src.float32_values;
        column.uint8_values = src.uint8_values;
        out.columns.push_back(std::move(column));
    }
    return out;
}

bool add_blocked_part(file_plan *plan,
                      std::uint32_t partition_id,
                      std::uint64_t row_begin,
                      const sparse::blocked_ell *part,
                      std::string *error) {
    if (plan == nullptr || part == nullptr) {
        set_error(error, "blocked-ELL partition is null");
        return false;
    }
    const std::uint64_t block_idx_count =
        (std::uint64_t) sparse::row_block_count(part) * (std::uint64_t) sparse::ell_width_blocks(part);
    const std::uint64_t value_count = (std::uint64_t) part->rows * (std::uint64_t) part->ell_cols;
    spec::matrix_descriptor desc{};
    desc.matrix_id = partition_id;
    desc.partition_id = partition_id;
    desc.layout = spec::matrix_layout_blocked_ell;
    desc.value_dtype = spec::dtype_f16;
    desc.index_dtype = spec::dtype_u32;
    desc.row_begin = row_begin;
    desc.rows = part->rows;
    desc.cols = part->cols;
    desc.nnz = part->nnz;
    desc.block_size = part->block_size;
    desc.ell_width = sparse::ell_width_blocks(part);
    desc.aux0 = part->ell_cols;
    desc.section_a_id = add_payload(plan,
                                    spec::section_kind_blocked_ell_block_col_idx,
                                    spec::dtype_u32,
                                    bytes_from_array(part->blockColIdx, (std::size_t) block_idx_count),
                                    block_idx_count);
    desc.section_b_id = add_payload(plan,
                                    spec::section_kind_blocked_ell_values,
                                    spec::dtype_f16,
                                    bytes_from_array(part->val, (std::size_t) value_count),
                                    value_count);
    plan->matrices.push_back(desc);
    return true;
}

bool add_sliced_part(file_plan *plan,
                     std::uint32_t partition_id,
                     std::uint64_t row_begin,
                     const sparse::sliced_ell *part,
                     std::string *error) {
    if (plan == nullptr || part == nullptr) {
        set_error(error, "sliced-ELL partition is null");
        return false;
    }
    const std::uint64_t slot_count = sparse::total_slots(part);
    spec::matrix_descriptor desc{};
    desc.matrix_id = partition_id;
    desc.partition_id = partition_id;
    desc.layout = spec::matrix_layout_sliced_ell;
    desc.value_dtype = spec::dtype_f16;
    desc.index_dtype = spec::dtype_u32;
    desc.row_begin = row_begin;
    desc.rows = part->rows;
    desc.cols = part->cols;
    desc.nnz = part->nnz;
    desc.slice_count = part->slice_count;
    desc.aux0 = slot_count;
    desc.section_a_id = add_payload(plan,
                                    spec::section_kind_sliced_ell_slice_row_offsets,
                                    spec::dtype_u32,
                                    bytes_from_array(part->slice_row_offsets, (std::size_t) part->slice_count + 1u),
                                    (std::uint64_t) part->slice_count + 1u);
    desc.section_b_id = add_payload(plan,
                                    spec::section_kind_sliced_ell_slice_widths,
                                    spec::dtype_u32,
                                    bytes_from_array(part->slice_widths, (std::size_t) part->slice_count),
                                    part->slice_count);
    desc.section_c_id = add_payload(plan,
                                    spec::section_kind_sliced_ell_col_idx,
                                    spec::dtype_u32,
                                    bytes_from_array(part->col_idx, (std::size_t) slot_count),
                                    slot_count);
    desc.section_d_id = add_payload(plan,
                                    spec::section_kind_sliced_ell_values,
                                    spec::dtype_f16,
                                    bytes_from_array(part->val, (std::size_t) slot_count),
                                    slot_count);
    plan->matrices.push_back(desc);
    return true;
}

template<typename ShardT>
struct optimized_shard_shape {
    std::uint64_t rows = 0u;
    std::uint64_t cols = 0u;
    std::uint64_t nnz = 0u;
};

std::vector<std::uint8_t> serialize_blocked_shard_blob(const bucketed_blocked_ell_shard *shard) {
    unsigned char *raw = nullptr;
    std::size_t bytes = 0u;
    if (shard == nullptr || !serialize_bucketed_blocked_ell_shard_blob(shard, &raw, &bytes)) {
        throw std::runtime_error("failed to serialize optimized Blocked-ELL shard");
    }
    std::vector<std::uint8_t> out(raw, raw + bytes);
    std::free(raw);
    return out;
}

std::vector<std::uint8_t> serialize_sliced_shard_blob(const bucketed_sliced_ell_shard *shard) {
    unsigned char *raw = nullptr;
    std::size_t bytes = 0u;
    if (shard == nullptr || !serialize_bucketed_sliced_ell_shard_blob(shard, &raw, &bytes)) {
        throw std::runtime_error("failed to serialize optimized Sliced-ELL shard");
    }
    std::vector<std::uint8_t> out(raw, raw + bytes);
    std::free(raw);
    return out;
}

std::vector<std::uint8_t> blob_from_input(const void *blob, std::size_t bytes) {
    if (blob == nullptr || bytes == 0u) throw std::runtime_error("optimized shard blob is empty");
    const auto *begin = static_cast<const std::uint8_t *>(blob);
    return std::vector<std::uint8_t>(begin, begin + bytes);
}

optimized_shard_shape<bucketed_blocked_ell_shard> blocked_blob_shape(const std::vector<std::uint8_t> &bytes) {
    bucketed_blocked_ell_shard shard;
    init(&shard);
    if (!deserialize_bucketed_blocked_ell_shard_blob(bytes.data(), bytes.size(), &shard)) {
        throw std::runtime_error("failed to deserialize optimized Blocked-ELL shard blob");
    }
    optimized_shard_shape<bucketed_blocked_ell_shard> shape{shard.rows, shard.cols, shard.nnz};
    clear(&shard);
    return shape;
}

optimized_shard_shape<bucketed_sliced_ell_shard> sliced_blob_shape(const std::vector<std::uint8_t> &bytes) {
    bucketed_sliced_ell_shard shard;
    init(&shard);
    if (!deserialize_bucketed_sliced_ell_shard_blob(bytes.data(), bytes.size(), &shard)) {
        throw std::runtime_error("failed to deserialize optimized Sliced-ELL shard blob");
    }
    optimized_shard_shape<bucketed_sliced_ell_shard> shape{shard.rows, shard.cols, shard.nnz};
    clear(&shard);
    return shape;
}

void append_blocked_shard_row(const bucketed_blocked_ell_shard &shard,
                              std::uint64_t local_row,
                              exporting::csr_matrix_export *out);
void append_sliced_shard_row(const bucketed_sliced_ell_shard &shard,
                             std::uint64_t local_row,
                             exporting::csr_matrix_export *out);

void append_row_value(exporting::csr_matrix_export *out, std::uint32_t col, const real::storage_t &raw) {
    const float value = __half2float(raw);
    if (value == 0.0f) return;
    out->indices.push_back((std::int64_t) col);
    out->data.push_back(value);
}

void append_blocked_row_from_file(const std::string &path,
                                  const spec::matrix_descriptor &matrix,
                                  const section_entry &idx_section,
                                  const section_entry &val_section,
                                  std::uint64_t local_row,
                                  exporting::csr_matrix_export *out) {
    const std::uint64_t width = matrix.ell_width;
    const std::uint64_t block = matrix.block_size;
    const std::uint64_t row_block = block == 0u ? 0u : local_row / block;
    const std::uint64_t idx_offset = idx_section.offset + row_block * width * sizeof(std::uint32_t);
    const std::uint64_t val_offset = val_section.offset + local_row * matrix.aux0 * sizeof(real::storage_t);
    std::vector<std::uint32_t> block_idx((std::size_t) width);
    std::vector<real::storage_t> values((std::size_t) matrix.aux0);
    if (block == 0u || matrix.aux0 != width * block) throw_format("blocked-ELL descriptor has inconsistent block metadata");
    if (!block_idx.empty() && !read_exact_at(path, idx_offset, block_idx.data(), block_idx.size())) throw_format("failed to read blocked-ELL block indices");
    if (!values.empty() && !read_exact_at(path, val_offset, values.data(), values.size())) throw_format("failed to read blocked-ELL values");
    for (std::uint64_t slot = 0u; slot < width; ++slot) {
        const std::uint32_t stored = block_idx[(std::size_t) slot];
        if (stored == sparse::blocked_ell_invalid_col) continue;
        for (std::uint64_t col_in_block = 0u; col_in_block < block; ++col_in_block) {
            const std::uint64_t col = (std::uint64_t) stored * block + col_in_block;
            if (col >= matrix.cols) continue;
            append_row_value(out, (std::uint32_t) col, values[(std::size_t) (slot * block + col_in_block)]);
        }
    }
}

void append_sliced_row_from_file(const std::string &path,
                                 const spec::matrix_descriptor &matrix,
                                 const section_entry &offset_section,
                                 const section_entry &width_section,
                                 const section_entry &idx_section,
                                 const section_entry &val_section,
                                 std::uint64_t local_row,
                                 exporting::csr_matrix_export *out) {
    const std::vector<std::uint32_t> slice_offsets = read_array_section<std::uint32_t>(path, offset_section);
    const std::vector<std::uint32_t> slice_widths = read_array_section<std::uint32_t>(path, width_section);
    if (slice_offsets.size() != (std::size_t) matrix.slice_count + 1u || slice_widths.size() != matrix.slice_count) {
        throw_format("sliced-ELL slice metadata count mismatch");
    }
    std::uint64_t slot_base = 0u;
    std::uint32_t slice = 0u;
    for (; slice < matrix.slice_count; ++slice) {
        const std::uint32_t begin = slice_offsets[(std::size_t) slice];
        const std::uint32_t end = slice_offsets[(std::size_t) slice + 1u];
        if (local_row < end) break;
        slot_base += (std::uint64_t) (end - begin) * slice_widths[(std::size_t) slice];
    }
    if (slice >= matrix.slice_count) return;
    const std::uint64_t row_in_slice = local_row - slice_offsets[(std::size_t) slice];
    const std::uint64_t width = slice_widths[(std::size_t) slice];
    const std::uint64_t row_slot = slot_base + row_in_slice * width;
    std::vector<std::uint32_t> cols((std::size_t) width);
    std::vector<real::storage_t> values((std::size_t) width);
    if (width != 0u && !read_exact_at(path, idx_section.offset + row_slot * sizeof(std::uint32_t), cols.data(), cols.size())) {
        throw_format("failed to read sliced-ELL column indices");
    }
    if (width != 0u && !read_exact_at(path, val_section.offset + row_slot * sizeof(real::storage_t), values.data(), values.size())) {
        throw_format("failed to read sliced-ELL values");
    }
    for (std::uint64_t slot = 0u; slot < width; ++slot) {
        const std::uint32_t col = cols[(std::size_t) slot];
        if (col == sparse::sliced_ell_invalid_col || col >= matrix.cols) continue;
        append_row_value(out, col, values[(std::size_t) slot]);
    }
}

const bucketed_blocked_ell_partition *find_blocked_partition(const bucketed_blocked_ell_shard &shard,
                                                             std::uint64_t local_row,
                                                             std::uint32_t *partition_row_out) {
    if (partition_row_out == nullptr || local_row >= shard.rows || shard.partition_row_offsets == nullptr) return nullptr;
    for (std::uint32_t partition = 0u; partition < shard.partition_count; ++partition) {
        const std::uint32_t begin = shard.partition_row_offsets[partition];
        const std::uint32_t end = shard.partition_row_offsets[partition + 1u];
        if (local_row >= begin && local_row < end) {
            *partition_row_out = (std::uint32_t) local_row - begin;
            return shard.partitions + partition;
        }
    }
    return nullptr;
}

const bucketed_sliced_ell_partition *find_sliced_partition(const bucketed_sliced_ell_shard &shard,
                                                           std::uint64_t local_row,
                                                           std::uint32_t *partition_row_out) {
    if (partition_row_out == nullptr || local_row >= shard.rows || shard.partition_row_offsets == nullptr) return nullptr;
    for (std::uint32_t partition = 0u; partition < shard.partition_count; ++partition) {
        const std::uint32_t begin = shard.partition_row_offsets[partition];
        const std::uint32_t end = shard.partition_row_offsets[partition + 1u];
        if (local_row >= begin && local_row < end) {
            *partition_row_out = (std::uint32_t) local_row - begin;
            return shard.partitions + partition;
        }
    }
    return nullptr;
}

void append_blocked_shard_row(const bucketed_blocked_ell_shard &shard,
                              std::uint64_t local_row,
                              exporting::csr_matrix_export *out) {
    std::uint32_t target_partition_row = 0u;
    const bucketed_blocked_ell_partition *part = find_blocked_partition(shard, local_row, &target_partition_row);
    if (part == nullptr) throw_format("optimized Blocked-ELL row is outside shard partition offsets");
    for (std::uint32_t segment = 0u; segment < part->segment_count; ++segment) {
        const sparse::blocked_ell *seg = part->segments + segment;
        const std::uint32_t block_size = seg->block_size;
        const std::uint32_t width_blocks = sparse::ell_width_blocks(seg);
        const std::uint32_t exec_row_base = part->segment_row_offsets[segment];
        if (block_size == 0u || seg->ell_cols != width_blocks * block_size) {
            throw_format("optimized Blocked-ELL segment has inconsistent block metadata");
        }
        for (std::uint32_t row = 0u; row < seg->rows; ++row) {
            if (part->exec_to_canonical_rows[exec_row_base + row] != target_partition_row) continue;
            const std::uint32_t row_block = row / block_size;
            for (std::uint32_t slot = 0u; slot < width_blocks; ++slot) {
                const types::idx_t block_col = seg->blockColIdx[(std::size_t) row_block * width_blocks + slot];
                if (block_col == sparse::blocked_ell_invalid_col) continue;
                for (std::uint32_t col_in_block = 0u; col_in_block < block_size; ++col_in_block) {
                    const std::uint32_t exec_col = (std::uint32_t) block_col * block_size + col_in_block;
                    if (exec_col >= part->cols) continue;
                    const std::uint32_t col = shard.exec_to_canonical_cols != nullptr
                        ? shard.exec_to_canonical_cols[exec_col]
                        : exec_col;
                    if (col >= shard.cols) throw_format("optimized Blocked-ELL column map is out of bounds");
                    const real::storage_t value =
                        seg->val[(std::size_t) row * seg->ell_cols + (std::size_t) slot * block_size + col_in_block];
                    append_row_value(out, col, value);
                }
            }
            return;
        }
    }
}

void append_sliced_shard_row(const bucketed_sliced_ell_shard &shard,
                             std::uint64_t local_row,
                             exporting::csr_matrix_export *out) {
    std::uint32_t target_partition_row = 0u;
    const bucketed_sliced_ell_partition *part = find_sliced_partition(shard, local_row, &target_partition_row);
    if (part == nullptr) throw_format("optimized Sliced-ELL row is outside shard partition offsets");
    for (std::uint32_t segment = 0u; segment < part->segment_count; ++segment) {
        const sparse::sliced_ell *seg = part->segments + segment;
        const std::uint32_t width = seg->slice_count != 0u ? seg->slice_widths[0] : 0u;
        const std::uint32_t exec_row_base = part->segment_row_offsets[segment];
        for (std::uint32_t row = 0u; row < seg->rows; ++row) {
            if (part->exec_to_canonical_rows[exec_row_base + row] != target_partition_row) continue;
            const std::size_t base = (std::size_t) row * width;
            for (std::uint32_t slot = 0u; slot < width; ++slot) {
                const std::uint32_t col = seg->col_idx[base + slot];
                if (col == sparse::sliced_ell_invalid_col) continue;
                if (col >= shard.cols) throw_format("optimized Sliced-ELL column index is out of bounds");
                append_row_value(out, col, seg->val[base + slot]);
            }
            return;
        }
    }
}

void append_optimized_blocked_row_from_file(const std::string &path,
                                            const spec::matrix_descriptor &matrix,
                                            const section_entry &blob_section,
                                            std::uint64_t local_row,
                                            exporting::csr_matrix_export *out) {
    std::vector<std::uint8_t> blob = read_bytes_at(path, blob_section.offset, blob_section.bytes);
    bucketed_blocked_ell_shard shard;
    init(&shard);
    if (!deserialize_bucketed_blocked_ell_shard_blob(blob.data(), blob.size(), &shard)) {
        throw_format("failed to deserialize optimized Blocked-ELL shard");
    }
    if (shard.rows != matrix.rows || shard.cols != matrix.cols || shard.nnz != matrix.nnz) {
        clear(&shard);
        throw_format("optimized Blocked-ELL shard shape does not match descriptor");
    }
    append_blocked_shard_row(shard, local_row, out);
    clear(&shard);
}

void append_optimized_sliced_row_from_file(const std::string &path,
                                           const spec::matrix_descriptor &matrix,
                                           const section_entry &blob_section,
                                           std::uint64_t local_row,
                                           exporting::csr_matrix_export *out) {
    std::vector<std::uint8_t> blob = read_bytes_at(path, blob_section.offset, blob_section.bytes);
    bucketed_sliced_ell_shard shard;
    init(&shard);
    if (!deserialize_bucketed_sliced_ell_shard_blob(blob.data(), blob.size(), &shard)) {
        throw_format("failed to deserialize optimized Sliced-ELL shard");
    }
    if (shard.rows != matrix.rows || shard.cols != matrix.cols || shard.nnz != matrix.nnz) {
        clear(&shard);
        throw_format("optimized Sliced-ELL shard shape does not match descriptor");
    }
    append_sliced_shard_row(shard, local_row, out);
    clear(&shard);
}

void append_csr_row_from_file(const std::string &path,
                              const spec::matrix_descriptor &matrix,
                              const section_entry &ptr_section,
                              const section_entry &idx_section,
                              const section_entry &val_section,
                              std::uint64_t local_row,
                              exporting::csr_matrix_export *out) {
    std::uint64_t bounds[2] = {0u, 0u};
    if (!read_exact_at(path, ptr_section.offset + local_row * sizeof(std::uint64_t), bounds, 2u)) {
        throw_format("failed to read CSR row pointer");
    }
    if (bounds[1] < bounds[0] || bounds[1] > matrix.nnz) throw_format("CSR row pointer is out of bounds");
    const std::uint64_t count = bounds[1] - bounds[0];
    std::vector<std::uint32_t> cols((std::size_t) count);
    std::vector<real::storage_t> values((std::size_t) count);
    if (count != 0u && !read_exact_at(path, idx_section.offset + bounds[0] * sizeof(std::uint32_t), cols.data(), cols.size())) {
        throw_format("failed to read CSR column indices");
    }
    if (count != 0u && !read_exact_at(path, val_section.offset + bounds[0] * sizeof(real::storage_t), values.data(), values.size())) {
        throw_format("failed to read CSR values");
    }
    for (std::uint64_t i = 0u; i < count; ++i) {
        if (cols[(std::size_t) i] >= matrix.cols) throw_format("CSR column index is out of bounds");
        append_row_value(out, cols[(std::size_t) i], values[(std::size_t) i]);
    }
}

void validate_matrix_sections(const std::string &path,
                              const std::vector<cshard_file::section_record> &sections,
                              const spec::matrix_descriptor &matrix) {
    if (matrix.value_dtype != spec::dtype_f16 || matrix.index_dtype != spec::dtype_u32) {
        throw_format("unsupported matrix descriptor dtype");
    }
    const section_entry *a = find_section(sections, matrix.section_a_id);
    const section_entry *b = find_section(sections, matrix.section_b_id);
    const section_entry *c = find_section(sections, matrix.section_c_id);
    const section_entry *d = find_section(sections, matrix.section_d_id);
    if (a == nullptr) throw_format("matrix descriptor references a missing section");
    if (matrix.layout == spec::matrix_layout_csr) {
        if (b == nullptr) throw_format("matrix descriptor references a missing second section");
        if (c == nullptr) throw_format("matrix descriptor references a missing third section");
        if (a->kind != spec::section_kind_csr_row_ptr || a->dtype != spec::dtype_u64 || a->element_count != matrix.rows + 1u) {
            throw_format("CSR row-pointer section metadata mismatch");
        }
        if (b->kind != spec::section_kind_csr_col_idx || b->dtype != spec::dtype_u32 || b->element_count != matrix.nnz) {
            throw_format("CSR column-index section metadata mismatch");
        }
        if (c->kind != spec::section_kind_csr_values || c->dtype != spec::dtype_f16 || c->element_count != matrix.nnz) {
            throw_format("CSR value section metadata mismatch");
        }
    } else if (matrix.layout == spec::matrix_layout_sliced_ell) {
        if (b == nullptr) throw_format("matrix descriptor references a missing second section");
        if (c == nullptr || d == nullptr) throw_format("sliced-ELL descriptor references a missing section");
    } else if (matrix.layout == spec::matrix_layout_bucketed_blocked_ell) {
        if (a->kind != spec::section_kind_optimized_blocked_ell_shard_blob || a->dtype != spec::dtype_u8) {
            throw_format("optimized Blocked-ELL blob section metadata mismatch");
        }
        const std::vector<std::uint8_t> blob = read_bytes_at(path, a->offset, a->bytes);
        const auto shape = blocked_blob_shape(blob);
        if (shape.rows != matrix.rows || shape.cols != matrix.cols || shape.nnz != matrix.nnz) {
            throw_format("optimized Blocked-ELL blob shape does not match descriptor");
        }
    } else if (matrix.layout == spec::matrix_layout_bucketed_sliced_ell) {
        if (a->kind != spec::section_kind_optimized_sliced_ell_shard_blob || a->dtype != spec::dtype_u8) {
            throw_format("optimized Sliced-ELL blob section metadata mismatch");
        }
        const std::vector<std::uint8_t> blob = read_bytes_at(path, a->offset, a->bytes);
        const auto shape = sliced_blob_shape(blob);
        if (shape.rows != matrix.rows || shape.cols != matrix.cols || shape.nnz != matrix.nnz) {
            throw_format("optimized Sliced-ELL blob shape does not match descriptor");
        }
    } else if (matrix.layout == spec::matrix_layout_blocked_ell) {
        if (b == nullptr) throw_format("matrix descriptor references a missing second section");
    } else {
        throw_format("unsupported matrix descriptor layout");
    }
}

void validate_matrix_range(const std::string &path,
                           const std::vector<cshard_file::section_record> &sections,
                           const std::vector<spec::matrix_descriptor> &matrices,
                           std::uint32_t begin,
                           std::uint32_t count,
                           std::uint64_t rows,
                           std::uint64_t cols,
                           std::uint64_t nnz) {
    if (count == 0u) throw_format("matrix descriptor range is empty");
    if (begin > matrices.size() || count > matrices.size() - begin) throw_format("matrix descriptor range is out of bounds");
    std::vector<const spec::matrix_descriptor *> selected;
    selected.reserve(count);
    for (std::uint32_t i = 0u; i < count; ++i) selected.push_back(&matrices[(std::size_t) begin + i]);
    std::sort(selected.begin(), selected.end(),
              [](const auto *lhs, const auto *rhs) { return lhs->row_begin < rhs->row_begin; });
    std::uint64_t row_cursor = 0u, nnz_sum = 0u;
    for (const spec::matrix_descriptor *matrix : selected) {
        if (matrix->row_begin != row_cursor) throw_format("matrix partitions do not cover rows contiguously");
        if (matrix->cols != cols) throw_format("matrix descriptor column count mismatch");
        if (matrix->rows > rows - row_cursor) throw_format("matrix descriptor row extent out of range");
        validate_matrix_sections(path, sections, *matrix);
        row_cursor += matrix->rows;
        nnz_sum += matrix->nnz;
    }
    if (row_cursor != rows) throw_format("matrix partitions do not cover all rows");
    if (nnz_sum != nnz) throw_format("matrix partition nnz sum mismatch");
}

exporting::csr_matrix_export read_rows_from_matrix_range(const std::string &path,
                                                         const std::vector<cshard_file::section_record> &sections,
                                                         const std::vector<spec::matrix_descriptor> &matrices,
                                                         std::uint32_t begin,
                                                         std::uint32_t count,
                                                         std::uint64_t cols,
                                                         std::uint64_t start,
                                                         std::uint64_t read_count,
                                                         const char *range_label) {
    if (begin > matrices.size() || count > matrices.size() - begin) throw_format("matrix descriptor range is out of bounds");
    std::vector<const spec::matrix_descriptor *> selected;
    selected.reserve(count);
    std::uint64_t total_rows = 0u;
    for (std::uint32_t i = 0u; i < count; ++i) {
        const spec::matrix_descriptor *matrix = &matrices[(std::size_t) begin + i];
        selected.push_back(matrix);
        total_rows += matrix->rows;
    }
    if (start > total_rows || read_count > total_rows - start) {
        throw std::out_of_range(std::string("cshard ") + range_label + " read range is outside the matrix");
    }
    std::sort(selected.begin(), selected.end(),
              [](const auto *lhs, const auto *rhs) { return lhs->row_begin < rhs->row_begin; });

    exporting::csr_matrix_export out;
    out.rows = read_count;
    out.cols = cols;
    out.indptr.assign((std::size_t) read_count + 1u, 0);
    for (std::uint64_t i = 0u; i < read_count; ++i) {
        const std::uint64_t global_row = start + i;
        auto it = std::upper_bound(selected.begin(), selected.end(), global_row,
                                   [](std::uint64_t row, const auto *matrix) { return row < matrix->row_begin; });
        if (it == selected.begin()) throw_format("failed to resolve row to matrix partition");
        const spec::matrix_descriptor &matrix = **(it - 1);
        const std::uint64_t local_row = global_row - matrix.row_begin;
        if (local_row >= matrix.rows) throw_format("resolved row outside matrix partition");
        const section_entry *a = find_section(sections, matrix.section_a_id);
        const section_entry *b = find_section(sections, matrix.section_b_id);
        const section_entry *c = find_section(sections, matrix.section_c_id);
        const section_entry *d = find_section(sections, matrix.section_d_id);
        if (a == nullptr) throw_format("matrix section missing while reading rows");
        if (matrix.layout == spec::matrix_layout_blocked_ell) {
            if (b == nullptr) throw_format("blocked-ELL section missing while reading rows");
            append_blocked_row_from_file(path, matrix, *a, *b, local_row, &out);
        } else if (matrix.layout == spec::matrix_layout_sliced_ell) {
            if (b == nullptr || c == nullptr || d == nullptr) throw_format("sliced-ELL section missing while reading rows");
            append_sliced_row_from_file(path, matrix, *a, *b, *c, *d, local_row, &out);
        } else if (matrix.layout == spec::matrix_layout_bucketed_blocked_ell) {
            append_optimized_blocked_row_from_file(path, matrix, *a, local_row, &out);
        } else if (matrix.layout == spec::matrix_layout_bucketed_sliced_ell) {
            append_optimized_sliced_row_from_file(path, matrix, *a, local_row, &out);
        } else if (matrix.layout == spec::matrix_layout_csr) {
            if (b == nullptr || c == nullptr) throw_format("CSR section missing while reading rows");
            append_csr_row_from_file(path, matrix, *a, *b, *c, local_row, &out);
        } else {
            throw_format("unsupported matrix layout while reading rows");
        }
        out.indptr[(std::size_t) i + 1u] = (std::int64_t) out.data.size();
    }
    return out;
}

std::pair<std::uint64_t, std::uint64_t> expected_local_range_for_window(const std::vector<std::uint32_t> &global_to_assay,
                                                                        std::uint64_t global_begin,
                                                                        std::uint64_t global_end) {
    bool seen = false;
    std::uint64_t begin = 0u, end = 0u, expected = 0u;
    if (global_begin > global_end || global_end > global_to_assay.size()) {
        throw std::runtime_error("optimized shard global row window is out of range");
    }
    for (std::uint64_t global = global_begin; global < global_end; ++global) {
        const std::uint32_t local = global_to_assay[(std::size_t) global];
        if (!dataset_assay_is_valid_row(local)) continue;
        if (!seen) {
            seen = true;
            begin = local;
            expected = local;
        }
        if (local != expected) {
            throw std::runtime_error("global shard window does not map to a contiguous assay-local row range");
        }
        ++expected;
    }
    if (!seen) return {0u, 0u};
    end = expected;
    return {begin, end};
}

void validate_window_matches(const std::vector<std::pair<std::uint64_t, std::uint64_t>> &reference,
                             std::size_t index,
                             std::uint64_t begin,
                             std::uint64_t end) {
    if (index >= reference.size()
        || reference[index].first != begin
        || reference[index].second != end) {
        throw std::runtime_error("assay optimized shard windows do not match the shared global sharding plan");
    }
}

void validate_reference_windows(const std::vector<std::pair<std::uint64_t, std::uint64_t>> &windows,
                                std::uint64_t global_rows) {
    std::uint64_t cursor = 0u;
    if (windows.empty()) throw std::runtime_error("optimized multi-assay cshard requires global shard windows");
    for (const auto &window : windows) {
        if (window.first != cursor || window.second <= window.first || window.second > global_rows) {
            throw std::runtime_error("optimized shard windows must cover global observations contiguously");
        }
        cursor = window.second;
    }
    if (cursor != global_rows) throw std::runtime_error("optimized shard windows do not cover all global observations");
}

void add_optimized_blob_matrix(file_plan *plan,
                               std::uint32_t matrix_id,
                               std::uint32_t partition_id,
                               std::uint32_t layout,
                               spec::section_kind section_kind,
                               std::uint64_t global_begin,
                               std::uint64_t global_end,
                               std::uint64_t local_begin,
                               const std::vector<std::uint8_t> &blob,
                               std::uint64_t rows,
                               std::uint64_t cols,
                               std::uint64_t nnz) {
    spec::matrix_descriptor desc{};
    desc.matrix_id = matrix_id;
    desc.partition_id = partition_id;
    desc.layout = layout;
    desc.value_dtype = spec::dtype_f16;
    desc.index_dtype = spec::dtype_u32;
    desc.row_begin = local_begin;
    desc.rows = rows;
    desc.cols = cols;
    desc.nnz = nnz;
    desc.aux0 = global_begin;
    desc.aux1 = global_end;
    desc.section_a_id = add_payload(plan,
                                    section_kind,
                                    spec::dtype_u8,
                                    blob,
                                    blob.size());
    plan->matrices.push_back(desc);
}

} // namespace

const char *layout_name(std::uint32_t layout) noexcept {
    switch (layout) {
    case spec::matrix_layout_blocked_ell: return "blocked_ell";
    case spec::matrix_layout_sliced_ell: return "sliced_ell";
    case spec::matrix_layout_csr: return "csr";
    case spec::matrix_layout_bucketed_blocked_ell: return "bucketed_blocked_ell";
    case spec::matrix_layout_bucketed_sliced_ell: return "bucketed_sliced_ell";
    default: return "unknown";
    }
}

table_view table_view::head(std::size_t count) const {
    table_view out;
    out.rows = std::min<std::uint64_t>(rows, count);
    out.columns.reserve(columns.size());
    for (const column &src : columns) {
        column dst;
        dst.name = src.name;
        dst.type = src.type;
        if (src.type == spec::table_column_text) {
            dst.text_values.assign(src.text_values.begin(), src.text_values.begin() + (std::ptrdiff_t) std::min<std::size_t>(src.text_values.size(), (std::size_t) out.rows));
        } else if (src.type == spec::table_column_float32) {
            dst.float32_values.assign(src.float32_values.begin(), src.float32_values.begin() + (std::ptrdiff_t) std::min<std::size_t>(src.float32_values.size(), (std::size_t) out.rows));
        } else if (src.type == spec::table_column_uint8) {
            dst.uint8_values.assign(src.uint8_values.begin(), src.uint8_values.begin() + (std::ptrdiff_t) std::min<std::size_t>(src.uint8_values.size(), (std::size_t) out.rows));
        }
        out.columns.push_back(std::move(dst));
    }
    return out;
}

cshard_file cshard_file::open(const std::string &path) {
    cshard_file out;
    std::ifstream in(path, std::ios::binary);
    if (!in) throw_format("failed to open file");
    in.read(reinterpret_cast<char *>(&out.header_), sizeof(out.header_));
    if (!in) throw_format("failed to read fixed header");
    out.path_ = path;
    const std::uint64_t actual_size = fs::file_size(path);
    if (out.header_.magic != spec::magic) throw_format("bad magic");
    if (out.header_.version_major != spec::version_major) throw_format("unsupported major version");
    if (out.header_.endian_tag != spec::endian_tag) throw_format("endian tag mismatch");
    if (out.header_.header_bytes != sizeof(spec::header)) throw_format("header size mismatch");
    if (out.header_.file_size != actual_size) throw_format("file size mismatch");
    if (out.header_.feature_order_hash == 0u) throw_format("missing required feature-order hash");
    if (out.header_.canonical_layout != spec::matrix_layout_blocked_ell
        && out.header_.canonical_layout != spec::matrix_layout_sliced_ell
        && out.header_.canonical_layout != spec::matrix_layout_csr
        && out.header_.canonical_layout != spec::matrix_layout_bucketed_blocked_ell
        && out.header_.canonical_layout != spec::matrix_layout_bucketed_sliced_ell) {
        throw_format("unknown canonical matrix layout");
    }

    {
        std::vector<section_entry> section_entries(out.header_.section_directory_count);
        if (!section_entries.empty()
            && !read_exact_at(path,
                             out.header_.section_directory_offset,
                             section_entries.data(),
                             section_entries.size())) {
            throw_format("failed to read section directory");
        }
        out.sections_.resize(section_entries.size());
        for (std::size_t i = 0u; i < section_entries.size(); ++i) out.sections_[i].entry = section_entries[i];
    }
    for (const auto &section : out.sections_) validate_section_bounds(out.header_, section.entry);

    out.matrices_.resize(out.header_.matrix_directory_count);
    if (!out.matrices_.empty()
        && !read_exact_at(path, out.header_.matrix_directory_offset, out.matrices_.data(), out.matrices_.size())) {
        throw_format("failed to read matrix directory");
    }

    std::vector<spec::table_column_descriptor> obs_columns(out.header_.obs_table_column_count);
    std::vector<spec::table_column_descriptor> var_columns(out.header_.var_table_column_count);
    if (!obs_columns.empty()
        && !read_exact_at(path, out.header_.obs_table_directory_offset, obs_columns.data(), obs_columns.size())) {
        throw_format("failed to read obs table directory");
    }
    if (!var_columns.empty()
        && !read_exact_at(path, out.header_.var_table_directory_offset, var_columns.data(), var_columns.size())) {
        throw_format("failed to read var table directory");
    }

    if (out.header_.assay_directory_count == 0u) {
        std::sort(out.matrices_.begin(), out.matrices_.end(),
                  [](const auto &lhs, const auto &rhs) { return lhs.row_begin < rhs.row_begin; });
        validate_matrix_range(path, out.sections_, out.matrices_, 0u, (std::uint32_t) out.matrices_.size(),
                              out.header_.rows, out.header_.cols, out.header_.nnz);
        out.obs_ = decode_table(path, out.sections_, obs_columns, out.header_.rows);
        out.var_ = decode_table(path, out.sections_, var_columns, out.header_.cols);
        if (feature_hash_from_table(out.var_) != 0u && feature_hash_from_table(out.var_) != out.header_.feature_order_hash) {
            throw_format("feature-order hash validation failed");
        }
    } else {
        if (out.header_.pairing_kind != dataset_pairing_exact_observation
            && out.header_.pairing_kind != dataset_pairing_partial_observation) {
            throw_format("unsupported multi-assay pairing kind");
        }
        out.global_observation_count_ = out.header_.global_observation_count;
        out.pairing_kind_ = out.header_.pairing_kind;
        if (out.global_observation_count_ == 0u) throw_format("multi-assay archive is missing global observations");
        out.obs_ = decode_table(path, out.sections_, obs_columns, out.global_observation_count_);

        spec::pairing_descriptor pairing{};
        if (!read_exact_at(path, out.header_.pairing_directory_offset, &pairing)) throw_format("failed to read pairing directory");
        if (pairing.pairing_kind != out.header_.pairing_kind
            || pairing.assay_count != out.header_.assay_directory_count
            || pairing.global_observation_count != out.global_observation_count_
            || pairing.assay_directory_begin != 0u
            || pairing.assay_directory_count != out.header_.assay_directory_count) {
            throw_format("pairing descriptor does not match header");
        }

        std::vector<spec::assay_descriptor> assay_records(out.header_.assay_directory_count);
        if (!read_exact_at(path, out.header_.assay_directory_offset, assay_records.data(), assay_records.size())) {
            throw_format("failed to read assay directory");
        }
        out.assays_.reserve(assay_records.size());
        std::unordered_map<std::string, bool> assay_ids;

        std::uint64_t nnz_sum = 0u;
        for (const spec::assay_descriptor &record : assay_records) {
            assay_description assay;
            assay.assay_id = record.assay_id;
            if (assay.assay_id.empty()) throw_format("assay descriptor has empty assay id");
            if (assay_ids.find(assay.assay_id) != assay_ids.end()) throw_format("duplicate assay id");
            assay_ids[assay.assay_id] = true;
            assay.semantics.modality = record.modality;
            assay.semantics.observation_unit = record.observation_unit;
            assay.semantics.feature_type = record.feature_type;
            assay.semantics.value_semantics = record.value_semantics;
            assay.semantics.processing_state = record.processing_state;
            assay.semantics.row_axis = record.row_axis;
            assay.semantics.col_axis = record.col_axis;
            assay.semantics.feature_namespace = record.feature_namespace;
            if (!dataset_assay_semantics_valid(&assay.semantics)) throw_format("assay semantics are invalid");
            if (record.global_observation_count != out.global_observation_count_) {
                throw_format("assay global observation count mismatch");
            }
            if (record.assay_row_count > std::numeric_limits<std::uint32_t>::max()
                || record.feature_count > std::numeric_limits<std::uint32_t>::max()) {
                throw_format("assay dimensions exceed cshard v1 row-map range");
            }
            assay.global_observation_count = record.global_observation_count;
            assay.rows = record.assay_row_count;
            assay.cols = record.feature_count;
            assay.nnz = record.nnz;
            assay.feature_order_hash = record.feature_order_hash;
            assay.matrix_descriptor_begin = record.matrix_descriptor_begin;
            assay.matrix_descriptor_count = record.matrix_descriptor_count;

            const section_entry *global_to_local = find_section(out.sections_, record.global_to_assay_rows_section_id);
            const section_entry *local_to_global = find_section(out.sections_, record.assay_to_global_rows_section_id);
            if (global_to_local == nullptr || local_to_global == nullptr) throw_format("assay row map references a missing section");
            if (global_to_local->kind != spec::section_kind_assay_global_to_local_rows
                || global_to_local->dtype != spec::dtype_u32
                || global_to_local->element_count != record.global_observation_count
                || global_to_local->bytes != record.global_observation_count * sizeof(std::uint32_t)) {
                throw_format("assay global-to-local row-map section metadata mismatch");
            }
            if (local_to_global->kind != spec::section_kind_assay_local_to_global_rows
                || local_to_global->dtype != spec::dtype_u32
                || local_to_global->element_count != record.assay_row_count
                || local_to_global->bytes != record.assay_row_count * sizeof(std::uint32_t)) {
                throw_format("assay local-to-global row-map section metadata mismatch");
            }
            assay.global_to_assay_row = read_array_section<std::uint32_t>(path, *global_to_local);
            assay.assay_row_to_global = read_array_section<std::uint32_t>(path, *local_to_global);
            dataset_assay_row_map_view row_map{};
            row_map.global_observation_count = (std::uint32_t) record.global_observation_count;
            row_map.assay_row_count = (std::uint32_t) record.assay_row_count;
            row_map.global_to_assay_row = assay.global_to_assay_row.data();
            row_map.assay_row_to_global = assay.assay_row_to_global.data();
            if (!dataset_validate_assay_row_map(&row_map)) throw_format("assay row map is invalid");

            assay.feature_table = decode_table_range(path,
                                                     out.sections_,
                                                     var_columns,
                                                     record.feature_table_column_begin,
                                                     record.feature_table_column_count,
                                                     record.feature_count);
            const std::uint64_t computed_feature_hash = feature_hash_from_table(assay.feature_table);
            if (computed_feature_hash != 0u && computed_feature_hash != record.feature_order_hash) {
                throw_format("assay feature-order hash validation failed");
            }
            validate_matrix_range(path,
                                  out.sections_,
                                  out.matrices_,
                                  record.matrix_descriptor_begin,
                                  record.matrix_descriptor_count,
                                  record.assay_row_count,
                                  record.feature_count,
                                  record.nnz);
            nnz_sum += record.nnz;
            out.assays_.push_back(std::move(assay));
        }
        if (nnz_sum != out.header_.nnz) throw_format("multi-assay nnz sum mismatch");
        std::vector<dataset_assay_view> pairing_views;
        pairing_views.reserve(out.assays_.size());
        for (const assay_description &assay : out.assays_) {
            dataset_assay_view view{};
            view.assay_id = assay.assay_id.c_str();
            view.semantics = assay.semantics;
            view.rows = assay.rows;
            view.cols = assay.cols;
            view.nnz = assay.nnz;
            view.feature_order_hash = assay.feature_order_hash;
            view.row_map.global_observation_count = (std::uint32_t) assay.global_observation_count;
            view.row_map.assay_row_count = (std::uint32_t) assay.rows;
            view.row_map.global_to_assay_row = assay.global_to_assay_row.data();
            view.row_map.assay_row_to_global = assay.assay_row_to_global.data();
            pairing_views.push_back(view);
        }
        dataset_pairing_view pairing_view{};
        pairing_view.pairing = out.header_.pairing_kind;
        pairing_view.assay_count = (std::uint32_t) pairing_views.size();
        pairing_view.assays = pairing_views.data();
        if (!dataset_validate_pairing_view(&pairing_view)) throw_format("multi-assay pairing is invalid");
    }
    out.canonical_layout_ = out.header_.canonical_layout;
    out.has_pack_manifest_ = out.header_.pack_manifest_count != 0u;
    return out;
}

bool cshard_file::validate(const std::string &path, std::string *error) {
    try {
        (void) cshard_file::open(path);
        return true;
    } catch (const std::exception &exc) {
        set_error(error, exc.what());
        return false;
    }
}

description cshard_file::describe() const {
    description out;
    out.path = path_;
    out.version_major = header_.version_major;
    out.version_minor = header_.version_minor;
    out.rows = header_.rows;
    out.cols = header_.cols;
    out.nnz = header_.nnz;
    out.partitions = matrices_.size();
    out.feature_order_hash = header_.feature_order_hash;
    out.canonical_layout = matrix_layout_to_string(header_.canonical_layout);
    out.has_pack_manifest = has_pack_manifest_;
    return out;
}

exporting::csr_matrix_export cshard_file::read_rows(std::uint64_t start, std::uint64_t count) const {
    if (is_multi_assay()) throw std::runtime_error("cshard read_rows is only valid for single-assay archives; use read_assay_rows");
    if (start > header_.rows || count > header_.rows - start) throw std::out_of_range("cshard read_rows range is outside the matrix");
    return read_rows_from_matrix_range(path_, sections_, matrices_, 0u, (std::uint32_t) matrices_.size(),
                                       header_.cols, start, count, "single-assay");
}

const assay_description& cshard_file::assay(std::uint32_t index) const {
    if (index >= assays_.size()) throw std::out_of_range("cshard assay index is out of range");
    return assays_[(std::size_t) index];
}

const assay_description* cshard_file::find_assay(const std::string &assay_id) const {
    for (const assay_description &entry : assays_) {
        if (entry.assay_id == assay_id) return &entry;
    }
    return nullptr;
}

paired_rows cshard_file::resolve_paired_rows(std::uint32_t global_observation) const {
    if (!is_multi_assay()) throw std::runtime_error("cshard resolve_paired_rows requires a multi-assay archive");
    if (pairing_kind_ != dataset_pairing_exact_observation && pairing_kind_ != dataset_pairing_partial_observation) {
        throw std::runtime_error("cshard pairing kind is not row-resolvable");
    }
    if (global_observation >= global_observation_count_) throw std::out_of_range("cshard global observation is out of range");
    paired_rows out;
    out.global_observation = global_observation;
    out.assay_rows.reserve(assays_.size());
    for (const assay_description &entry : assays_) {
        out.assay_rows.push_back(entry.global_to_assay_row[(std::size_t) global_observation]);
    }
    return out;
}

exporting::csr_matrix_export cshard_file::read_assay_rows(const std::string &assay_id,
                                                          std::uint64_t start,
                                                          std::uint64_t count) const {
    const assay_description *entry = find_assay(assay_id);
    if (entry == nullptr) throw std::out_of_range("cshard assay id was not found");
    if (start > entry->rows || count > entry->rows - start) throw std::out_of_range("cshard assay row range is outside the matrix");
    return read_rows_from_matrix_range(path_,
                                       sections_,
                                       matrices_,
                                       entry->matrix_descriptor_begin,
                                       entry->matrix_descriptor_count,
                                       entry->cols,
                                       start,
                                       count,
                                       "assay");
}

multi_assay_description cshard_file::multi_assay() const {
    multi_assay_description out;
    out.global_observation_count = global_observation_count_;
    out.pairing_kind = pairing_kind_;
    out.global_obs = obs_;
    out.assays = assays_;
    return out;
}

bool write_csr(const std::string &path,
               const exporting::csr_matrix_export &csr,
               const table_view &obs,
               const table_view &var,
               const writer_options &options,
               std::string *error) {
    try {
        validate_csr_for_write(csr);
        file_plan plan = base_plan(csr.rows, csr.cols, csr.data.size(), spec::matrix_layout_csr, obs, var, options);
        (void) add_csr_matrix(&plan, csr, 0u, 0u, 0u);
        return write_plan(path, &plan, error);
    } catch (const std::exception &exc) {
        set_error(error, exc.what());
        return false;
    }
}

bool write_multi_assay_csr(const std::string &path,
                           const table_view &global_obs,
                           const std::vector<csr_assay_input> &assays,
                           std::uint32_t pairing_kind,
                           const writer_options &options,
                           std::string *error) {
    try {
        if (assays.empty()) throw std::runtime_error("multi-assay cshard requires at least one assay");
        if (global_obs.rows == 0u) throw std::runtime_error("multi-assay cshard requires at least one global observation");
        if (global_obs.rows > std::numeric_limits<std::uint32_t>::max()) {
            throw std::runtime_error("global observation count exceeds cshard v1 row-map range");
        }
        if (pairing_kind != dataset_pairing_exact_observation
            && pairing_kind != dataset_pairing_partial_observation) {
            throw std::runtime_error("multi-assay cshard writer supports exact and partial observation pairing only");
        }
        validate_table_for_write(global_obs, global_obs.rows, "global obs");

        file_plan plan;
        plan.header.rows = global_obs.rows;
        plan.header.cols = 0u;
        plan.header.canonical_layout = spec::matrix_layout_csr;
        plan.header.global_observation_count = global_obs.rows;
        plan.header.pairing_kind = pairing_kind;
        add_metadata(&plan, "format_role", "experimental_standby_archive");
        add_metadata(&plan, "schema", "cshard_v1");
        add_metadata(&plan, "matrix_layout", layout_name(spec::matrix_layout_csr));
        add_metadata(&plan, "multi_assay", "true");
        add_table_columns(&plan, global_obs, false, &plan.obs_columns);

        std::unordered_map<std::string, bool> assay_ids;
        std::vector<dataset_assay_view> pairing_assays;
        pairing_assays.reserve(assays.size());
        std::uint64_t combined_feature_hash = 1469598103934665603ull;

        for (std::size_t assay_index = 0u; assay_index < assays.size(); ++assay_index) {
            const csr_assay_input &src = assays[assay_index];
            if (src.assay_id.empty()) throw std::runtime_error("assay id must not be empty");
            if (src.assay_id.size() >= 32u) throw std::runtime_error("assay id exceeds cshard v1 descriptor width");
            if (assay_ids.find(src.assay_id) != assay_ids.end()) throw std::runtime_error("duplicate assay id");
            assay_ids[src.assay_id] = true;
            if (!dataset_assay_semantics_valid(&src.semantics)) throw std::runtime_error("assay semantics are invalid");
            validate_csr_for_write(src.matrix);
            validate_table_for_write(src.features, src.matrix.cols, "assay feature");
            if (src.global_to_assay_row.size() != global_obs.rows) throw std::runtime_error("assay global-to-local row map length mismatch");
            if (src.assay_row_to_global.size() != src.matrix.rows) throw std::runtime_error("assay local-to-global row map length mismatch");

            dataset_assay_row_map_view row_map{};
            row_map.global_observation_count = (std::uint32_t) global_obs.rows;
            row_map.assay_row_count = (std::uint32_t) src.matrix.rows;
            row_map.global_to_assay_row = src.global_to_assay_row.data();
            row_map.assay_row_to_global = src.assay_row_to_global.data();
            if (!dataset_validate_assay_row_map(&row_map)) throw std::runtime_error("assay row map is invalid");

            const std::uint64_t assay_feature_hash = src.feature_order_hash != 0u
                ? src.feature_order_hash
                : feature_hash_from_table(src.features);
            if (assay_feature_hash == 0u) throw std::runtime_error("assay feature-order hash could not be computed");

            spec::assay_descriptor desc{};
            copy_fixed(desc.assay_id, sizeof(desc.assay_id), src.assay_id);
            desc.modality = src.semantics.modality;
            desc.observation_unit = src.semantics.observation_unit;
            desc.feature_type = src.semantics.feature_type;
            desc.value_semantics = src.semantics.value_semantics;
            desc.processing_state = src.semantics.processing_state;
            desc.row_axis = src.semantics.row_axis;
            desc.col_axis = src.semantics.col_axis;
            desc.feature_namespace = src.semantics.feature_namespace;
            desc.global_observation_count = global_obs.rows;
            desc.assay_row_count = src.matrix.rows;
            desc.feature_count = src.matrix.cols;
            desc.nnz = src.matrix.data.size();
            desc.feature_order_hash = assay_feature_hash;
            desc.matrix_descriptor_begin = (std::uint32_t) plan.matrices.size();
            desc.matrix_descriptor_count = 1u;
            const auto feature_range = append_table_columns(&plan, src.features, &plan.var_columns);
            desc.feature_table_column_begin = feature_range.first;
            desc.feature_table_column_count = feature_range.second;
            desc.global_to_assay_rows_section_id =
                add_payload(&plan,
                            spec::section_kind_assay_global_to_local_rows,
                            spec::dtype_u32,
                            bytes_from_array(src.global_to_assay_row.data(), src.global_to_assay_row.size()),
                            src.global_to_assay_row.size());
            desc.assay_to_global_rows_section_id =
                add_payload(&plan,
                            spec::section_kind_assay_local_to_global_rows,
                            spec::dtype_u32,
                            bytes_from_array(src.assay_row_to_global.data(), src.assay_row_to_global.size()),
                            src.assay_row_to_global.size());
            (void) add_csr_matrix(&plan, src.matrix, (std::uint32_t) assay_index, 0u, 0u);
            plan.assays.push_back(desc);
            plan.header.cols += src.matrix.cols;
            plan.header.nnz += src.matrix.data.size();

            dataset_assay_view pairing_assay{};
            pairing_assay.assay_id = src.assay_id.c_str();
            pairing_assay.semantics = src.semantics;
            pairing_assay.rows = src.matrix.rows;
            pairing_assay.cols = src.matrix.cols;
            pairing_assay.nnz = src.matrix.data.size();
            pairing_assay.feature_order_hash = assay_feature_hash;
            pairing_assay.row_map = row_map;
            pairing_assays.push_back(pairing_assay);

            combined_feature_hash = fnv1a64(src.assay_id.data(), src.assay_id.size(), combined_feature_hash);
            combined_feature_hash = fnv1a64(&assay_feature_hash, sizeof(assay_feature_hash), combined_feature_hash);
        }

        dataset_pairing_view pairing{};
        pairing.pairing = pairing_kind;
        pairing.assay_count = (std::uint32_t) pairing_assays.size();
        pairing.assays = pairing_assays.data();
        if (!dataset_validate_pairing_view(&pairing)) throw std::runtime_error("multi-assay pairing is invalid");

        spec::pairing_descriptor pairing_desc{};
        pairing_desc.pairing_kind = pairing_kind;
        pairing_desc.assay_count = (std::uint32_t) assays.size();
        pairing_desc.global_observation_count = global_obs.rows;
        pairing_desc.assay_directory_begin = 0u;
        pairing_desc.assay_directory_count = (std::uint32_t) assays.size();
        plan.pairings.push_back(pairing_desc);
        plan.header.feature_order_hash = options.feature_order_hash != 0u ? options.feature_order_hash : combined_feature_hash;
        add_metadata(&plan, "feature_order_hash", std::to_string(plan.header.feature_order_hash));
        return write_plan(path, &plan, error);
    } catch (const std::exception &exc) {
        set_error(error, exc.what());
        return false;
    }
}

bool write_multi_assay_optimized_blocked_ell(const std::string &path,
                                             const table_view &global_obs,
                                             const std::vector<optimized_blocked_ell_assay_input> &assays,
                                             std::uint32_t pairing_kind,
                                             const writer_options &options,
                                             std::string *error) {
    try {
        if (assays.empty()) throw std::runtime_error("optimized multi-assay cshard requires at least one assay");
        if (global_obs.rows == 0u) throw std::runtime_error("optimized multi-assay cshard requires at least one global observation");
        if (global_obs.rows > std::numeric_limits<std::uint32_t>::max()) {
            throw std::runtime_error("global observation count exceeds cshard v1 row-map range");
        }
        if (pairing_kind != dataset_pairing_exact_observation
            && pairing_kind != dataset_pairing_partial_observation) {
            throw std::runtime_error("optimized multi-assay cshard writer supports exact and partial observation pairing only");
        }
        if (assays[0].shards.empty()) throw std::runtime_error("optimized multi-assay cshard requires at least one global shard window");
        validate_table_for_write(global_obs, global_obs.rows, "global obs");

        std::vector<std::pair<std::uint64_t, std::uint64_t>> reference_windows;
        reference_windows.reserve(assays[0].shards.size());
        for (const auto &shard : assays[0].shards) reference_windows.emplace_back(shard.global_row_begin, shard.global_row_end);
        validate_reference_windows(reference_windows, global_obs.rows);

        file_plan plan;
        plan.header.rows = global_obs.rows;
        plan.header.cols = 0u;
        plan.header.canonical_layout = spec::matrix_layout_bucketed_blocked_ell;
        plan.header.global_observation_count = global_obs.rows;
        plan.header.pairing_kind = pairing_kind;
        add_metadata(&plan, "format_role", "experimental_standby_archive");
        add_metadata(&plan, "schema", "cshard_v1");
        add_metadata(&plan, "matrix_layout", layout_name(spec::matrix_layout_bucketed_blocked_ell));
        add_metadata(&plan, "multi_assay", "true");
        add_table_columns(&plan, global_obs, false, &plan.obs_columns);

        std::unordered_map<std::string, bool> assay_ids;
        std::vector<dataset_assay_view> pairing_assays;
        pairing_assays.reserve(assays.size());
        std::uint64_t combined_feature_hash = 1469598103934665603ull;

        for (std::size_t assay_index = 0u; assay_index < assays.size(); ++assay_index) {
            const optimized_blocked_ell_assay_input &src = assays[assay_index];
            if (src.assay_id.empty()) throw std::runtime_error("assay id must not be empty");
            if (src.assay_id.size() >= 32u) throw std::runtime_error("assay id exceeds cshard v1 descriptor width");
            if (assay_ids.find(src.assay_id) != assay_ids.end()) throw std::runtime_error("duplicate assay id");
            assay_ids[src.assay_id] = true;
            if (!dataset_assay_semantics_valid(&src.semantics)) throw std::runtime_error("assay semantics are invalid");
            validate_table_for_write(src.features, src.features.rows, "assay feature");
            if (src.global_to_assay_row.size() != global_obs.rows) throw std::runtime_error("assay global-to-local row map length mismatch");
            if (src.assay_row_to_global.size() > std::numeric_limits<std::uint32_t>::max()
                || src.features.rows > std::numeric_limits<std::uint32_t>::max()) {
                throw std::runtime_error("assay dimensions exceed cshard v1 descriptor range");
            }
            if (src.shards.size() != reference_windows.size()) {
                throw std::runtime_error("assay optimized shard count does not match the shared global sharding plan");
            }

            dataset_assay_row_map_view row_map{};
            row_map.global_observation_count = (std::uint32_t) global_obs.rows;
            row_map.assay_row_count = (std::uint32_t) src.assay_row_to_global.size();
            row_map.global_to_assay_row = src.global_to_assay_row.data();
            row_map.assay_row_to_global = src.assay_row_to_global.data();
            if (!dataset_validate_assay_row_map(&row_map)) throw std::runtime_error("assay row map is invalid");

            const std::uint64_t assay_feature_hash = src.feature_order_hash != 0u
                ? src.feature_order_hash
                : feature_hash_from_table(src.features);
            if (assay_feature_hash == 0u) throw std::runtime_error("assay feature-order hash could not be computed");

            spec::assay_descriptor desc{};
            copy_fixed(desc.assay_id, sizeof(desc.assay_id), src.assay_id);
            desc.modality = src.semantics.modality;
            desc.observation_unit = src.semantics.observation_unit;
            desc.feature_type = src.semantics.feature_type;
            desc.value_semantics = src.semantics.value_semantics;
            desc.processing_state = src.semantics.processing_state;
            desc.row_axis = src.semantics.row_axis;
            desc.col_axis = src.semantics.col_axis;
            desc.feature_namespace = src.semantics.feature_namespace;
            desc.global_observation_count = global_obs.rows;
            desc.assay_row_count = src.assay_row_to_global.size();
            desc.feature_count = src.features.rows;
            desc.feature_order_hash = assay_feature_hash;
            desc.matrix_descriptor_begin = (std::uint32_t) plan.matrices.size();
            desc.matrix_descriptor_count = (std::uint32_t) src.shards.size();
            const auto feature_range = append_table_columns(&plan, src.features, &plan.var_columns);
            desc.feature_table_column_begin = feature_range.first;
            desc.feature_table_column_count = feature_range.second;
            desc.global_to_assay_rows_section_id =
                add_payload(&plan,
                            spec::section_kind_assay_global_to_local_rows,
                            spec::dtype_u32,
                            bytes_from_array(src.global_to_assay_row.data(), src.global_to_assay_row.size()),
                            src.global_to_assay_row.size());
            desc.assay_to_global_rows_section_id =
                add_payload(&plan,
                            spec::section_kind_assay_local_to_global_rows,
                            spec::dtype_u32,
                            bytes_from_array(src.assay_row_to_global.data(), src.assay_row_to_global.size()),
                            src.assay_row_to_global.size());

            for (std::size_t shard_index = 0u; shard_index < src.shards.size(); ++shard_index) {
                const optimized_blocked_ell_shard_input &shard_input = src.shards[shard_index];
                if (shard_input.assay_id != src.assay_id) throw std::runtime_error("optimized shard assay id does not match assay descriptor");
                validate_window_matches(reference_windows, shard_index, shard_input.global_row_begin, shard_input.global_row_end);
                const auto expected_range = expected_local_range_for_window(src.global_to_assay_row,
                                                                            shard_input.global_row_begin,
                                                                            shard_input.global_row_end);
                if (expected_range.first != shard_input.local_row_begin || expected_range.second != shard_input.local_row_end) {
                    throw std::runtime_error("optimized shard local row range does not match assay row map");
                }
                std::vector<std::uint8_t> blob = shard_input.shard != nullptr
                    ? serialize_blocked_shard_blob(shard_input.shard)
                    : blob_from_input(shard_input.serialized_blob, shard_input.serialized_blob_bytes);
                const auto shape = shard_input.shard != nullptr
                    ? optimized_shard_shape<bucketed_blocked_ell_shard>{shard_input.shard->rows, shard_input.shard->cols, shard_input.shard->nnz}
                    : blocked_blob_shape(blob);
                const std::uint64_t rows = shard_input.local_row_end - shard_input.local_row_begin;
                if (shape.rows != rows || shape.cols != src.features.rows) {
                    throw std::runtime_error("optimized Blocked-ELL shard shape does not match assay descriptor range");
                }
                add_optimized_blob_matrix(&plan,
                                          (std::uint32_t) assay_index,
                                          (std::uint32_t) shard_index,
                                          spec::matrix_layout_bucketed_blocked_ell,
                                          spec::section_kind_optimized_blocked_ell_shard_blob,
                                          shard_input.global_row_begin,
                                          shard_input.global_row_end,
                                          shard_input.local_row_begin,
                                          blob,
                                          shape.rows,
                                          shape.cols,
                                          shape.nnz);
                desc.nnz += shape.nnz;
            }

            plan.assays.push_back(desc);
            plan.header.cols += src.features.rows;
            plan.header.nnz += desc.nnz;

            dataset_assay_view pairing_assay{};
            pairing_assay.assay_id = src.assay_id.c_str();
            pairing_assay.semantics = src.semantics;
            pairing_assay.rows = src.assay_row_to_global.size();
            pairing_assay.cols = src.features.rows;
            pairing_assay.nnz = desc.nnz;
            pairing_assay.feature_order_hash = assay_feature_hash;
            pairing_assay.row_map = row_map;
            pairing_assays.push_back(pairing_assay);

            combined_feature_hash = fnv1a64(src.assay_id.data(), src.assay_id.size(), combined_feature_hash);
            combined_feature_hash = fnv1a64(&assay_feature_hash, sizeof(assay_feature_hash), combined_feature_hash);
        }

        dataset_pairing_view pairing{};
        pairing.pairing = pairing_kind;
        pairing.assay_count = (std::uint32_t) pairing_assays.size();
        pairing.assays = pairing_assays.data();
        if (!dataset_validate_pairing_view(&pairing)) throw std::runtime_error("multi-assay pairing is invalid");

        spec::pairing_descriptor pairing_desc{};
        pairing_desc.pairing_kind = pairing_kind;
        pairing_desc.assay_count = (std::uint32_t) assays.size();
        pairing_desc.global_observation_count = global_obs.rows;
        pairing_desc.assay_directory_begin = 0u;
        pairing_desc.assay_directory_count = (std::uint32_t) assays.size();
        plan.pairings.push_back(pairing_desc);
        plan.header.feature_order_hash = options.feature_order_hash != 0u ? options.feature_order_hash : combined_feature_hash;
        add_metadata(&plan, "feature_order_hash", std::to_string(plan.header.feature_order_hash));
        return write_plan(path, &plan, error);
    } catch (const std::exception &exc) {
        set_error(error, exc.what());
        return false;
    }
}

bool write_multi_assay_optimized_sliced_ell(const std::string &path,
                                            const table_view &global_obs,
                                            const std::vector<optimized_sliced_ell_assay_input> &assays,
                                            std::uint32_t pairing_kind,
                                            const writer_options &options,
                                            std::string *error) {
    try {
        if (assays.empty()) throw std::runtime_error("optimized multi-assay cshard requires at least one assay");
        if (global_obs.rows == 0u) throw std::runtime_error("optimized multi-assay cshard requires at least one global observation");
        if (global_obs.rows > std::numeric_limits<std::uint32_t>::max()) {
            throw std::runtime_error("global observation count exceeds cshard v1 row-map range");
        }
        if (pairing_kind != dataset_pairing_exact_observation
            && pairing_kind != dataset_pairing_partial_observation) {
            throw std::runtime_error("optimized multi-assay cshard writer supports exact and partial observation pairing only");
        }
        if (assays[0].shards.empty()) throw std::runtime_error("optimized multi-assay cshard requires at least one global shard window");
        validate_table_for_write(global_obs, global_obs.rows, "global obs");

        std::vector<std::pair<std::uint64_t, std::uint64_t>> reference_windows;
        reference_windows.reserve(assays[0].shards.size());
        for (const auto &shard : assays[0].shards) reference_windows.emplace_back(shard.global_row_begin, shard.global_row_end);
        validate_reference_windows(reference_windows, global_obs.rows);

        file_plan plan;
        plan.header.rows = global_obs.rows;
        plan.header.cols = 0u;
        plan.header.canonical_layout = spec::matrix_layout_bucketed_sliced_ell;
        plan.header.global_observation_count = global_obs.rows;
        plan.header.pairing_kind = pairing_kind;
        add_metadata(&plan, "format_role", "experimental_standby_archive");
        add_metadata(&plan, "schema", "cshard_v1");
        add_metadata(&plan, "matrix_layout", layout_name(spec::matrix_layout_bucketed_sliced_ell));
        add_metadata(&plan, "multi_assay", "true");
        add_table_columns(&plan, global_obs, false, &plan.obs_columns);

        std::unordered_map<std::string, bool> assay_ids;
        std::vector<dataset_assay_view> pairing_assays;
        pairing_assays.reserve(assays.size());
        std::uint64_t combined_feature_hash = 1469598103934665603ull;

        for (std::size_t assay_index = 0u; assay_index < assays.size(); ++assay_index) {
            const optimized_sliced_ell_assay_input &src = assays[assay_index];
            if (src.assay_id.empty()) throw std::runtime_error("assay id must not be empty");
            if (src.assay_id.size() >= 32u) throw std::runtime_error("assay id exceeds cshard v1 descriptor width");
            if (assay_ids.find(src.assay_id) != assay_ids.end()) throw std::runtime_error("duplicate assay id");
            assay_ids[src.assay_id] = true;
            if (!dataset_assay_semantics_valid(&src.semantics)) throw std::runtime_error("assay semantics are invalid");
            validate_table_for_write(src.features, src.features.rows, "assay feature");
            if (src.global_to_assay_row.size() != global_obs.rows) throw std::runtime_error("assay global-to-local row map length mismatch");
            if (src.assay_row_to_global.size() > std::numeric_limits<std::uint32_t>::max()
                || src.features.rows > std::numeric_limits<std::uint32_t>::max()) {
                throw std::runtime_error("assay dimensions exceed cshard v1 descriptor range");
            }
            if (src.shards.size() != reference_windows.size()) {
                throw std::runtime_error("assay optimized shard count does not match the shared global sharding plan");
            }

            dataset_assay_row_map_view row_map{};
            row_map.global_observation_count = (std::uint32_t) global_obs.rows;
            row_map.assay_row_count = (std::uint32_t) src.assay_row_to_global.size();
            row_map.global_to_assay_row = src.global_to_assay_row.data();
            row_map.assay_row_to_global = src.assay_row_to_global.data();
            if (!dataset_validate_assay_row_map(&row_map)) throw std::runtime_error("assay row map is invalid");

            const std::uint64_t assay_feature_hash = src.feature_order_hash != 0u
                ? src.feature_order_hash
                : feature_hash_from_table(src.features);
            if (assay_feature_hash == 0u) throw std::runtime_error("assay feature-order hash could not be computed");

            spec::assay_descriptor desc{};
            copy_fixed(desc.assay_id, sizeof(desc.assay_id), src.assay_id);
            desc.modality = src.semantics.modality;
            desc.observation_unit = src.semantics.observation_unit;
            desc.feature_type = src.semantics.feature_type;
            desc.value_semantics = src.semantics.value_semantics;
            desc.processing_state = src.semantics.processing_state;
            desc.row_axis = src.semantics.row_axis;
            desc.col_axis = src.semantics.col_axis;
            desc.feature_namespace = src.semantics.feature_namespace;
            desc.global_observation_count = global_obs.rows;
            desc.assay_row_count = src.assay_row_to_global.size();
            desc.feature_count = src.features.rows;
            desc.feature_order_hash = assay_feature_hash;
            desc.matrix_descriptor_begin = (std::uint32_t) plan.matrices.size();
            desc.matrix_descriptor_count = (std::uint32_t) src.shards.size();
            const auto feature_range = append_table_columns(&plan, src.features, &plan.var_columns);
            desc.feature_table_column_begin = feature_range.first;
            desc.feature_table_column_count = feature_range.second;
            desc.global_to_assay_rows_section_id =
                add_payload(&plan,
                            spec::section_kind_assay_global_to_local_rows,
                            spec::dtype_u32,
                            bytes_from_array(src.global_to_assay_row.data(), src.global_to_assay_row.size()),
                            src.global_to_assay_row.size());
            desc.assay_to_global_rows_section_id =
                add_payload(&plan,
                            spec::section_kind_assay_local_to_global_rows,
                            spec::dtype_u32,
                            bytes_from_array(src.assay_row_to_global.data(), src.assay_row_to_global.size()),
                            src.assay_row_to_global.size());

            for (std::size_t shard_index = 0u; shard_index < src.shards.size(); ++shard_index) {
                const optimized_sliced_ell_shard_input &shard_input = src.shards[shard_index];
                if (shard_input.assay_id != src.assay_id) throw std::runtime_error("optimized shard assay id does not match assay descriptor");
                validate_window_matches(reference_windows, shard_index, shard_input.global_row_begin, shard_input.global_row_end);
                const auto expected_range = expected_local_range_for_window(src.global_to_assay_row,
                                                                            shard_input.global_row_begin,
                                                                            shard_input.global_row_end);
                if (expected_range.first != shard_input.local_row_begin || expected_range.second != shard_input.local_row_end) {
                    throw std::runtime_error("optimized shard local row range does not match assay row map");
                }
                std::vector<std::uint8_t> blob = shard_input.shard != nullptr
                    ? serialize_sliced_shard_blob(shard_input.shard)
                    : blob_from_input(shard_input.serialized_blob, shard_input.serialized_blob_bytes);
                const auto shape = shard_input.shard != nullptr
                    ? optimized_shard_shape<bucketed_sliced_ell_shard>{shard_input.shard->rows, shard_input.shard->cols, shard_input.shard->nnz}
                    : sliced_blob_shape(blob);
                const std::uint64_t rows = shard_input.local_row_end - shard_input.local_row_begin;
                if (shape.rows != rows || shape.cols != src.features.rows) {
                    throw std::runtime_error("optimized Sliced-ELL shard shape does not match assay descriptor range");
                }
                add_optimized_blob_matrix(&plan,
                                          (std::uint32_t) assay_index,
                                          (std::uint32_t) shard_index,
                                          spec::matrix_layout_bucketed_sliced_ell,
                                          spec::section_kind_optimized_sliced_ell_shard_blob,
                                          shard_input.global_row_begin,
                                          shard_input.global_row_end,
                                          shard_input.local_row_begin,
                                          blob,
                                          shape.rows,
                                          shape.cols,
                                          shape.nnz);
                desc.nnz += shape.nnz;
            }

            plan.assays.push_back(desc);
            plan.header.cols += src.features.rows;
            plan.header.nnz += desc.nnz;

            dataset_assay_view pairing_assay{};
            pairing_assay.assay_id = src.assay_id.c_str();
            pairing_assay.semantics = src.semantics;
            pairing_assay.rows = src.assay_row_to_global.size();
            pairing_assay.cols = src.features.rows;
            pairing_assay.nnz = desc.nnz;
            pairing_assay.feature_order_hash = assay_feature_hash;
            pairing_assay.row_map = row_map;
            pairing_assays.push_back(pairing_assay);

            combined_feature_hash = fnv1a64(src.assay_id.data(), src.assay_id.size(), combined_feature_hash);
            combined_feature_hash = fnv1a64(&assay_feature_hash, sizeof(assay_feature_hash), combined_feature_hash);
        }

        dataset_pairing_view pairing{};
        pairing.pairing = pairing_kind;
        pairing.assay_count = (std::uint32_t) pairing_assays.size();
        pairing.assays = pairing_assays.data();
        if (!dataset_validate_pairing_view(&pairing)) throw std::runtime_error("multi-assay pairing is invalid");

        spec::pairing_descriptor pairing_desc{};
        pairing_desc.pairing_kind = pairing_kind;
        pairing_desc.assay_count = (std::uint32_t) assays.size();
        pairing_desc.global_observation_count = global_obs.rows;
        pairing_desc.assay_directory_begin = 0u;
        pairing_desc.assay_directory_count = (std::uint32_t) assays.size();
        plan.pairings.push_back(pairing_desc);
        plan.header.feature_order_hash = options.feature_order_hash != 0u ? options.feature_order_hash : combined_feature_hash;
        add_metadata(&plan, "feature_order_hash", std::to_string(plan.header.feature_order_hash));
        return write_plan(path, &plan, error);
    } catch (const std::exception &exc) {
        set_error(error, exc.what());
        return false;
    }
}

bool convert_csh5_to_cshard(const std::string &input_path,
                            const std::string &output_path,
                            const writer_options &options,
                            std::string *error) {
    exporting::dataset_summary summary;
    std::vector<exporting::observation_metadata_column> obs_columns;
    std::vector<exporting::annotation_column> var_columns;
    if (!exporting::load_dataset_summary(input_path.c_str(), &summary, error)) return false;
    (void) exporting::load_observation_metadata(input_path.c_str(), &obs_columns, nullptr);
    (void) exporting::load_feature_metadata(input_path.c_str(), &var_columns, nullptr);
    const table_view obs = table_from_summary_obs(summary, obs_columns);
    const table_view var = table_from_summary_var(summary, var_columns);

    if (summary.matrix_format == "blocked_ell") {
        sharded<sparse::blocked_ell> view;
        shard_storage storage;
        init(&view);
        init(&storage);
        const auto cleanup = [&]() {
            clear(&storage);
            clear(&view);
        };
        if (!load_header(input_path.c_str(), &view, &storage)) {
            cleanup();
            set_error(error, "failed to load input .csh5 blocked-ELL header");
            return false;
        }
        file_plan plan = base_plan(view.rows, view.cols, view.nnz, spec::matrix_layout_blocked_ell, obs, var, options);
        for (unsigned long partition_id = 0u; partition_id < view.num_partitions; ++partition_id) {
            if (!fetch_partition(&view, &storage, partition_id)) {
                cleanup();
                set_error(error, "failed to fetch blocked-ELL partition from input .csh5");
                return false;
            }
            if (!add_blocked_part(&plan,
                                  (std::uint32_t) partition_id,
                                  first_row_in_partition(&view, partition_id),
                                  view.parts[partition_id],
                                  error)) {
                cleanup();
                return false;
            }
            (void) drop_partition(&view, partition_id);
        }
        cleanup();
        return write_plan(output_path, &plan, error);
    }

    if (summary.matrix_format == "sliced_ell") {
        sharded<sparse::sliced_ell> view;
        shard_storage storage;
        init(&view);
        init(&storage);
        const auto cleanup = [&]() {
            clear(&storage);
            clear(&view);
        };
        if (!load_header(input_path.c_str(), &view, &storage)) {
            cleanup();
            set_error(error, "failed to load input .csh5 sliced-ELL header");
            return false;
        }
        file_plan plan = base_plan(view.rows, view.cols, view.nnz, spec::matrix_layout_sliced_ell, obs, var, options);
        for (unsigned long partition_id = 0u; partition_id < view.num_partitions; ++partition_id) {
            if (!fetch_partition(&view, &storage, partition_id)) {
                cleanup();
                set_error(error, "failed to fetch sliced-ELL partition from input .csh5");
                return false;
            }
            if (!add_sliced_part(&plan,
                                 (std::uint32_t) partition_id,
                                 first_row_in_partition(&view, partition_id),
                                 view.parts[partition_id],
                                 error)) {
                cleanup();
                return false;
            }
            (void) drop_partition(&view, partition_id);
        }
        cleanup();
        return write_plan(output_path, &plan, error);
    }

    set_error(error, "unsupported input .csh5 matrix_format for .cshard conversion: " + summary.matrix_format);
    return false;
}

} // namespace cellshard::cshard
