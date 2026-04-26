#include "../../../include/CellShard/io/cshard.hh"

#include "../../../include/CellShard/formats/blocked_ell.cuh"
#include "../../../include/CellShard/formats/sliced_ell.cuh"
#include "../../../include/CellShard/io/csh5/api.cuh"
#include "../../../include/CellShard/runtime/host/sharded_host.cuh"
#include "../../../include/CellShard/runtime/storage/disk.cuh"

#include <algorithm>
#include <array>
#include <cstring>
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

} // namespace

const char *layout_name(std::uint32_t layout) noexcept {
    switch (layout) {
    case spec::matrix_layout_blocked_ell: return "blocked_ell";
    case spec::matrix_layout_sliced_ell: return "sliced_ell";
    case spec::matrix_layout_csr: return "csr";
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
        && out.header_.canonical_layout != spec::matrix_layout_csr) {
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
    std::sort(out.matrices_.begin(), out.matrices_.end(),
              [](const auto &lhs, const auto &rhs) { return lhs.row_begin < rhs.row_begin; });
    std::uint64_t row_cursor = 0u;
    std::uint64_t nnz_sum = 0u;
    for (const auto &matrix : out.matrices_) {
        if (matrix.row_begin != row_cursor) throw_format("matrix partitions do not cover rows contiguously");
        if (matrix.cols != out.header_.cols) throw_format("matrix descriptor column count mismatch");
        if (matrix.rows > out.header_.rows - row_cursor) throw_format("matrix descriptor row extent out of range");
        if (matrix.value_dtype != spec::dtype_f16 || matrix.index_dtype != spec::dtype_u32) {
            throw_format("unsupported matrix descriptor dtype");
        }
        if (find_section(out.sections_, matrix.section_a_id) == nullptr || find_section(out.sections_, matrix.section_b_id) == nullptr) {
            throw_format("matrix descriptor references a missing section");
        }
        if ((matrix.layout == spec::matrix_layout_sliced_ell || matrix.layout == spec::matrix_layout_csr)
            && find_section(out.sections_, matrix.section_c_id) == nullptr) {
            throw_format("matrix descriptor references a missing third section");
        }
        if (matrix.layout == spec::matrix_layout_sliced_ell && find_section(out.sections_, matrix.section_d_id) == nullptr) {
            throw_format("sliced-ELL descriptor references a missing value section");
        }
        row_cursor += matrix.rows;
        nnz_sum += matrix.nnz;
    }
    if (row_cursor != out.header_.rows) throw_format("matrix partitions do not cover all rows");
    if (nnz_sum != out.header_.nnz) throw_format("matrix partition nnz sum mismatch");

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
    out.obs_ = decode_table(path, out.sections_, obs_columns, out.header_.rows);
    out.var_ = decode_table(path, out.sections_, var_columns, out.header_.cols);
    if (feature_hash_from_table(out.var_) != 0u && feature_hash_from_table(out.var_) != out.header_.feature_order_hash) {
        throw_format("feature-order hash validation failed");
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
    exporting::csr_matrix_export out;
    if (start > header_.rows || count > header_.rows - start) throw std::out_of_range("cshard read_rows range is outside the matrix");
    out.rows = count;
    out.cols = header_.cols;
    out.indptr.assign((std::size_t) count + 1u, 0);
    for (std::uint64_t i = 0u; i < count; ++i) {
        const std::uint64_t global_row = start + i;
        const auto it = std::upper_bound(matrices_.begin(), matrices_.end(), global_row,
                                         [](std::uint64_t row, const auto &matrix) { return row < matrix.row_begin; });
        if (it == matrices_.begin()) throw_format("failed to resolve row to matrix partition");
        const spec::matrix_descriptor &matrix = *(it - 1);
        const std::uint64_t local_row = global_row - matrix.row_begin;
        if (local_row >= matrix.rows) throw_format("resolved row outside matrix partition");
        const section_entry *a = find_section(sections_, matrix.section_a_id);
        const section_entry *b = find_section(sections_, matrix.section_b_id);
        const section_entry *c = find_section(sections_, matrix.section_c_id);
        const section_entry *d = find_section(sections_, matrix.section_d_id);
        if (a == nullptr || b == nullptr) throw_format("matrix section missing while reading rows");
        if (matrix.layout == spec::matrix_layout_blocked_ell) {
            append_blocked_row_from_file(path_, matrix, *a, *b, local_row, &out);
        } else if (matrix.layout == spec::matrix_layout_sliced_ell) {
            if (c == nullptr || d == nullptr) throw_format("sliced-ELL section missing while reading rows");
            append_sliced_row_from_file(path_, matrix, *a, *b, *c, *d, local_row, &out);
        } else if (matrix.layout == spec::matrix_layout_csr) {
            if (c == nullptr) throw_format("CSR section missing while reading rows");
            append_csr_row_from_file(path_, matrix, *a, *b, *c, local_row, &out);
        } else {
            throw_format("unsupported matrix layout while reading rows");
        }
        out.indptr[(std::size_t) i + 1u] = (std::int64_t) out.data.size();
    }
    return out;
}

bool write_csr(const std::string &path,
               const exporting::csr_matrix_export &csr,
               const table_view &obs,
               const table_view &var,
               const writer_options &options,
               std::string *error) {
    try {
        if (csr.indptr.size() != (std::size_t) csr.rows + 1u || csr.indices.size() != csr.data.size()) {
            throw std::runtime_error("invalid CSR input");
        }
        file_plan plan = base_plan(csr.rows, csr.cols, csr.data.size(), spec::matrix_layout_csr, obs, var, options);
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
        desc.matrix_id = 0u;
        desc.partition_id = 0u;
        desc.layout = spec::matrix_layout_csr;
        desc.value_dtype = spec::dtype_f16;
        desc.index_dtype = spec::dtype_u32;
        desc.rows = csr.rows;
        desc.cols = csr.cols;
        desc.nnz = csr.data.size();
        desc.section_a_id = add_payload(&plan,
                                        spec::section_kind_csr_row_ptr,
                                        spec::dtype_u64,
                                        bytes_from_array(row_ptr.data(), row_ptr.size()),
                                        row_ptr.size());
        desc.section_b_id = add_payload(&plan,
                                        spec::section_kind_csr_col_idx,
                                        spec::dtype_u32,
                                        bytes_from_array(col_idx.data(), col_idx.size()),
                                        col_idx.size());
        desc.section_c_id = add_payload(&plan,
                                        spec::section_kind_csr_values,
                                        spec::dtype_f16,
                                        bytes_from_array(values.data(), values.size()),
                                        values.size());
        plan.matrices.push_back(desc);
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
