#include "h5ad_writer.hh"

#include <cstdio>
#include <string>

int main(int argc, char **argv) {
    std::string error;
    if (argc != 3) {
        std::fprintf(stderr, "usage: %s <input.series.csh5> <output.h5ad>\n", argc > 0 ? argv[0] : "cellshardH5adExport");
        return 1;
    }
    if (!cellshard::exporting::write_series_file_to_h5ad_with_python(argv[1], argv[2], &error)) {
        std::fprintf(stderr, "cellshardH5adExport failed: %s\n", error.c_str());
        return 1;
    }
    return 0;
}
