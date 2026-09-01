#include <CellShard/artifact/atom_store/coverage_index_v1.hh>
#include <cassert>
namespace atom_store = cellshard::artifact::atom_store;
int main() {
    atom_store::exact_coverage_record_v1 records[] = {
        {1, 0, 4, {1, 1}, 11}, {1, 4, 3, {2, 2}, 12},
        {2, 0, 1, {3, 3}, 13}};
    assert(atom_store::validate_exact_coverage_index_v1(records, 3).valid());
    records[1].item_begin = 3;
    assert(atom_store::validate_exact_coverage_index_v1(records, 3).code
        == atom_store::coverage_index_code_v1::overlapping_coverage);
    return 0;
}
