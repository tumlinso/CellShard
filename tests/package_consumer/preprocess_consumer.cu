#include <CellShardPreprocess/runtime.hh>

int main() {
    cspre::status status{};
    cspre::clear_status(&status);
    return status.code == cspre::status_ok ? 0 : 1;
}
