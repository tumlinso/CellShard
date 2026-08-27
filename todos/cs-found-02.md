

<!-- todo-orchestrator:v2-managed:start -->
# CS-FOUND-02: Legacy contract and validation baseline

Task revision: `166`; current project revision is in `todo-status.md`.

## Objective
Protect current CSPACK01 and native CPEXEC01 behavior with focused positive, malformed-input, ownership, publication, and CUDA upload tests.

## State
- Lifecycle: `done`
- Execution: `closed`
- Parallel policy: `serial`
- Result: `implemented`

## Next Action
Add only the legacy codec/container baseline tests, run G1 gates, record results, reach G1, and stop without claiming CS-FOUND-03.

## Ownership
- `exclusive`: `CMakeLists.txt`
- `exclusive`: `tests/execution_payload_cpu_test.cc`
- `exclusive`: `tests/execution_payload_test.cu`
- `read`: `docs/SPEC_CSPACK_V1.md`
- `read`: `include/CellShard/io/pack/execution_payload.cuh`
- `read`: `include/CellShard/io/pack/packfile.cuh`
- `read`: `src/io/pack/execution_payload.cu`
- `read`: `src/io/pack/packfile.cu`

## Dependencies
- `task`: `CS-FOUND-01`
- `checkpoint`: `CS-FOUND-G0`
<!-- todo-orchestrator:v2-managed:end -->
