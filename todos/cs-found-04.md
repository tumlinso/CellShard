

<!-- todo-orchestrator:v2-managed:start -->
# CS-FOUND-04: Domain and partition descriptors

Task revision: `263`; current project revision is in `todo-status.md`.

## Objective
Implement domain, partition-map, partition-selection, partition-descriptor, and explicit-order domain-binding contracts as CS-FOUND-I2A.

## State
- Lifecycle: `done`
- Execution: `closed`
- Parallel policy: `parallel_safe`
- Result: `implemented`

## Next Action
Implement only I2A in its exclusive files and complete positive/negative validation.

## Ownership
- `exclusive`: `CMakeLists.txt`
- `exclusive`: `include/CellShard/domain`
- `exclusive`: `include/CellShard/domain.hh`
- `exclusive`: `tests/foundation_domain_test.cc`
- `read`: `include/CellShard/identity`
- `read`: `tests/foundation_identity_test.cc`

## Dependencies
- `task`: `CS-FOUND-03`
<!-- todo-orchestrator:v2-managed:end -->
