

<!-- todo-orchestrator:v2-managed:start -->
# CS-FOUND-03: Strong identity, digest, and status primitives

Task revision: `263`; current project revision is in `todo-status.md`.

## Objective
Implement the specified zero-cost strong IDs, tagged content digest, array view, and typed status primitive as CS-FOUND-I1.

## State
- Lifecycle: `done`
- Execution: `closed`
- Parallel policy: `serial`
- Result: `implemented`

## Next Action
Implement and validate CS-FOUND-I1 only; do not modify legacy identities.

## Ownership
- `exclusive`: `include/CellShard/identity`
- `exclusive`: `include/CellShard/identity.hh`
- `exclusive`: `tests/foundation_identity_test.cc`
- `read`: `include/CellShard/core`

## Dependencies
- `task`: `CS-FOUND-02`
- `checkpoint`: `CS-FOUND-G1`
<!-- todo-orchestrator:v2-managed:end -->
