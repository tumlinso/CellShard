

<!-- todo-orchestrator:v2-managed:start -->
# CS-STORAGE-01: Own file-backed sampling materialization and compute handoff

Task revision: `311`; current project revision is in `todo-status.md`.

## Objective
Extend existing CellShard export/storage surfaces with a neutral selected-row structural handoff, add the optional Cellerator integration adapter and tests where justified, and update live ownership documentation without changing CPE2 semantics or requiring Cellerator for normal builds.

## State
- Lifecycle: `done`
- Execution: `closed`
- Parallel policy: `serial`
- Result: `implemented`

## Next Action
Reuse existing dataset export and execution-payload APIs, implement only the CellShard-owned adapter boundary, add focused synthetic tests, update current docs, and validate CPU/CUDA/package buildability.

## Ownership
- `exclusive`: `AGENTS.md`
- `exclusive`: `CMakeLists.txt`
- `exclusive`: `README.md`
- `exclusive`: `SUPPORT.md`
- `exclusive`: `docs/FORMAT_ROLES.md`
- `exclusive`: `export`
- `exclusive`: `include/CellShard/export`
- `exclusive`: `include/CellShard/interop/cellerator`
- `exclusive`: `src/interop/cellerator`
- `exclusive`: `tests`
- `read`: `docs`
- `read`: `include`
- `read`: `src`

## Dependencies
_None._
<!-- todo-orchestrator:v2-managed:end -->
