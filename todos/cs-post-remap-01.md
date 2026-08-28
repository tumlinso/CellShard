

<!-- todo-orchestrator:v2-managed:start -->
# CS-POST-REMAP-01: Replace Cellerator runtime coupling with neutral CellShard delivery descriptors

Task revision: `305`; current project revision is in `todo-status.md`.

## Objective
Refactor CellShard distributed and mask-group APIs so public/runtime code no longer includes Cellerator runtime or distributed headers, names Cellerator runtime types, links Cellerator::dist, or synthesizes that target; preserve compatibility-format headers separately and prove an independent CUDA-enabled CellShard build.

## State
- Lifecycle: `done`
- Execution: `closed`
- Parallel policy: `project_exclusive`
- Result: `implemented`

## Next Action
Replace local_context and Cellerator::dist use with narrow neutral inputs, remove Cellerator::dist CMake synthesis/linkage, and validate CellShard independently with CUDA enabled.

## Ownership
- `exclusive`: `.todo-orchestrator`
- `exclusive`: `ARCHITECTURE_FOLLOWUPS.md`
- `exclusive`: `CMakeLists.txt`
- `exclusive`: `README.md`
- `exclusive`: `SUPPORT.md`
- `exclusive`: `cellshard-cellerator-cycle-plan.json`
- `exclusive`: `include/CellShard/runtime`
- `exclusive`: `src/runtime`
- `exclusive`: `tests`
- `exclusive`: `todo-status.md`
- `exclusive`: `todos.md`
- `read`: `docs`
- `read`: `export`
- `read`: `include/CellShard`
- `read`: `src`

## Dependencies
_None._
<!-- todo-orchestrator:v2-managed:end -->
