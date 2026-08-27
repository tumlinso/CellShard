

<!-- todo-orchestrator:v2-managed:start -->
# CS-FOUND-14: Public API, package, and documentation migration

Task revision: `166`; current project revision is in `todo-status.md`.

## Objective
Expose and document implemented CS-FOUND contracts honestly while preserving optional CUDA and format-role boundaries.

## State
- Lifecycle: `planned`
- Execution: `ready`
- Parallel policy: `integration_exclusive`
- Result: `-`

## Next Action
Update only implemented package/docs surfaces, add CS_FOUND_CONTRACT, run installed-package smoke, and reach G6 without rewriting the architecture review.

## Ownership
- `exclusive`: `AGENTS.md`
- `exclusive`: `CMakeLists.txt`
- `exclusive`: `README.md`
- `exclusive`: `SUPPORT.md`
- `exclusive`: `docs/CS_FOUND_CONTRACT.md`
- `exclusive`: `docs/FORMAT_ROLES.md`
- `exclusive`: `docs/PARKING_LOT.md`
- `exclusive`: `docs/SPEC_CSPACK_V1.md`
- `exclusive`: `include/CellShard/CellShard.hh`
- `exclusive`: `include/CellShard/artifact.hh`
- `exclusive`: `include/CellShard/io/pack.hh`
- `exclusive`: `include/CellShard/runtime/residency.hh`
- `exclusive`: `include/CellShard/runtime/source.hh`
- `exclusive`: `tests/package_consumer`
- `read`: `CellShard_complete_architecture_review.md`
- `read`: `include/CellShard`

## Dependencies
- `task`: `CS-FOUND-07`
- `task`: `CS-FOUND-09`
- `task`: `CS-FOUND-12`
- `task`: `CS-FOUND-13`
- `checkpoint`: `CS-FOUND-G5`
<!-- todo-orchestrator:v2-managed:end -->
