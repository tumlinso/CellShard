# Active Objectives

## Summary
Use this file as the canonical index for substantial multi-step work.

## Shared Assumptions
_None recorded yet._

## Suggested Skills
- `todo-orchestrator` - Keep the multi-step CellShard/Cellerator migration ledger current.
- `cuda` - Use for pack-build, staging, and benchmark decisions on the native V100/Volta path.

## Useful Reference Files
- `AGENTS.md` - CellShard ownership, runtime, format, and package-surface rules.
- `README.md` - Current storage, pack delivery, Python, and distributed runtime posture.
- `docs/FORMAT_ROLES.md` - Archive-vs-pack role boundary.
- `docs/SPEC_CSPACK_V1.md` - Current CSPACK01 byte contract that must remain stable.
- `include/CellShard/io/common/matrix_traits.hh` - Current matrix-owned trait surface to replace or narrow.
- `include/CellShard/runtime/layout/sharded.cuh` - Current partition and shard metadata model.
- `../Cellerator/AGENTS.md` - Cellerator ownership for Core layouts and optimized compute bindings.

## Workstreams
- `cellshard-cpp-access-adapter-refactor` | status: in_progress | owner: codex | file: `todos/cellshard-cpp-access-adapter-refactor.md` | objective: Implement max-performance C++ access adapters so CellShard owns biological delivery and dense/compressed fallbacks while Cellerator owns optimized sparse bindings.

## Global Blockers
_None recorded yet._

## Progress Notes
- Added include/CellShard/access adapter contract, dense/compressed fallback bindings, and cellShardAccessAdapterTest fake external binding smoke.
- Added optional Cellerator Core interop adapter header for dense, compressed, Blocked-ELL, Sliced-ELL, and quantized Blocked-ELL layout descriptors.
- Updated README, SUPPORT, FORMAT_ROLES, CSPACK, and CSHARD docs to describe CellShard-owned biological metadata plus adapter-owned optimized layouts.
- Validation passed: CellShard configure, full build, cellShardAccessAdapterTest, cellShardExportRuntimeTest, cellShardCshardTest, cellShardInspectPackageTest, cellShardDenseRuntimeTest, and cellShardOptimizedSlicedExecutionTest.
- Validation passed: Cellerator configure, cellshardAccessAdapterCompileTest build/run, quantizedMatrixTest, and exactSearchRuntimeTest.
- Validation caveats: cellShardMaskGroupsRuntimeTest exits 14 at the row-keep assertion with no stderr; computeAutogradRuntimeTest and quantizeModelTest are not present in the current Cellerator CMake target list.
- Package consumer now includes CellShard/access.hh and static-checks dense fallback adapter visibility from the installed package.

## Next Actions
- Create or resume a workstream ledger under `todos/` for the next substantial task.
- Finish the hard cutover by routing CSH5/CSPACK materialization through archive_to_pack and replacing remaining Cellerator consumers of CellShard/formats optimized-layout aliases.

## Done Criteria
- Every active workstream in `todos/` is reflected here with a current status.
- CellShard exposes a documented header-only C++ adapter API with dense/compressed fallback adapters and a fake external adapter test.
- Public docs no longer imply CellShard owns CelleratorCore optimized sparse matrix layouts.
- Current CellShard and Cellerator build/test gates pass or failures are classified with exact commands.
