---
slug: "cellshard-cpp-access-adapter-refactor"
status: "closed"
execution: "closed"
owner: "codex"
created_at: "2026-05-31T14:17:11Z"
last_heartbeat_at: "2026-05-31T14:58:38Z"
last_reviewed_at: "2026-05-31T14:58:38Z"
stale_after_days: 3
objective: "Implement max-performance C++ access adapters so CellShard owns biological delivery and dense/compressed fallbacks while Cellerator owns optimized sparse bindings."
---

# Current Objective

## Summary
Implement max-performance C++ access adapters so CellShard owns biological delivery and dense/compressed fallbacks while Cellerator owns optimized sparse bindings.

## Quick Start
- Read AGENTS.md, README.md, docs/FORMAT_ROLES.md, docs/SPEC_CSPACK_V1.md, include/CellShard/io/common/matrix_traits.hh, include/CellShard/runtime/layout/sharded.cuh, and Cellerator/AGENTS.md before continuing.
- Goal: make CellShard expose a max-performance C++ header adapter contract for biology-centric archive/pack delivery while keeping only dense and compressed fallback payloads CellShard-owned.
- Cellerator should own Cellerator Blocked-ELL, Sliced-ELL, quantized, and future optimized layout adapters.

## Planning Notes
- Primary integration API is header-only C++ templates and typed adapter views, not virtual methods or callback tables.
- Hot paths must be pointer/span/preallocated-buffer based and avoid scalar traversal, runtime dispatch, hidden conversion, and vector-based public hot APIs.
- First target is production .csh5 -> .cspack; .cshard remains secondary but docs must stay honest.

## Assumptions
- Hard cutover is preferred over compatibility shims for Cellerator-owned matrix aliases.
- CellShard owns packed row-major dense and compressed sparse fallback/interchange payloads only.
- No cudaBioTypes dependency is added to CellShard.

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

## Plan
_None recorded yet._

## Tasks
- [x] Create CellShard access adapter contract and fake external adapter compile/runtime smoke.
- [x] Document dense/compressed fallback ownership and adapter-owned optimized layouts.
- [x] Migrate CellShard CSH5/CSPACK path toward adapter-based archive-to-pack hooks without changing CSPACK01 bytes.
- [x] Move Cellerator optimized sparse binding ownership into Cellerator headers and update consumers.
- [x] Run CellShard and Cellerator validation gates or record blockers.

## Blockers
_None recorded yet._

## Progress Notes
- Added include/CellShard/access adapter contract, dense/compressed fallback bindings, and cellShardAccessAdapterTest fake external binding smoke.
- Added optional Cellerator Core interop adapter header for dense, compressed, Blocked-ELL, Sliced-ELL, and quantized Blocked-ELL layout descriptors.
- Updated README, SUPPORT, FORMAT_ROLES, CSPACK, and CSHARD docs to describe CellShard-owned biological metadata plus adapter-owned optimized layouts.
- Validation passed: CellShard configure, full build, cellShardAccessAdapterTest, cellShardExportRuntimeTest, cellShardCshardTest, cellShardInspectPackageTest, cellShardDenseRuntimeTest, and cellShardOptimizedSlicedExecutionTest.
- Validation passed: Cellerator configure, cellshardAccessAdapterCompileTest build/run, quantizedMatrixTest, and exactSearchRuntimeTest.
- Validation caveats: cellShardMaskGroupsRuntimeTest exits 14 at the row-keep assertion with no stderr; computeAutogradRuntimeTest and quantizeModelTest are not present in the current Cellerator CMake target list.
- Package consumer now includes CellShard/access.hh and static-checks dense fallback adapter visibility from the installed package.
- Integrated CellShard runtime sharded payload helpers with access::payload_traits<MatrixT>, so MatrixT policy now comes from the adapter/fallback trait surface for aux, nnz, byte sizing, and optional debug scalar inspection.
- Hard-cut Cellerator optimized CellShard consumers from CellShard format aliases to Cellerator matrix types plus Cellerator-owned CellShard access interop traits.
- Validation passed after the sharded/MatrixT cutover: CellShard full build, cellShardAccessAdapterTest, cellShardDenseRuntimeTest, cellShardOptimizedSlicedExecutionTest, cellShardExportRuntimeTest, cellShardCshardTest, and cellShardInspectPackageTest; Cellerator adapter/sparse/model runtime targets and quantizedMatrixTest/exactSearchRuntimeTest passed.
- Routed production ensure_cspack_ready materialization through access::archive_to_pack<csh5_shard_archive_binding, cspack_shard_pack_binding, csh5_to_cspack_default_policy>; typed dense, quantized, blocked, and sliced CSPACK01 writers remain byte-compatible implementation functions.
- Validation passed after the CSH5/CSPACK routing change: CellShard full build, cellShardAccessAdapterTest, cellShardDenseRuntimeTest, cellShardOptimizedSlicedExecutionTest, cellShardExportRuntimeTest, cellShardCshardTest, and cellShardInspectPackageTest.
- Closed after CellShard, Cellerator, and CellStack root pointer commits were pushed. Known caveats remain tracked as validation notes rather than active implementation blockers.

## Next Actions
- Workstream closed. Reopen only for a follow-up that removes remaining compatibility format headers or broadens external policy injection.

## Done Criteria
- CellShard exposes a documented header-only C++ adapter API with dense/compressed fallback adapters and a fake external adapter test.
- Public docs no longer imply CellShard owns Cellerator optimized sparse matrix layouts.
- Current CellShard and Cellerator build/test gates pass or failures are classified with exact commands.

<!-- todo-orchestrator:v2-managed:start -->
# cellshard-cpp-access-adapter-refactor: Implement max-performance C++ access adapters so CellShard owns biological delivery and dense/compressed fallbacks while Cellerator owns optimized sparse bindings.

Task revision: `2`; current project revision is in `todo-status.md`.

## Objective
Implement max-performance C++ access adapters so CellShard owns biological delivery and dense/compressed fallbacks while Cellerator owns optimized sparse bindings.

## State
- Lifecycle: `closed`
- Execution: `idle`
- Parallel policy: `serial`
- Result: `-`

## Next Action
closed; reopen only for compatibility-header removal or broader external policy injection.

## Ownership
_No structured ownership._

## Dependencies
_None._
<!-- todo-orchestrator:v2-managed:end -->
