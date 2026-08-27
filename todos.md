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
- `cellshard-cpp-access-adapter-refactor` | status: closed | owner: codex | file: `todos/cellshard-cpp-access-adapter-refactor.md` | objective: Implement max-performance C++ access adapters so CellShard owns biological delivery and dense/compressed fallbacks while Cellerator owns optimized sparse bindings.

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
- Integrated CellShard runtime sharded payload helpers with access::payload_traits<MatrixT>, so MatrixT policy now comes from adapter/fallback traits for aux, nnz, byte sizing, and optional debug scalar inspection.
- Hard-cut Cellerator optimized consumers from CellShard format aliases to Cellerator matrix types plus Cellerator-owned CellShard access interop traits.
- Validation passed after the sharded/MatrixT cutover: CellShard full build, focused runtime/package tests, Cellerator adapter/sparse/model runtime targets, quantizedMatrixTest, and exactSearchRuntimeTest.
- Routed production ensure_cspack_ready materialization through access::archive_to_pack<csh5_shard_archive_binding, cspack_shard_pack_binding, csh5_to_cspack_default_policy>; typed dense, quantized, blocked, and sliced CSPACK01 writers remain byte-compatible implementation functions.
- Validation passed after the CSH5/CSPACK routing change: CellShard full build, cellShardAccessAdapterTest, cellShardDenseRuntimeTest, cellShardOptimizedSlicedExecutionTest, cellShardExportRuntimeTest, cellShardCshardTest, and cellShardInspectPackageTest.
- Closed after CellShard, Cellerator, and CellStack root pointer commits were pushed. Known caveats remain tracked as validation notes rather than active implementation blockers.

## Next Actions
- Create or resume a workstream ledger under `todos/` for the next substantial task.
- Reopen a new workstream only for a follow-up that removes remaining compatibility format headers or broadens external policy injection.

## Done Criteria
- Every active workstream in `todos/` is reflected here with a current status.
- CellShard exposes a documented header-only C++ adapter API with dense/compressed fallback adapters and a fake external adapter test.
- Public docs no longer imply CellShard owns Cellerator optimized sparse matrix layouts.
- Current CellShard and Cellerator build/test gates pass or failures are classified with exact commands.

<!-- todo-orchestrator:v2-managed:start -->
# Todo Orchestrator v2 Projection

Project revision: `166`

## Workstreams
- `CS-FOUND-00` | kind: epic | status: in_progress | parent: - | objective: Establish stable CellShard identity, domain, partition, image, extent, source, snapshot, host-residency, and device-residency contracts before placement, streaming, routing, or multi-node work.
- `CS-FOUND-01` | kind: task | status: done | parent: CS-FOUND-00 | objective: Reconstruct current CellShard implementation and reconcile it with the architecture review before introducing new architecture code.
- `CS-FOUND-02` | kind: task | status: done | parent: CS-FOUND-00 | objective: Protect current CSPACK01 and native CPEXEC01 behavior with focused positive, malformed-input, ownership, publication, and CUDA upload tests.
- `CS-FOUND-03` | kind: task | status: done | parent: CS-FOUND-00 | objective: Implement the specified zero-cost strong IDs, tagged content digest, array view, and typed status primitive as CS-FOUND-I1.
- `CS-FOUND-04` | kind: task | status: done | parent: CS-FOUND-00 | objective: Implement domain, partition-map, partition-selection, partition-descriptor, and explicit-order domain-binding contracts as CS-FOUND-I2A.
- `CS-FOUND-05` | kind: task | status: done | parent: CS-FOUND-00 | objective: Represent producer-owned opaque execution images through producer ABI, target, projection, reuse, owning descriptor, and allocation-free view as CS-FOUND-I2B.
- `CS-FOUND-06` | kind: task | status: done | parent: CS-FOUND-00 | objective: Separate immutable storage objects/extents from mutable source locations and define the non-owning storage-independent exact-read boundary as CS-FOUND-I2C.
- `CS-FOUND-07` | kind: task | status: done | parent: CS-FOUND-00 | objective: Freeze the descriptor ABI checkpoint, then implement separate in-memory artifact/source catalogs and snapshot validation as CS-FOUND-I3A.
- `CS-FOUND-08` | kind: task | status: done | parent: CS-FOUND-00 | objective: Implement deterministic explicit little-endian CPEXEC02 buffer encoding/decoding as CS-FOUND-I3B without CSPACK file integration.
- `CS-FOUND-09` | kind: integration_task | status: done | parent: CS-FOUND-00 | objective: Integrate independently inspectable CPEXEC02 entries into unchanged CSPACK01 with atomic publication and expose CS-FOUND-I4.
- `CS-FOUND-10` | kind: task | status: planned | parent: CS-FOUND-00 | objective: Reach G3, then resolve one image extent through LocalFileSource into one move-only host residency allocation as CS-FOUND-I5A.
- `CS-FOUND-12` | kind: task | status: planned | parent: CS-FOUND-00 | objective: Provide explicit-context legacy CPEXEC01 and sharded adapters without defining new semantics through row shards.
- `CS-FOUND-11` | kind: task | status: planned | parent: CS-FOUND-00 | objective: Stage a validated host image through a caller allocator with one asynchronous H2D copy and move-only device residency as CS-FOUND-I5B.
- `CS-FOUND-13` | kind: integration_task | status: planned | parent: CS-FOUND-00 | objective: Freeze the image/residency checkpoint and prove the specified fake producer to host/device fake consumer pipeline with exact identity and byte preservation.
- `CS-FOUND-14` | kind: integration_task | status: planned | parent: CS-FOUND-00 | objective: Expose and document implemented CS-FOUND contracts honestly while preserving optional CUDA and format-role boundaries.
- `CS-FOUND-15` | kind: validation_task | status: planned | parent: CS-FOUND-00 | objective: Validate the full feasible CS-FOUND program, reconcile ledger and repository, generate final projections, and recommend but do not create the broader next physical-runtime investigation.
- `cellshard-cpp-access-adapter-refactor` | kind: workstream | status: closed | parent: - | objective: Implement max-performance C++ access adapters so CellShard owns biological delivery and dense/compressed fallbacks while Cellerator owns optimized sparse bindings.
<!-- todo-orchestrator:v2-managed:end -->
