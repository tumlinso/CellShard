# JBC integration receipt

CS-JBC-V20 consumed 22 immutable producer artifacts, positions 0 through 21,
against frozen CellShard base
`7762a5925fe18b2ca45ab8a436f3461804ed2ad9`. Project Control applied every
artifact in queue order. The cumulative frozen integration tree before the
V20-owned aggregation edits is
`db53b20195d29aedc711962fdcb6d85a4bbf809f`; every queue entry is recorded as
`integrated` with an empty conflict record.

The final aggregation adds the public JBC host-runtime package target, CUDA
transport/collective implementations, complete source-linked test and benchmark
targets, representative umbrella-header exposure, and an explicit NCCL discovery
contract. It does not move Cellerator numerical semantics into CellShard or
promote an unmeasured biological-performance claim.

Acceptance evidence:

- 304 host JBC tests passed in the integrated host build.
- The same 304 tests passed under ASan+UBSan with leak detection.
- All host sources passed the strict warning build; the umbrella consumer keeps
  legacy signed-mask warnings from external compatibility headers visible but
  non-fatal.
- Four host benchmark/reference executables passed.
- Six CUDA test executables and the dual-NUMA campaign executable built for
  `sm_70` with CUDA 12.9 and NCCL 2.29.2.
- Host standalone install and installed-package consumer build passed.
- `git diff --check` and every Project Control post-merge integration gate
  passed.

The Runtime lane owns the accepted controller-reserved four-V100 evidence,
including the corrected fork-plus-exec process model. V20 reuses that evidence
and makes no new throughput or biological-novelty claim.
