# Repack

`src/repack/` owns shard-level CUDA repacking for execution-time sparse views.

Scope:
- shard-local filtered assembly from uploaded sparse parts
- shard-local row bucketing using existing `bucket/` operators
- sorted Blocked-ELL rebuild on GPU for the rebuilt shard view

Non-goals:
- replace ingest-time per-part conversion
- replace `convert/` format primitives
- change packfile or shard metadata contracts

Current posture:
- authoritative input boundary is a shard span from `sharded<>`
- input parts are uploaded `device::blocked_ell_view` records
- row filters/remaps are shard-local
- feature filters/remaps are matrix-global
- rebuilt Blocked-ELL rows follow the bucketed row order; callers must use the
  returned bucket plan when they need original row order
