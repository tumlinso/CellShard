# CellShard provider, registry, and integration path reservations

This `CS-JBC-B05` record reserves non-overlapping paths and integration
ownership for the biology-native compiler program. A reservation is inert: it
does not create a public ABI, register a provider, add a target, change an
export, or activate a lane. Provider tasks may implement only inside their
reserved source-linked areas after their prerequisites are authoritative.

The machine-readable companion is
[`planning/jbc/cellshard_provider_path_reservations_v1.csv`](../../planning/jbc/cellshard_provider_path_reservations_v1.csv).

## Provenance and authority cursor

The exact preledger input is the `CS-JBC-B05` object in
`proposed_todos.json`. Its newline-terminated compact sorted JSON SHA-256 is
`3c0a48049fc7b344e794ad6025aa94463e9db98b7b4de1479f054596bf69f40a`;
the ledger-recorded package digest is
`6ef0c3ba6cc37a6b513209a0d830c541f5aab6f872b3eac7732cb8dc28945e2f`.

Project Control observation `2026-09-01T06:11:17Z` established:

- CellShard commit `790379b30cb4bef82e660fa2ee866ab14e413330`;
- clean source-worktree fingerprint
  `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855`;
- Todo UUID `a52537a5-20db-4aeb-a126-dd0128c71fda`, revision `322`, and
  semantic/workflow fingerprint
  `921e22eac7a8bf041f0d4ad365f37dc222d0bb00777a9c37a3eb94de62231ac0`;
- zero provider skew; and
- B05 as the sole active serial bootstrap task, with B06 queued.

A separately timestamped Project Control observation
`2026-09-01T06:14:52Z`, after the reservations were drafted, re-established
Todo revision `322`, commit `790379b30cb4bef82e660fa2ee866ab14e413330`,
and zero provider skew. Its semantic/workflow authority fingerprint was
`2f5271168e55c40c90f4dfaac0d2fd6936cc9eb1530db97ded463586a08f0c6c`.
The bounded worktree fingerprint
`97c5bd4c8cf381bdeabe7042445bb5a0b3b7c21601a91f78514dac35f9ecdcd4`
contained exactly this record and its CSV companion; B05 remained the sole
active task and no downstream lane was active.

## Frozen CellShard interface locations

These locations come directly from the preledger interface catalog. They are
proposed ownership destinations until the named owning task freezes the
contract.

| Interface | Owner | Reserved location | Meaning |
| --- | --- | --- | --- |
| `JBC-I12` | CellShard | `include/CellShard/compiler/atom/atom_v1.hh` | Common level-relative atom envelope. |
| `JBC-I13` | CellShard | `include/CellShard/compiler/evidence/atlas_v1.hh` | Proposal evidence, provenance, strata, stability, negative evidence, and rescan state. |
| `JBC-I14` | CellShard | `include/CellShard/compiler/composition/production_v1.hh` | Typed production and multi-parent derivation DAG. |
| `JBC-I15` | CellShard | `include/CellShard/compiler/basis/basis_v1.hh` | Biological execution basis manifest and no-basis fallback. |
| `JBC-I16` | CellShard with Cellerator algebra | `include/CellShard/compiler/partial/partial_atom_v1.hh` | Persistent partial with exact contribution and merge/finalize algebra. |
| `JBC-I17` | CellShard | `include/CellShard/artifact/atom_store/` | Immutable atom-store generation and physical instance family. |
| `JBC-I18` | CellShard | `include/CellShard/compiler/graph/provider_v1.hh` and `schedule_v1.hh` | Provider-neutral global operation graph and portable schedule; the owning task resolves the latter inside the reserved graph/schedule roots. |
| `JBC-I19` | CellShard | `include/CellShard/compiler/certification/distributed_certificate_v1.hh` | Exact owners, contributors, halos, replicas, recovery, and omission/duplicate proof. |
| `JBC-I20` | CellShard | `include/CellShard/runtime/v2/` | Topology realization, acquisition, transport, residency lease, readiness, and reconstruction. |

Each frozen interface must use the smallest pointer-first contract required for
fan-out. A change creates an adjacent version. Every consumer receipt records
interface ID, version, content hash, source commit, Todo revision, and exact
paths; a matching type or filename is not authority.

## Provider path policy

The CSV reserves 23 lane-level ownership rows derived from the preledger task
write scopes. The key families are:

```text
include/CellShard/compiler/{atom,evidence,certification}/
include/CellShard/compiler/discovery/{bicluster,co_support,factor_topic,motif}/
include/CellShard/compiler/discovery/{multimodal,operation_trace,overlap}/
include/CellShard/compiler/discovery/{sequence_compat,support_signature,trajectory}/
include/CellShard/compiler/{composition,grammar,basis,partial}/
include/CellShard/compiler/{graph,schedule}/
include/CellShard/artifact/atom_store/
include/CellShard/runtime/v2/
```

Implementations mirror public ownership under `src/`; independent fixtures
mirror it under `tests/jbc/`; complete-cost campaigns use the explicitly
reserved `bench/jbc/` subtrees. The persistence lane additionally owns the
future `docs/SPEC_ATOM_STORE_V1.md`. Validation/integration owns durable
evidence under `docs/JBC/evidence/`.

A provider publishes a self-contained, source-linked fragment with:

- stable provider identity, interface version/hash receipt, and capability;
- exact accepted domains, orders, species, planes, and evidence prerequisites;
- declared cold allocations, capacities, asymptotic time, and peak memory;
- deterministic invalid/stale/capacity/candidate-explosion diagnostics;
- exact rescan/certification boundary and canonical fallback;
- cost/reuse evidence and explicit no-candidate/no-promotion outcomes; and
- tests and fixtures that compile without editing a central registry.

Discovery providers propose only. They do not publish executable ownership,
rewrite atom species, infer scientific semantics, or register themselves by
editing a shared file. Runtime providers carry operational topology, device,
path, handle, and service state without putting those values into portable
biological identity.

## Registry ownership

Registries are versioned aggregations, not ad-hoc lists edited by every
provider. Their owning tasks are:

- atom species: `CS-JBC-L-ATOM-CORE`;
- explicit productions: `CS-JBC-L-EXPLICIT-GRAMMAR`;
- atom-store codecs: `CS-JBC-L-PERSISTENCE`;
- transport providers: `CS-JBC-L-RUNTIME`; and
- final cross-provider discovery, format, target, test, and package aggregation:
  `CS-JBC-L-VALIDATION-INTEGRATION`.

Before the relevant registry contract is frozen, downstream work consumes
direct fixtures or mocks. Once frozen, provider fragments expose descriptors or
factories through their isolated paths. Only the registry owner or validation
integration lane assembles them. Duplicate identity, incompatible version,
missing receipt, capability conflict, or dependency cycle fails closed and
leaves the canonical/reference fallback available.

## Integration-only surfaces

The following remain untouched by ordinary provider tasks:

- root `CMakeLists.txt`, including top-level test/benchmark registration;
- `include/CellShard/CellShard.hh` and any other umbrella header;
- installed target lists, package configuration, exports, and package-consumer
  registration;
- central discovery, species, production, codec, transport, format, and
  profiler registries;
- final format registration and compatibility-route aggregation;
- shared program/evidence summaries outside the provider's own evidence path;
  and
- the parent Cellerator `components/CellShard` gitlink.

The validation/integration lane performs CellShard central aggregation after
source-linked provider fragments and interface receipts exist. CellShard is
committed, validated, and pushed first. A separately authorized parent
Cellerator integration task advances the gitlink once per checkpoint bundle
and reruns standalone and embedded builds. An ordinary provider task never
mutates both repositories.

## Complexity and validation posture

B05 is a cold reservation pass over `P=23` lane rows and `I=9` proposed
CellShard interfaces: `O(P + I)` validation time and storage, with no runtime
allocation or execution effect. Future providers must declare their own
asymptotic and peak-memory bounds; unrestricted all-pairs or subgraph
enumeration is allowed only for a declared exact small-instance oracle.

This task changes no header, source, CMake target, registry, export, package, or
wire contract, so no build is required. The exact later build entry points are:

```bash
cmake -S /home/tumlinson/Cellerator/components/CellShard \
  -B /home/tumlinson/Cellerator/components/CellShard/build \
  -DCELLSHARD_BUILD_TESTS=ON
cmake --build /home/tumlinson/Cellerator/components/CellShard/build \
  -j "$(nproc)"

cmake -S /home/tumlinson/Cellerator -B /home/tumlinson/Cellerator/build
cmake --build /home/tumlinson/Cellerator/build -j "$(nproc)"
```

The closed `cellshard-cpp-access-adapter-refactor` remains historical evidence
and is not reopened or duplicated. `CS-JBC-B06` owns the final bootstrap
validation and cross-repository handoff; B05 does not claim or complete it.
