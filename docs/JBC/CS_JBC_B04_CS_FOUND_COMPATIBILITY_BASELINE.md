# Frozen CS-FOUND compatibility baseline

This is the `CS-JBC-B04` snapshot of the frozen CS-FOUND contracts that the
biology-native compiler successor must import, adapt, or retain. It records
current compatibility authority; it does not revise an interface, promote a
format, define the successor atom ABI, or claim that compatibility machinery is
the successor architecture. The frozen charter remains
[`CELLSHARD_JBC_SUCCESSOR_CHARTER_V1.md`](CELLSHARD_JBC_SUCCESSOR_CHARTER_V1.md).

The machine-readable companion is
[`planning/jbc/cs_found_compatibility_baseline_v1.csv`](../../planning/jbc/cs_found_compatibility_baseline_v1.csv).

## Provenance and authority cursor

The exact preledger input is the `CS-JBC-B04` object in
`proposed_todos.json`. Its newline-terminated compact sorted JSON SHA-256 is
`e3c427f212b8153a0fcfe0aa8f3400afa969e8ee630035da13403c64d6c2d2b1`;
the ledger-recorded package digest is
`6ef0c3ba6cc37a6b513209a0d830c541f5aab6f872b3eac7732cb8dc28945e2f`.

Project Control observation `2026-09-01T06:05:52Z` established:

- CellShard commit `62ccc87036355cec3b974eb2323782c54d9a75aa`;
- clean source-worktree fingerprint
  `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855`;
- Todo UUID `a52537a5-20db-4aeb-a126-dd0128c71fda`, revision `320`, and
  semantic/workflow fingerprint
  `08268de357d4e625841a5ac474443ea9a6fa657ebd6499f035f9fc74d82d963f`;
- zero skew among semantic state, workflow, status, and export providers; and
- B04 as the sole active task in the serial bootstrap lane, with B05 queued.

A separately timestamped pre-completion observation at
`2026-09-01T06:09:13Z` again reported revision `320`, the same source commit,
zero provider skew, and semantic/workflow fingerprint
`21bd49057014e5fb04f7c669ced1386dee271caa87b36cf9173733606f21d205`.
Its bounded source cursor identified only this baseline and its CSV companion,
with worktree fingerprint
`0e61329c1f2ae5e04dec73d6d4bfc9691aef665acd29614b8abd082c1353d5a2`.
A contemporaneous opaque-handle sync kept B04 claimed at revision `320`. The
stable-revision fingerprint change records liveness and scoped evidence
materialization; no recovery or unrelated task action was performed.

The authoritative interface records report all nine contracts below as
`frozen`. Interface content hashes are authority hashes, not reconstructed
hashes of concatenated source files.

## Frozen interface inventory

| Interface | Version | Owner | Authority content hash | Frozen contract |
| --- | ---: | --- | --- | --- |
| `CS-FOUND-I1` | 1 | `CS-FOUND-03` | `7ecf116c9568d646b180e28a59ddf1ab9adbd10017a29b2a26a12255842adb30` | Strong IDs, content digests, allocation-free array views, and status vocabulary under `identity.hh` and `identity/`. |
| `CS-FOUND-I2A` | 1 | `CS-FOUND-04` | `217a095db360a5ff8ac6fbc9edc9e20acbf2833463c7c988be124c6ef4efe4f3` | Domains, generations, partition maps, partitions, exact selections, orders, and bindings under `domain.hh` and `domain/`. |
| `CS-FOUND-I2B` | 1 | `CS-FOUND-05` | `7c86e72818699a566787ed1561d97f8448ce199974b827a81712f4ff5777696a` | Image/projection identity, target capabilities, reuse, sizes, alignment, payload digest, domains, dependencies, routes, and nonowning view. |
| `CS-FOUND-I2C` | 1 | `CS-FOUND-06` | `840ba657df09cfac3a186d23f088da5bfa99a1df5bbcf66c11cf6dae8052330c` | Storage objects, aligned extents, operational source locations, capabilities, and the nonowning exact-range payload-source boundary. |
| `CS-FOUND-I3A` | 1 | `CS-FOUND-07` | `b09fd2ca65112edfea3739199be618a0b2b547550733b0d01ce771ba658b0914` | Separate immutable artifact and operational source catalogs plus closed snapshot validation. |
| `CS-FOUND-I3B` | 1 | `CS-FOUND-08` | `32c45f1dc09643d711f04e7e9f086cda22f25a67b986c8e34e4bad3d97bd229a` | Portable deterministic CPEXEC02 image-envelope encode/decode contract. |
| `CS-FOUND-I4` | 2 | `CS-FOUND-09` | `32c45f1dc09643d711f04e7e9f086cda22f25a67b986c8e34e4bad3d97bd229a` | Atomic CPEXEC02 publication and metadata-only inspection within unchanged CSPACK01. |
| `CS-FOUND-I5A` | 1 | `CS-FOUND-10` | `3d5c4533eeb14b93bddc85067338cea5745790c81bff06407489411f902df82e` | Local exact source and one-allocation move-only host residency with digest validation. |
| `CS-FOUND-I5B` | 1 | `CS-FOUND-11` | `e5a76d38652886e16a8b7b06c3068c0fac5b1736b2216465d7e82d1386210b24` | Caller-allocator, caller-stream, move-only device residency with asynchronous staging. |

I3B and I4 intentionally share an authority content hash because they freeze
successive behavior around the same public `image_envelope.hh` contract. I4 is
version 2 and adds publication/inspection obligations; it does not mutate the
top-level CSPACK01 bytes.

## Exact current-source fingerprints

These SHA-256 values pin the live contract sources at the B04 source commit:

| Source | SHA-256 |
| --- | --- |
| `identity/strong_id.hh` | `f24b82ed7f7abcf4135fc18e5f2c16e3be51b01e94a319144df2e92c92eada24` |
| `identity/digest.hh` | `95227620750d1a627f7d5f7e600b574d0de420eaa8bcf6e263e4076fd3b45c8f` |
| `domain/descriptor.hh` | `5186c77a24685844f0c2019af5389057f7420a25bbe4636a6702477050b07c55` |
| `domain/partition.hh` | `9656dd95a5045cd39f503b2fd545abb049716c1c059757355ea4b8f65a2cb5ed` |
| `artifact/image.hh` | `4f14646aae32595b3286b30933bd11a2c105168a3a0ebedf1f2aa89e99580b59` |
| `artifact/extent.hh` | `2d651e6b37c1c54fa102e1bb97e1daf6c6fabb40938e36fcd813653669f87140` |
| `artifact/catalog.hh` | `ee83646e7359e8e4516f9558d6d0386bb59a4adcbc32eeaf2ff73c4c3ef2a523` |
| `artifact/snapshot.hh` | `e11f6ea07cd3794aab4970346b1c4690d94118557e29bbb2628e4adc8f3a1bb0` |
| `runtime/source/payload_source.hh` | `40afc1192d855e7463c15086d6c2e093e8d2947f362d825617593665f95c39d6` |
| `runtime/source/local_file_source.hh` | `cbceae04e90de8c76b4930a699aab45de394683a57e40c4bc015accce2bde2e8` |
| `runtime/residency/host.hh` | `f6401ad337e8c3f8930634244958de26ec67204abab19e01b2a4988711eedc47` |
| `runtime/residency/device.cuh` | `9a216a85e84d5a0eda701a19db139ea1326c096312fc741a2d4e9dd443315871` |
| `io/pack/image_envelope.hh` | `7d5abe21c34fab99273212cdc7ec0d966473237e79b9d52ab8699b1d5fe478c4` |

Paths are relative to `include/CellShard/`. Implementation and tests remain the
behavior authority where a header declares an out-of-line operation.

## Exact import obligations

An import is an explicit validation and mapping step. It is not a cast, alias,
field-copy convention, or permission to widen a frozen interface in place.

1. **I1 identity.** Preserve each legacy strong-ID type and numeric value as
   its declared legacy identity. A successor atom, species, evidence,
   materialization, replica, or resident identity receives an adjacent,
   namespace-qualified contract. Paths, pointers, devices, locations, and
   service/placement epochs never enter portable biological identity.
2. **I2A domains and partitions.** Validate domain/map/partition generations,
   element counts, ordinals, range bounds, non-overlap, exact selection count,
   and order. Import canonical coverage and recovery explicitly. Equal shape,
   range, or ordinal never establishes biological equivalence. An opaque
   selection digest still requires an external exact coverage certificate.
3. **I2B images.** Import a valid image as a physical-view candidate, not an
   atom or certified logical contribution. Preserve projection, target,
   producer ABI, structure, geometry, operation, encoding, sizes, alignment,
   reuse, digest, domain bindings, dependencies, and routes.
4. **I2C storage and sources.** Validate storage-object and extent size,
   bounds, alignment, digest, provider capability, and destination capacity.
   Extents express byte coverage, not biological ownership. Provider and
   locator remain mutable operational state outside immutable artifact identity.
5. **I3A catalogs and snapshots.** Revalidate uniqueness, generation closure,
   domain/map/partition bindings, extent totals, dependency acyclicity, route
   closure, and source reachability. Import into adjacent atom-store catalogs;
   do not silently append atom roots to the frozen snapshot identity.
6. **I3B/I4 CPEXEC02 and CSPACK.** Validate schema, endian marker, counts,
   sizes, bounds, checksum, identities, shard/table offsets, and payload digest
   before mapping CPEXEC02 metadata. Preserve the CSPACK01 magic and top-level
   offset table. Metadata-only inspection imports an extent without reading or
   allocating the payload.
7. **I5A host acquisition.** Perform an exact validated source read into one
   explicit aligned allocation and verify the payload digest. Allocation,
   provider context, descriptor, and path remain operational state.
8. **I5B device staging.** Use the caller's allocator, device, and stream;
   preserve image/digest/byte identity, enqueue the explicit asynchronous copy,
   restore device state, and clean up on failure. Device ordinal and pointer are
   never portable identity.

Every failed import returns a deterministic error and leaves the corresponding
CS-FOUND route available as the canonical/reference fallback. Import does not
delete or replace the legacy source.

## Format and runtime compatibility

- CSH5 remains the canonical metadata-rich archive and artifact-generation
  source.
- CSPACK01 remains the per-shard execution container. CPEXEC01 remains its
  native legacy envelope; CPEXEC02 remains its deterministic portable image
  envelope.
- CSHARD01 remains experimental standby.
- `sharded<T>`, local placement, access adapters, and direct CSH5/CPEXEC paths
  remain compatibility/reference machinery until owning migrations land with
  differential fixtures.
- The closed `cellshard-cpp-access-adapter-refactor` remains closed historical
  evidence; this baseline neither reopens nor duplicates it.
- CSG1, CPE2, CPK1, CSH5, CSPACK01, CPEXEC01, CPEXEC02, and CSHARD01 bytes are
  not silently reinterpreted. Changes require adjacent versions and explicit
  compatibility routes.

## Preserved validation surface

The existing focused targets registered in `CMakeLists.txt` remain the
independent compatibility references:

```text
cellShardFoundationIdentityTest
cellShardFoundationDomainTest
cellShardFoundationArtifactTest
cellShardPayloadSourceTest
cellShardFoundationSnapshotTest
cellShardImageEnvelopeTest
cellShardHostResidencyTest
cellShardFoundationLegacyAdapterTest
cellShardOpaqueImagePipelineTest
cellShardExecutionPayloadCpuTest
cellShardOpaqueImagePipelineCudaTest
cellShardDeviceResidencyTest
cellShardExecutionPayloadTest
```

Later successor importers must add deterministic valid and malformed/stale
fixtures, randomized properties, exact-oracle comparisons, and differential
checks against these routes. B04 changes no code or target, so running the
existing binaries would not add evidence beyond their frozen interface gates.

The exact standalone and embedded build entry points are:

```bash
cmake -S /home/tumlinson/Cellerator/components/CellShard \
  -B /home/tumlinson/Cellerator/components/CellShard/build \
  -DCELLSHARD_BUILD_TESTS=ON
cmake --build /home/tumlinson/Cellerator/components/CellShard/build \
  -j "$(nproc)"

cmake -S /home/tumlinson/Cellerator -B /home/tumlinson/Cellerator/build
cmake --build /home/tumlinson/Cellerator/build -j "$(nproc)"
```

`CS-JBC-B05` owns provider, registry, and integration path reservation. This
baseline does not claim or complete it.
