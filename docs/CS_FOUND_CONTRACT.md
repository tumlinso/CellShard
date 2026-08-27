# CS-FOUND Contract

CellShard is a performance-first, biology-informed physical-runtime library.
It may carry explicit cell, gene, sequence, ordering, projection, reuse, and
target facts when they improve physical realization, but producer libraries
retain the biological theory and mathematical operation semantics.

CS-FOUND establishes strong identities; explicit domain, partition-map,
selection, order, projection, image, storage-object, extent, source, catalog,
snapshot, and residency contracts; deterministic `CPEXEC02`; and an opaque
image path from `CSPACK01` through local storage to host and caller-allocated
device memory. Producer payload bytes remain opaque. Operational metadata is
explicit, while paths, replicas, service epochs, and placement epochs remain
outside immutable image identity.

The foundational load path performs one source-to-host payload allocation and
one caller-owned device allocation plus one `cudaMemcpyAsync`. This is a
validated vertical slice, not a rule for every future streaming engine.

`CPEXEC01` and legacy row shards remain compatibility machinery. A row shard is
not a biological domain or partition, and adapters require caller-supplied new
identities rather than manufacturing them from legacy counters.

`.csh5` remains the current canonical durable dataset and append source.
`.cspack` is a generated execution-artifact family, `.cspool` is a bounded local
ingest artifact, and `.cshard` remains experimental. Durable or precomputed
artifacts support fast execution; `.csh5` is not assumed to be the permanent
hot execution substrate.

Networking is out of scope. CS-FOUND also does not implement collectives,
topology discovery, distributed scheduling, a streaming engine, cache policy,
biological route compilation, dynamic representation/precision selection, or
the eventual physical-runtime optimizer.

The next investigation should examine blocked disk streaming, working-set
construction, representation and residency choice, movement versus
recomputation, accelerator and storage topology, placement, biological
locality, multi-device execution, and eventual multi-node realization. It is
not created or named by CS-FOUND.
