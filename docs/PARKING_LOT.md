# Parking Lot

This file holds ideas that may be valuable later but should not distract from the current release target.

The current priority is:

> Define and implement `.cspack` v1 as the CSPACK format for CellShard and Cellerator.

Anything that does not directly help `.cspack` v1 become inspectable, loadable, and executable belongs here until the current milestone is complete.

---

## Current Milestone

### `.cspack` v1

Goal:

```text
.csh5 -> .cspack -> GPU execution
```

The first successful version should:

- Define the `.cspack` v1 binary contract.
- Store Blocked-ELL execution payloads.
- Support FP16 values.
- Support `uint32` block column indices.
- Use little-endian encoding.
- Use 64-bit offsets.
- Use fixed payload alignment.
- Include a metadata table.
- Include a section directory.
- Include shard and partition directories.
- Include generation identifiers.
- Include a feature-order hash.
- Load without touching HDF5.
- Run one Blocked-ELL SpMM benchmark from `.cspack`.

Everything else goes below.

---

## Format Ideas To Defer

### `.cshard` native archive format

A future HDF5-free durable archive format.

Do not implement until `.cspack` v1 is stable.

Potential features:

- Native canonical sparse storage.
- Columnar observation metadata.
- Columnar feature metadata.
- Generation table.
- Preprocessing provenance.
- Pack manifests.
- `.h5ad` export support.
- `.cspack` regeneration support.

---

### Quantized `.cspack`

A future CSPACK extension.

Potential features:

- 1-bit, 2-bit, 4-bit, and 8-bit Blocked-ELL values.
- Per-feature affine scaling.
- Per-module scaling.
- Row offsets.
- Column scales.
- Fused decode + SpMM kernels.
- Quantization calibration metadata.
- Biological drift metrics.

Do not implement until plain FP16 `.cspack` works end-to-end.

---

### Sliced ELL support

May be useful as a simpler or compatibility-oriented execution layout.

Do not include in `.cspack` v1.

Potential uses:

- Validation against Blocked-ELL.
- Simpler kernels.
- Intermediate conversion path.
- Benchmark baseline.

---

### CSR runtime support inside `.cspack`

CSR may be useful as a fallback or debugging layout.

Do not make it part of the primary execution path for v1.

Potential uses:

- Export to SciPy.
- Export to Torch sparse.
- Validation.
- Debugging.
- Preprocessing algorithms that naturally need CSR semantics.

---

### Distributed pack layout

Future support for multi-node or service-based execution.

Do not include in `.cspack` v1.

Potential features:

- Multi-file pack manifests.
- Shard ownership tables.
- Service epochs.
- Remote shard refresh.
- Reader/writer coordination.
- Lock-free generation refresh.
- Multi-process worker coordination.

---

### Trajectory-native sections

Future biological extension.

Potential sections:

- Developmental time.
- Forward-neighbor graph.
- Cell-state graph.
- Lineage scaffold.
- Pseudotime ordering.
- RNA velocity fields.
- Branch probabilities.
- Terminal-state annotations.

Do not include until sparse projection from `.cspack` works.

---

### Forward-neighbor index format

Future execution index.

Potential files or sections:

- `.fnindex`
- latent vectors
- centroid tables
- shard-local neighbor indexes
- top-k workspaces
- row-to-time maps
- forward-only search windows

Do not implement until the base runtime can stage `.cspack` partitions cleanly.

---

### Learned sparse packing

Interesting research direction, not a v1 file-format task.

Potential approaches:

- Sparse regression to group rows/columns into blocks.
- Autoencoder-based sparse layout learning.
- Correlation-based block assignment.
- Regulatory-module-aware packing.
- Differentiable packing objective.

Do not implement before a deterministic Blocked-ELL builder exists.

---

### Biology-aware quantization

Future research direction.

Potential approaches:

- Per-gene scale/offset tables.
- Per-regulatory-module scaling.
- Binary activity + residual encoding.
- TF/module-aware quantization groups.
- Expression-range-aware calibration.
- Preservation of marker gene rankings.

Do not implement before generic quantized Blocked-ELL has an honest benchmark.

---

### Torch custom autograd

Useful, but not the first release target.

Potential features:

- Custom sparse projection autograd.
- Fused sparse preprocessing ops.
- Torch extension wrappers.
- Device-resident gradients.
- No dense fallback in primary path.

Do not implement until Cellerator can consume `.cspack` directly.

---

### Full model library

Do not build a large model zoo early.

Possible future models:

- Developmental projector.
- Forward-neighbor model.
- Sparse VAE.
- Regulatory module encoder.
- Chromatin peak encoder.
- Multimodal RNA/ATAC projector.
- Trajectory transition model.

For the first public workflow, choose one flagship model only.

---

### Rich biological metadata

Useful for archives, but not for `.cspack` v1.

Potential metadata:

- Cell type labels.
- Batch annotations.
- Embryo/sample identifiers.
- Experimental protocol.
- Tissue/stage labels.
- Feature annotations.
- Gene symbols and Ensembl IDs.
- Peak coordinates.
- Genome build.
- Modality-specific metadata.

Keep rich metadata in `.csh5` now and `.cshard` later.

`.cspack` should only contain metadata required for correct execution and reproducibility.

---

### Zarr support

Useful for ecosystem compatibility.

Do not prioritize until `.cspack` is stable.

Potential uses:

- Cloud-native datasets.
- Chunked array interoperability.
- Large public dataset import.
- Browser or remote object store support.

---

### MuData support

Useful for multimodal biology.

Do not prioritize until the base single-modality pack works.

Potential uses:

- RNA + ATAC.
- RNA + protein.
- RNA + chromatin marks.
- Cross-modality feature maps.
- Shared observation tables.

---

### GPU-direct storage

Interesting performance direction.

Do not prioritize for first release.

Potential features:

- GPUDirect Storage.
- Direct file-to-device staging.
- Async staged shard reads.
- Pinned host staging buffers.
- Overlapped I/O and compute.

First make ordinary `.cspack` loading correct.

---

### Multi-GPU scheduling

Important later, but keep out of v1.

Potential features:

- One shard per GPU.
- NVLink-aware shard placement.
- Device-local work queues.
- NCCL reductions.
- Cross-GPU neighbor merge.
- Topology-aware partitioning.

First make one-GPU execution boring and reliable.

---

## Paper / Documentation Ideas

### Contribution framing

Possible framing:

> A GPU-native sparse execution substrate for large omics matrices, with an CSPACK format optimized for Blocked-ELL sparse projection and biological model workflows.

---

### Core comparison

Potential baselines:

- `.h5ad` + Scanpy/SciPy.
- `.h5ad` + PyTorch sparse.
- `.csh5` current path.
- `.cspack` execution path.
- CSR vs Blocked-ELL.
- FP16 Blocked-ELL vs quantized Blocked-ELL later.

---

### Biological demo candidates

Choose one later:

- Mouse embryo developmental projection.
- Forward-neighbor trajectory scaffold.
- Sparse RNA projection.
- Chromatin peak projection.
- Regulatory module reconstruction.
- Multimodal RNA/ATAC sparse encoding.

Do not build all of them for the first release.

---

## Rules For Using This File

When a new idea appears, ask:

1. Does it help `.cspack` v1 become inspectable?
2. Does it help `.cspack` v1 become loadable?
3. Does it help `.cspack` v1 execute Blocked-ELL SpMM?
4. Does it help document the current format contract?

If not, put it here.

The goal is not to kill ideas.

The goal is to stop good ideas from destroying the current milestone.
