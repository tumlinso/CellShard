from ._cellshard import (
    ClientSnapshotRef,
    CsrMatrixExport,
    DatasetClient,
    DatasetCodecSummary,
    SourceDatasetSummary,
    DatasetFile,
    DatasetOwner,
    DatasetPartitionSummary,
    DatasetShardSummary,
    DatasetSummary,
    EmbeddedMetadataTable,
    ExecutionPartitionMetadata,
    ExecutionShardMetadata,
    GlobalMetadataSnapshot,
    ObservationMetadataColumn,
    PackDeliveryDescriptor,
    PackDeliveryRequest,
    RuntimeServiceMetadata,
    bootstrap_dataset_client,
    deserialize_global_metadata_snapshot,
    describe_pack_delivery,
    load_dataset_as_csr,
    load_dataset_global_metadata_snapshot,
    load_dataset_rows_as_csr,
    load_dataset_summary,
    make_client_snapshot_ref,
    open_dataset,
    open_dataset_owner,
    publish_runtime_service_cutover,
    serialize_global_metadata_snapshot,
    stage_append_only_runtime_service,
    validate_client_snapshot_ref,
    write_h5ad,
)

__version__ = "0.1.0"


_STALE_CLIENT_MARKERS = (
    "stale",
    "snapshot_id does not match the owner snapshot",
)


def _is_stale_client_error(exc: Exception) -> bool:
    message = str(exc)
    return any(marker in message for marker in _STALE_CLIENT_MARKERS)


def _normalize_format(fmt: str) -> str:
    normalized = fmt.strip().lower().replace("-", "_")
    aliases = {
        "torch": "torch",
        "torch_csr": "torch",
        "torch_sparse_csr": "torch",
        "scipy": "scipy",
        "scipy_csr": "scipy",
        "csr": "csr",
        "numpy": "csr",
        "numpy_csr": "csr",
    }
    if normalized not in aliases:
        raise ValueError("format must be one of: 'torch', 'scipy', or 'csr'")
    return aliases[normalized]


class Dataset:
    """High-level Python facade over the owner/client CellShard retrieval path."""

    def __init__(self, path: str):
        self._path = path
        self._owner = open_dataset_owner(path)
        self._client = bootstrap_dataset_client(self._owner)

    @property
    def path(self) -> str:
        return self._path

    @property
    def snapshot(self):
        return self._client.snapshot

    @property
    def summary(self):
        return self.snapshot.summary

    @property
    def shape(self):
        return (self.summary.rows, self.summary.cols)

    @property
    def rows_count(self) -> int:
        return self.summary.rows

    @property
    def cols_count(self) -> int:
        return self.summary.cols

    @property
    def nnz(self) -> int:
        return self.summary.nnz

    @property
    def obs_names(self):
        return self.summary.obs_names

    @property
    def var_names(self):
        return self.summary.var_names

    @property
    def var_ids(self):
        return self.summary.var_ids

    def refresh(self):
        self._client = bootstrap_dataset_client(self._owner)
        return self

    def describe(self) -> dict:
        runtime = self.snapshot.runtime_service
        return {
            "path": self.path,
            "shape": self.shape,
            "nnz": self.nnz,
            "matrix_format": self.summary.matrix_format,
            "payload_layout": self.summary.payload_layout,
            "num_partitions": self.summary.num_partitions,
            "num_shards": self.summary.num_shards,
            "service_mode": runtime.service_mode,
            "canonical_generation": runtime.canonical_generation,
            "execution_plan_generation": runtime.execution_plan_generation,
            "pack_generation": runtime.pack_generation,
            "service_epoch": runtime.service_epoch,
        }

    def _with_refresh(self, fn):
        try:
            return fn()
        except RuntimeError as exc:
            if not _is_stale_client_error(exc):
                raise
            self.refresh()
            return fn()

    def _materialize(self, fmt: str):
        normalized = _normalize_format(fmt)
        if normalized == "torch":
            return self._with_refresh(lambda: self._client.materialize_torch_sparse_csr())
        if normalized == "scipy":
            return self._with_refresh(lambda: self._client.materialize_scipy_csr())
        return self._with_refresh(lambda: self._client.materialize_csr())

    def _materialize_rows(self, row_indices, fmt: str):
        normalized = _normalize_format(fmt)
        if normalized == "torch":
            return self._with_refresh(lambda: self._client.materialize_rows_torch_sparse_csr(row_indices))
        if normalized == "scipy":
            return self._with_refresh(lambda: self._client.materialize_rows_scipy_csr(row_indices))
        return self._with_refresh(lambda: self._client.materialize_rows_csr(row_indices))

    def matrix(self, format: str = "torch"):
        return self._materialize(format)

    def rows(self, row_indices, format: str = "torch"):
        return self._materialize_rows(row_indices, format)

    def head(self, count: int, format: str = "torch"):
        if count < 0:
            raise ValueError("count must be non-negative")
        return self.rows(range(min(count, self.rows_count)), format=format)

    def partition(self, partition_id: int):
        return open_dataset(self.path).materialize_partition(partition_id)

    def write_h5ad(self, output_path: str):
        return write_h5ad(self.path, output_path)

    def __getitem__(self, item):
        if isinstance(item, slice):
            start, stop, step = item.indices(self.rows_count)
            return self.rows(range(start, stop, step), format="torch")
        return self.rows(item, format="torch")


def open(path: str) -> Dataset:
    return Dataset(path)

__all__ = [
    "CsrMatrixExport",
    "ClientSnapshotRef",
    "DatasetClient",
    "DatasetCodecSummary",
    "SourceDatasetSummary",
    "Dataset",
    "DatasetFile",
    "DatasetOwner",
    "DatasetPartitionSummary",
    "DatasetShardSummary",
    "DatasetSummary",
    "EmbeddedMetadataTable",
    "ExecutionPartitionMetadata",
    "ExecutionShardMetadata",
    "GlobalMetadataSnapshot",
    "ObservationMetadataColumn",
    "PackDeliveryDescriptor",
    "PackDeliveryRequest",
    "RuntimeServiceMetadata",
    "bootstrap_dataset_client",
    "deserialize_global_metadata_snapshot",
    "describe_pack_delivery",
    "load_dataset_as_csr",
    "load_dataset_global_metadata_snapshot",
    "load_dataset_rows_as_csr",
    "load_dataset_summary",
    "make_client_snapshot_ref",
    "open",
    "open_dataset",
    "open_dataset_owner",
    "publish_runtime_service_cutover",
    "serialize_global_metadata_snapshot",
    "stage_append_only_runtime_service",
    "validate_client_snapshot_ref",
    "write_h5ad",
    "__version__",
]
