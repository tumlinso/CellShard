from ._cellshard import (
    BlockedEllPartition,
    ClientSnapshotRef,
    CsrMatrixExport,
    CshardDescription,
    CshardFile,
    CshardTable,
    CshardTableColumn,
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
    NativeMatrixView,
    NativeRowSelection,
    ObservationMetadataColumn,
    PackDeliveryDescriptor,
    PackDeliveryRequest,
    RuntimeServiceMetadata,
    SlicedEllPartition,
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
    open_cshard,
    publish_runtime_service_cutover,
    serialize_global_metadata_snapshot,
    stage_append_only_runtime_service,
    validate_client_snapshot_ref,
    validate_cshard,
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
        "native": "native",
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
        raise ValueError("format must be one of: 'native', 'torch', 'scipy', or 'csr'")
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
        if normalized == "native":
            return NativeMatrixView(self.path)
        if normalized == "torch":
            return self._with_refresh(lambda: self._client.materialize_torch_sparse_csr())
        if normalized == "scipy":
            return self._with_refresh(lambda: self._client.materialize_scipy_csr())
        return self._with_refresh(lambda: self._client.materialize_csr())

    def _materialize_rows(self, row_indices, fmt: str):
        normalized = _normalize_format(fmt)
        if normalized == "native":
            return NativeRowSelection(self.path, row_indices)
        if normalized == "torch":
            return self._with_refresh(lambda: self._client.materialize_rows_torch_sparse_csr(row_indices))
        if normalized == "scipy":
            return self._with_refresh(lambda: self._client.materialize_rows_scipy_csr(row_indices))
        return self._with_refresh(lambda: self._client.materialize_rows_csr(row_indices))

    def matrix(self, format: str = "native"):
        return self._materialize(format)

    def rows(self, row_indices, format: str = "native"):
        if isinstance(row_indices, int):
            row_indices = [row_indices]
        return self._materialize_rows(row_indices, format)

    def head(self, count: int, format: str = "native"):
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
            return self.rows(range(start, stop, step), format="native")
        return self.rows(item, format="native")


class CshardDataset:
    """High-level facade for experimental standby .cshard archives."""

    def __init__(self, path: str):
        self._path = path
        self._file = open_cshard(path)

    @property
    def path(self) -> str:
        return self._path

    @property
    def obs(self):
        return self._file.obs()

    @property
    def var(self):
        return self._file.var()

    def describe(self) -> dict:
        desc = self._file.describe()
        return {
            "path": desc.path,
            "version": (desc.version_major, desc.version_minor),
            "shape": (desc.rows, desc.cols),
            "nnz": desc.nnz,
            "partitions": desc.partitions,
            "canonical_layout": desc.canonical_layout,
            "feature_order_hash": desc.feature_order_hash,
            "has_pack_manifest": desc.has_pack_manifest,
        }

    def read_rows(self, start: int, count: int, format: str = "csr"):
        if start < 0 or count < 0:
            raise ValueError("start and count must be non-negative")
        csr = self._file.read_rows(start, count)
        normalized = _normalize_format(format)
        if normalized == "torch":
            return csr.to_torch_sparse_csr()
        if normalized == "scipy":
            return csr.to_scipy_csr()
        if normalized == "native":
            raise ValueError(".cshard row reads expose CSR interop only; use format='csr', 'scipy', or 'torch'")
        return csr


def open(path: str) -> Dataset:
    if str(path).endswith(".cshard"):
        return CshardDataset(path)
    return Dataset(path)

__all__ = [
    "CsrMatrixExport",
    "ClientSnapshotRef",
    "BlockedEllPartition",
    "DatasetClient",
    "DatasetCodecSummary",
    "SourceDatasetSummary",
    "CshardDescription",
    "CshardFile",
    "CshardTable",
    "CshardTableColumn",
    "CshardDataset",
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
    "NativeMatrixView",
    "NativeRowSelection",
    "ObservationMetadataColumn",
    "PackDeliveryDescriptor",
    "PackDeliveryRequest",
    "RuntimeServiceMetadata",
    "SlicedEllPartition",
    "bootstrap_dataset_client",
    "deserialize_global_metadata_snapshot",
    "describe_pack_delivery",
    "load_dataset_as_csr",
    "load_dataset_global_metadata_snapshot",
    "load_dataset_rows_as_csr",
    "load_dataset_summary",
    "make_client_snapshot_ref",
    "open",
    "open_cshard",
    "open_dataset",
    "open_dataset_owner",
    "publish_runtime_service_cutover",
    "serialize_global_metadata_snapshot",
    "stage_append_only_runtime_service",
    "validate_client_snapshot_ref",
    "validate_cshard",
    "write_h5ad",
    "__version__",
]
