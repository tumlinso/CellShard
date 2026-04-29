#!/usr/bin/env python3
import argparse
import importlib
import importlib.util
import pathlib
import sys

import numpy as np


def import_cellshard(build_dir: str | None):
    try:
        return importlib.import_module("cellshard")
    except ModuleNotFoundError as exc:
        if "cellshard._cellshard" not in str(exc) or not build_dir:
            raise

    sys.modules.pop("cellshard", None)
    sys.path.insert(0, build_dir)
    extension = importlib.import_module("_cellshard")
    sys.modules["cellshard._cellshard"] = extension
    return importlib.import_module("cellshard")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", type=pathlib.Path)
    parser.add_argument("--build-dir", default=None)
    args = parser.parse_args()

    cellshard = import_cellshard(args.build_dir)
    ds = cellshard.open(str(args.dataset))

    native = ds.matrix()
    assert isinstance(native, cellshard.NativeMatrixView)
    assert isinstance(ds.matrix(format="native"), cellshard.NativeMatrixView)
    assert native.shape == (2, 4)
    assert native.layout == "blocked_ell"
    assert native.num_partitions == 1
    assert native.num_shards == 1

    selection = ds[:2]
    assert isinstance(selection, cellshard.NativeRowSelection)
    assert selection.shape == (2, 4)
    assert isinstance(ds.head(1), cellshard.NativeRowSelection)

    csr = ds.matrix(format="csr")
    assert isinstance(csr, cellshard.CsrMatrixExport)
    assert csr.rows == 2 and csr.cols == 4
    row_csr = selection.to_csr()
    assert row_csr.rows == 2 and row_csr.cols == 4

    part = ds.partition(0)
    assert isinstance(part, cellshard.BlockedEllPartition)
    assert part.rows == 2 and part.cols == 4 and part.nnz == 4
    assert part.block_size == 2 and part.ell_cols == 4
    assert part.row_block_count == 1 and part.ell_width_blocks == 2
    assert np.asarray(part.block_col_idx).dtype == np.uint32
    assert np.asarray(part.block_col_idx).tolist() == [[0, 1]]
    assert np.asarray(part.values_storage).dtype == np.uint16
    assert np.asarray(part.values_storage).shape == (2, 4)
    expected = np.array([[1.0, 0.0, 2.0, 0.0], [0.0, 3.0, 0.0, 4.0]], dtype=np.float32)
    np.testing.assert_allclose(np.asarray(part.values_float32()), expected)

    if importlib.util.find_spec("torch") is not None:
        torch_matrix = ds.matrix(format="torch")
        assert tuple(torch_matrix.shape) == (2, 4)
        assert bool(torch_matrix.is_sparse_csr)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
