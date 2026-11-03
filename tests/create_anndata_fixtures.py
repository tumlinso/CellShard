#!/usr/bin/env python3
"""Generate native H5AD fixtures with AnnData, then mutate rejection cases."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import anndata as ad
import h5py
import numpy as np
import pandas as pd
from scipy import sparse

if hasattr(ad.settings, "allow_write_nullable_strings"):
    ad.settings.allow_write_nullable_strings = True


def write_h5ad(path: Path, value: ad.AnnData) -> None:
    value.write_h5ad(path, convert_strings_to_categoricals=False)


def csr_integer(path: Path) -> None:
    obs_names = [f"cell-{i}" for i in range(6)]
    var_names = [f"gene-{i}" for i in range(5)]
    labels = ["a", None, "", "d", "e", "f"]
    label_column = (
        pd.array(labels, dtype="string")
        if hasattr(ad.settings, "allow_write_nullable_strings")
        else pd.Categorical(labels, categories=["a", "", "d", "e", "f"])
    )
    obs = pd.DataFrame(
        {
            "condition": ["control", "control", "treated", "treated", "control", "treated"],
            "label": label_column,
            "batch": np.array([1, 1, 2, 2, 3, 3], dtype=np.int16),
        },
        index=obs_names,
    )
    var = pd.DataFrame(
        {
            "gene_id": [f"ENSG{i:06d}" for i in range(5)],
            "symbol": [f"G{i}" for i in range(5)],
            "feature_name": [f"feature-name-{i}" for i in range(5)],
            "feature_type": ["Gene Expression" for _ in range(5)],
        },
        index=var_names,
    )

    # Preserve two duplicate coordinates, an explicit zero, and an empty row.
    x = sparse.csr_matrix(
        (
            np.array([2, 0, 3, 4, 5, 8, 9, 10], dtype=np.int32),
            np.array([1, 1, 4, 3, 3, 0, 2, 4], dtype=np.int32),
            np.array([0, 3, 3, 6, 7, 7, 8], dtype=np.int32),
        ),
        shape=(6, 5),
    )
    write_h5ad(path, ad.AnnData(X=x, obs=obs, var=var))


def csc_float(path: Path) -> None:
    rows, cols = 64, 4
    obs_names = [f"wide-cell-{i}" for i in range(rows)]
    var_names = [f"wide-gene-{i}" for i in range(cols)]
    obs = pd.DataFrame(
        {"condition": ["left" if i < 32 else "right" for i in range(rows)]},
        index=obs_names,
    )
    var = pd.DataFrame({"feature_id": [f"F{i}" for i in range(cols)]}, index=var_names)

    starts = (0, 8, 16, 24)
    ends = (38, 47, 55, 62)
    data: list[float] = []
    indices: list[int] = []
    indptr = [0]
    for col, (first, last) in enumerate(zip(starts, ends)):
        for row in range(first, last + 1):
            indices.append(row)
            data.append(0.0 if col == 0 and row == 0 else col + row / 100.0)
            if (col == 0 and row == 0) or (col == 3 and row == 24):
                indices.append(row)
                data.append(0.5 if col == 0 else 3.5)
        indptr.append(len(data))

    x = sparse.csc_matrix(
        (
            np.asarray(data, dtype=np.float32),
            np.asarray(indices, dtype=np.int32),
            np.asarray(indptr, dtype=np.int32),
        ),
        shape=(rows, cols),
    )
    write_h5ad(path, ad.AnnData(X=x, obs=obs, var=var))


def layers_raw(path: Path) -> None:
    obs_names = ["sample-a", "sample-b", "sample-c", "sample-d"]
    obs = pd.DataFrame(
        {"condition": ["untreated", "treated", "treated", "untreated"], "batch": [1, 1, 2, 2]},
        index=obs_names,
    )
    var = pd.DataFrame({"gene_id": ["X0", "X1", "X2", "X3"]}, index=["x0", "x1", "x2", "x3"])
    x = sparse.csr_matrix(
        (
            np.array([1.25, 2.5, 4.0], dtype=np.float32),
            np.array([0, 3, 1], dtype=np.int32),
            np.array([0, 1, 2, 3, 3], dtype=np.int32),
        ),
        shape=(4, 4),
    )
    value = ad.AnnData(X=x, obs=obs, var=var)
    value.layers["counts"] = sparse.csc_matrix(
        np.array(
            [
                [7, 0, 0, 0],
                [0, 8, 0, 0],
                [0, 0, 9, 0],
                [0, 0, 0, 10],
            ],
            dtype=np.int32,
        )
    )
    raw_var = pd.DataFrame(
        {"gene_id": ["R-A", "R-B", "R-C"], "symbol": ["raw-a", "raw-b", "raw-c"]},
        index=["raw-feature-a", "raw-feature-b", "raw-feature-c"],
    )
    raw_x = sparse.csr_matrix(
        (
            np.array([11, 12, 13, 14], dtype=np.int16),
            np.array([0, 1, 2, 0], dtype=np.int32),
            np.array([0, 1, 2, 3, 4], dtype=np.int32),
        ),
        shape=(4, 3),
    )
    value.raw = ad.AnnData(X=raw_x, obs=obs.copy(), var=raw_var)
    write_h5ad(path, value)


def empty_matrix(path: Path) -> None:
    obs = pd.DataFrame(index=["empty-cell-a", "empty-cell-b", "empty-cell-c"])
    var = pd.DataFrame(index=["empty-gene-a", "empty-gene-b"])
    x = sparse.csr_matrix(
        (
            np.array([], dtype=np.int32),
            np.array([], dtype=np.int32),
            np.zeros(4, dtype=np.int32),
        ),
        shape=(3, 2),
    )
    write_h5ad(path, ad.AnnData(X=x, obs=obs, var=var))


def dtype_branches(path: Path) -> None:
    obs = pd.DataFrame(index=["dtype-row-a", "dtype-row-b"])
    var = pd.DataFrame(index=["dtype-col-a", "dtype-col-b"])
    indices = np.array([0, 1], dtype=np.int32)
    indptr = np.array([0, 1, 2], dtype=np.int32)

    def csr(values: list[object], dtype: np.dtype) -> sparse.csr_matrix:
        return sparse.csr_matrix(
            (np.asarray(values, dtype=dtype), indices.copy(), indptr.copy()), shape=(2, 2)
        )

    value = ad.AnnData(X=csr([True, False], np.dtype(np.bool_)), obs=obs, var=var)
    layers = {
        "int8": (np.int8, [-3, 5]),
        "uint8": (np.uint8, [3, 5]),
        "int16": (np.int16, [-3, 5]),
        "uint16": (np.uint16, [3, 5]),
        "int32": (np.int32, [-3, 5]),
        "uint32": (np.uint32, [3, 5]),
        "int64": (np.int64, [-3, 5]),
        "uint64": (np.uint64, [3, 5]),
        "float32": (np.float32, [1.25, -2.5]),
        "float64": (np.float64, [1.25, -2.5]),
    }
    for name, (dtype, data) in layers.items():
        value.layers[name] = csr(data, np.dtype(dtype))
    write_h5ad(path, value)


def dense_matrix(path: Path) -> None:
    obs = pd.DataFrame(index=["dense-cell-a", "dense-cell-b"])
    var = pd.DataFrame(index=["dense-gene-a", "dense-gene-b"])
    x = np.array([[1.0, 0.0], [0.0, 2.0]], dtype=np.float32)
    write_h5ad(path, ad.AnnData(X=x, obs=obs, var=var))


def set_text_attribute(obj: h5py.Group | h5py.File, name: str, value: str) -> None:
    if name in obj.attrs:
        del obj.attrs[name]
    obj.attrs.create(name, value, dtype=h5py.string_dtype(encoding="utf-8"))


def mutate_rejections(output_dir: Path) -> None:
    source = output_dir / "csr_integer.h5ad"

    legacy = output_dir / "legacy_unversioned.h5ad"
    shutil.copyfile(source, legacy)
    with h5py.File(legacy, "r+") as h5:
        for key in ("encoding-type", "encoding-version"):
            if key in h5.attrs:
                del h5.attrs[key]
        x = h5["X"]
        shape = np.asarray(x.attrs["shape"], dtype=np.uint64)
        for key in ("encoding-type", "encoding-version", "shape"):
            if key in x.attrs:
                del x.attrs[key]
        set_text_attribute(x, "h5sparse_format", "csr")
        x.attrs["h5sparse_shape"] = shape

    unversioned = output_dir / "modern_unversioned.h5ad"
    shutil.copyfile(source, unversioned)
    with h5py.File(unversioned, "r+") as h5:
        del h5["X"].attrs["encoding-version"]

    unsupported = output_dir / "unsupported_version.h5ad"
    shutil.copyfile(source, unsupported)
    with h5py.File(unsupported, "r+") as h5:
        set_text_attribute(h5["X"], "encoding-version", "99.0.0")

    unsupported_encoding = output_dir / "unsupported_encoding.h5ad"
    shutil.copyfile(source, unsupported_encoding)
    with h5py.File(unsupported_encoding, "r+") as h5:
        set_text_attribute(h5["X"], "encoding-type", "unknown_sparse_encoding")


def validate_fixtures(output_dir: Path) -> None:
    with h5py.File(output_dir / "csr_integer.h5ad", "r") as h5:
        x = h5["X"]
        if x.attrs["encoding-type"] not in (b"csr_matrix", "csr_matrix"):
            raise RuntimeError("csr_integer.h5ad did not retain a modern CSR X")
        if h5["X/data"].dtype != np.dtype(np.int32):
            raise RuntimeError("csr_integer.h5ad did not retain int32 values")
        data = h5["X/data"][:]
        indices = h5["X/indices"][:]
        indptr = h5["X/indptr"][:]
        if data.size != 8 or np.count_nonzero(data == 0) != 1:
            raise RuntimeError("CSR fixture lost its explicit zero or stored-entry count")
        if int((indices[indptr[0] : indptr[1]] == 1).sum()) != 2:
            raise RuntimeError("CSR fixture lost its duplicate coordinate")
        if indptr[2] != indptr[1]:
            raise RuntimeError("CSR fixture no longer contains an empty row")

    with h5py.File(output_dir / "csc_float.h5ad", "r") as h5:
        x = h5["X"]
        actual_encoding = x.attrs["encoding-type"]
        if isinstance(actual_encoding, bytes):
            actual_encoding = actual_encoding.decode("utf-8")
        if actual_encoding != "csc_matrix" or h5["X/data"].dtype != np.dtype(np.float32):
            raise RuntimeError("CSC fixture lost its modern encoding or float32 values")
        if h5["X/data"].shape != (160,) or np.count_nonzero(h5["X/data"][:] == 0) != 1:
            raise RuntimeError("CSC fixture lost its explicit zero or stored-entry count")

    with h5py.File(output_dir / "layers_raw.h5ad", "r") as h5:
        if h5["layers/counts"].attrs["encoding-type"] not in (b"csc_matrix", "csc_matrix"):
            raise RuntimeError("layers/counts is not CSC")
        if h5["raw/X"].attrs["encoding-type"] not in (b"csr_matrix", "csr_matrix"):
            raise RuntimeError("raw/X is not CSR")
        if list(h5["raw/X"].attrs["shape"]) != [4, 3]:
            raise RuntimeError("raw/X did not keep an independent three-feature var axis")

    with h5py.File(output_dir / "empty.h5ad", "r") as h5:
        if h5["X/data"].shape != (0,) or h5["X/indptr"].shape != (4,):
            raise RuntimeError("empty fixture did not retain a zero-NNZ CSR matrix")

    with h5py.File(output_dir / "dtype_branches.h5ad", "r") as h5:
        if h5["X/data"].dtype != np.dtype(np.bool_):
            raise RuntimeError("dtype fixture did not retain bool X")
        for name, dtype in (
            ("int8", np.int8), ("uint8", np.uint8),
            ("int16", np.int16), ("uint16", np.uint16),
            ("int32", np.int32), ("uint32", np.uint32),
            ("int64", np.int64), ("uint64", np.uint64),
            ("float32", np.float32), ("float64", np.float64),
        ):
            if h5[f"layers/{name}/data"].dtype != np.dtype(dtype):
                raise RuntimeError(f"dtype fixture lost the {name} layer type")

    with h5py.File(output_dir / "dense.h5ad", "r") as h5:
        if not isinstance(h5["X"], h5py.Dataset):
            raise RuntimeError("dense fixture X is not a dense HDF5 dataset")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_dir", type=Path)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    csr_integer(args.output_dir / "csr_integer.h5ad")
    csc_float(args.output_dir / "csc_float.h5ad")
    layers_raw(args.output_dir / "layers_raw.h5ad")
    empty_matrix(args.output_dir / "empty.h5ad")
    dtype_branches(args.output_dir / "dtype_branches.h5ad")
    dense_matrix(args.output_dir / "dense.h5ad")
    mutate_rejections(args.output_dir)
    validate_fixtures(args.output_dir)


if __name__ == "__main__":
    main()
