"""Rewrite an anemoi obs_count store with a float32 data array.

A migration tool for stores written before nese-dataprep's Anemoi target switched its
counts to float32. Stores produced after that change need no conversion; check the
dtype of `<store>/data` before reaching for this.

The problem it solves: int32 counts are right for the data but wrong for the join, where
numpy promotes int32 + float32 to float64, so the joined dataset hands the model a
float64 batch. anemoi overrides transfer_batch_to_device to only move the tensor, not
cast it, so Lightning's precision plugin never intervenes and the first Linear fails
with "mat1 and mat2 must have the same dtype".
"""

import shutil
import sys
from pathlib import Path

import zarr


def main(src: Path, dst: Path) -> None:
    if dst.exists():
        shutil.rmtree(dst)

    shutil.copytree(src, dst)
    store = zarr.open(str(dst), mode="a")
    counts = store["data"]
    src_dtype = counts.dtype

    if src_dtype != "float32":
        tmp = store.create_dataset(
            "data_f32",
            shape=counts.shape,
            chunks=counts.chunks,
            dtype="float32",
            compressor=counts.compressor,
            overwrite=True,
            # Same hazard the store itself guards against, and it only appears once the
            # array is float: zarr defaults fill_value to the dtype's zero, and xarray
            # reads fill_value as _FillValue and CF-decodes every match. Without this,
            # every zero count -- the vast majority of cells -- comes back NaN from
            # xr.open_zarr. anemoi reads through the raw zarr API and is unaffected,
            # which is exactly what makes it easy to miss.
            fill_value=None,
        )
        # create_dataset makes a bare array, so the source attributes have to be
        # carried over explicitly. _ARRAY_DIMENSIONS is what names the axes for
        # xarray; without it xarray refuses the store entirely.
        tmp.attrs.update(dict(counts.attrs))

        for i in range(counts.shape[0]):
            tmp[i] = counts[i].astype("float32")

        del store["data"]
        store.move("data_f32", "data")

    # copytree brought the source .zmetadata along, and neither create_dataset nor
    # move updates it, so the consolidated metadata still advertises the old dtype.
    # Readers that trust it -- xarray.open_zarr defaults to consolidated=True -- then
    # decode the float32 chunks as int32 and hand back integers. Rewrite it last, so
    # the consolidated view matches the .zarray files on disk.
    zarr.consolidate_metadata(store.store)

    check = zarr.open_consolidated(str(dst), mode="r")["data"].dtype
    print(f"{dst}: {src_dtype} -> {check} (consolidated metadata rewritten)")
    assert check == "float32", f"{dst}: consolidated dtype is {check}, expected float32"


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]))
