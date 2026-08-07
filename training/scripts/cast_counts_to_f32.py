"""Rewrite an anemoi obs_count store with a float32 data array.

nese-dataprep writes counts as int32, which is right for the data but wrong for the
join: numpy promotes int32 + float32 to float64, so the joined dataset hands the model
a float64 batch. anemoi overrides transfer_batch_to_device to only move the tensor, not
cast it, so Lightning's precision plugin never intervenes and the first Linear fails
with "mat1 and mat2 must have the same dtype".

The permanent fix belongs in nese-dataprep (write the counts float32). This converts an
existing store so the pipeline can run without re-binning from source.
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

    if counts.dtype == "float32":
        print(f"{dst} already float32")
        return

    tmp = store.create_dataset(
        "data_f32",
        shape=counts.shape,
        chunks=counts.chunks,
        dtype="float32",
        compressor=counts.compressor,
        overwrite=True,
    )
    for i in range(counts.shape[0]):
        tmp[i] = counts[i].astype("float32")

    del store["data"]
    store.move("data_f32", "data")

    print(f"{dst}: {counts.dtype} -> {zarr.open(str(dst), mode='r')['data'].dtype}")


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]))
