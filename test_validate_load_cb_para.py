"""Regression test: validate_forward_search.py's corrected-reference path
must derive detector_width/detector_height from Projection's real array
shape, not trust the file's stored detector_width/detector_height scalars
-- some dataset files (confirmed on data/proj_shepplogan512.hdf5) store
those two swapped relative to the actual array, which the old code took
at face value."""
import sys

import h5py
import numpy as np

sys.path.insert(0, ".")
from validate_forward_search import _load_cb_para

PATH = "/tmp/validate_test_mismatched_dims.hdf5"

# Real array: num_projs=2, height=10, width=6. Stored scalars are
# deliberately wrong and unrelated to a simple swap, so the test can only
# pass if the loader ignores them entirely and uses the array's real shape.
NUM_PROJS, H, W = 2, 10, 6
STORED_WIDTH, STORED_HEIGHT = 999, 888


def make_hdf5(path):
    rng = np.random.default_rng(0)
    projs = rng.random((NUM_PROJS, H, W)).astype(np.float32)
    with h5py.File(path, "w") as f:
        f["voxelSize"] = 1.0
        f["Volumen_num_xz"] = float(W)
        f["Volumen_num_y"] = float(H)
        f["SDD"] = 1000.0
        f["SOD"] = 700.0
        f["pixelSize"] = 0.5
        f["num_projs"] = float(NUM_PROJS)
        f["detector_width"] = float(STORED_WIDTH)
        f["detector_height"] = float(STORED_HEIGHT)
        f["Angle"] = np.linspace(0, 2 * np.pi, NUM_PROJS)
        f["Projection"] = projs
    return projs


def test_dims_derived_from_array_shape_not_stored_scalars():
    projs = make_hdf5(PATH)
    with h5py.File(PATH, "r") as f:
        cb_para = _load_cb_para(f, projs)

    assert cb_para["detector_width"] == W, (
        f"expected detector_width={W} (from array shape), "
        f"got {cb_para['detector_width']} (stored scalar was {STORED_WIDTH})"
    )
    assert cb_para["detector_height"] == H, (
        f"expected detector_height={H} (from array shape), "
        f"got {cb_para['detector_height']} (stored scalar was {STORED_HEIGHT})"
    )
    print(f"test_dims_derived_from_array_shape_not_stored_scalars: PASS "
          f"detector_width={cb_para['detector_width']} detector_height={cb_para['detector_height']}")


if __name__ == "__main__":
    test_dims_derived_from_array_shape_not_stored_scalars()
