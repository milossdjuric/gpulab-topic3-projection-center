"""Regression test: benchmark_backends.py's reference search must run the
same in-process-corrected reference validate_forward_search.py already uses
(zero-init-on-oob + detector-dims-from-array-shape), instead of subprocessing
reference/Topic_3_forwardsearching.py directly -- so it no longer crashes on
datasets that trigger either of that file's two known bugs. The file on disk
is never touched; the correction is applied at call time, same as --fix-ref.

Uses a synthetic file with swapped detector_width/detector_height scalars
(mirrors the real data/proj_shepplogan512.hdf5 bug) and a tiny search grid
(3x3x3=27 combos) so the test runs in well under a second."""
import os
import sys

import h5py
import numpy as np

sys.path.insert(0, ".")
from benchmark_backends import run_reference_search

PATH = "/tmp/benchmark_test_swapped_dims.hdf5"
OUT_DIR = "/tmp/benchmark_test_swapped_dims_out"

NUM_PROJS, H, W = 4, 10, 6
# Deliberately swapped relative to the real array, mirroring the actual bug.
STORE_WIDTH, STORE_HEIGHT = H, W


def make_hdf5(path):
    rng = np.random.default_rng(1)
    projs = rng.random((NUM_PROJS, H, W)).astype(np.float32)
    with h5py.File(path, "w") as f:
        f["voxelSize"] = 1.0
        f["Volumen_num_xz"] = float(W)
        f["Volumen_num_y"] = float(H)
        f["SDD"] = 1000.0
        f["SOD"] = 700.0
        f["pixelSize"] = 0.5
        f["num_projs"] = float(NUM_PROJS)
        f["detector_width"] = float(STORE_WIDTH)
        f["detector_height"] = float(STORE_HEIGHT)
        f["Angle"] = np.linspace(0, 2 * np.pi, NUM_PROJS)
        f["Projection"] = projs


def test_reference_search_survives_swapped_dims():
    make_hdf5(PATH)
    os.makedirs(OUT_DIR, exist_ok=True)

    elapsed, pose = run_reference_search(
        PATH, OUT_DIR, xshift=5.0, alpha=5.0, beta=5.0,
        xshift_step=5.0, alpha_step=5.0, beta_step=5.0,
    )

    assert elapsed > 0
    for key in ("xshift", "alpha", "beta"):
        assert key in pose, f"missing key: {key}"
    assert "MSE" in pose or "mse" in pose, "missing MSE/mse key"

    print(f"test_reference_search_survives_swapped_dims: PASS  pose={pose}")


if __name__ == "__main__":
    test_reference_search_survives_swapped_dims()
