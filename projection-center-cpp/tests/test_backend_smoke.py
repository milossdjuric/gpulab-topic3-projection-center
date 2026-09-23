"""Smoke test: the pybind11 backend compiles, loads, and returns a sane
pose dict for a tiny synthetic dataset, using mode="buffer" (the only mode
validated to run on this dev machine's GPU - see memory env-opencl-gpu-setup.md
for the Image2D driver bug that rules out mode="image" here)."""
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import h5py
from make_synth_hdf5 import make_synth_hdf5
from backend import _backend

SYNTH_PATH = "/tmp/forward_search_test_synth.hdf5"


def main():
    make_synth_hdf5(SYNTH_PATH)
    with h5py.File(SYNTH_PATH, "r") as f:
        projs = f["Projection"][()]
        SDD, SOD, pixel_size = f["SDD"][()], f["SOD"][()], f["pixelSize"][()]
        W = int(f["detector_width"][()])
        H = int(f["detector_height"][()])

    result = _backend.search(
        projs, SDD, SOD, pixel_size, W, H,
        xshift=5.0, alpha=2.0, beta=2.0,
        xshift_step=5.0, alpha_step=2.0, beta_step=2.0,
        mode="buffer",
    )

    for key in ("xshift", "alpha", "beta", "MSE", "center_x", "center_y"):
        assert key in result, f"missing key: {key}"
        assert math.isfinite(result[key]), f"{key} is not finite: {result[key]}"

    print("test_backend_smoke: PASS", result)


if __name__ == "__main__":
    main()
