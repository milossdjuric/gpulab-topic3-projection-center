"""Regression test: computeCOR() must expose real device-side GPU kernel
execution time (via OpenCL event profiling), not just host-side wall-clock
stage timestamps. Without this, there's no way to measure whether a kernel
optimization actually helped -- whole-program wall time is dominated by
HDF5 I/O and CPU-side sinogram building, not the ~48ms search kernel
itself (see docs/ARCHITECTURE.md's report-mapping section)."""
import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

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

    assert "kernel_ms" in result, "missing key: kernel_ms"
    assert math.isfinite(result["kernel_ms"]), f"kernel_ms is not finite: {result['kernel_ms']}"
    assert result["kernel_ms"] > 0.0, f"kernel_ms should be positive, got {result['kernel_ms']}"
    assert result["kernel_ms"] < 5000.0, f"kernel_ms implausibly large for tiny synthetic data: {result['kernel_ms']}"

    print("test_kernel_timing: PASS", result["kernel_ms"], "ms")


if __name__ == "__main__":
    main()
