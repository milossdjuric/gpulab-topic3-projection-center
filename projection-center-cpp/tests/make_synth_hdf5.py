"""Generate a tiny synthetic HDF5 input for smoke-testing forward_search
(both the CLI and the pybind backend), since the real lab dataset at
/lgrp/edu-2026-1-gpulab isn't reachable from a personal dev machine."""
import sys

import h5py
import numpy as np


def make_synth_hdf5(path, num_projs=4, H=64, W=64, seed=0):
    rng = np.random.default_rng(seed)
    projs = rng.random((num_projs, H, W)).astype(np.float32)
    with h5py.File(path, "w") as f:
        f["voxelSize"] = 1.0
        f["Volumen_num_xz"] = float(W)
        f["Volumen_num_y"] = float(H)
        f["SDD"] = 1000.0
        f["SOD"] = 700.0
        f["pixelSize"] = 0.5
        f["num_projs"] = float(num_projs)
        f["detector_width"] = float(W)
        f["detector_height"] = float(H)
        f["Angle"] = np.linspace(0, 2 * np.pi, num_projs)
        f["Projection"] = projs


if __name__ == "__main__":
    make_synth_hdf5(sys.argv[1] if len(sys.argv) > 1 else "synth.hdf5")
