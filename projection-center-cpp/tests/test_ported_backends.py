"""Regression check for the ported backends: _backend.search()/resample()
with backend="opencl" and backend="cpu" are C++ ports of
projection-center-python-opencl/'s OpenCLBackend (PyOpenCL) and CpuBackend
(NumPy), so for the same input they must find the same pose and produce
the same resampled projections (up to float rounding) as the Python
originals.

Also checks kernels/python_opencl_port.cl is still an exact copy of the
Python package's KERNEL_SOURCE, so the two can't silently drift apart.

Runs on the course's own 128px Shepp-Logan dataset (small enough for the
NumPy CPU path). Usage, from projection-center-cpp/:

    python3 tests/test_ported_backends.py [--data ../data/proj_shepplogan128.hdf5]
"""
import argparse
import math
import os
import sys
import tempfile

FORWARD_SEARCH_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPO_ROOT = os.path.dirname(FORWARD_SEARCH_DIR)
sys.path.insert(0, FORWARD_SEARCH_DIR)
sys.path.insert(0, os.path.join(REPO_ROOT, "projection-center-python-opencl", "src"))

import h5py
import numpy as np
from backend import _backend
from projection_center_searching.backends import KERNEL_SOURCE
from projection_center_searching.models import ResampleConfig, SearchConfig
from projection_center_searching.pipeline import run_resample, run_search

# Resampled pixels may differ by float32 rounding (different summation /
# instruction order), never by more than this, relative to the data range.
RESAMPLE_REL_TOL = 1e-4


def check_kernel_copy():
    with open(os.path.join(FORWARD_SEARCH_DIR, "kernels", "python_opencl_port.cl")) as f:
        assert f.read() == KERNEL_SOURCE, \
            "kernels/python_opencl_port.cl differs from backends.py's KERNEL_SOURCE -- recopy it"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=os.path.join(REPO_ROOT, "data", "proj_shepplogan128.hdf5"))
    args = ap.parse_args()

    check_kernel_copy()

    with h5py.File(args.data, "r") as f:
        projs = f["Projection"][()]
        SDD, SOD, px = f["SDD"][()], f["SOD"][()], f["pixelSize"][()]
    H, W = projs.shape[1:]
    config = SearchConfig()

    with tempfile.TemporaryDirectory() as tmp:
        for backend in ("opencl", "cpu"):
            # Python original, through its own pipeline functions.
            pose_path = os.path.join(tmp, f"{backend}_pose.json")
            out_path = os.path.join(tmp, f"{backend}_resampled.h5")
            py_pose = run_search(args.data, pose_path, config, backend_name=backend)
            run_resample(args.data, pose_path, out_path, ResampleConfig(), backend_name=backend)
            with h5py.File(out_path, "r") as f:
                py_projs = f["Projection"][()]
                py_geo = (f["SDD"][()], f["SOD"][()], f["pixelSize"][()])

            # C++ port, through the pybind11 interface.
            pose = _backend.search(
                projs, SDD, SOD, px, W, H,
                xshift=config.xshift_range_mm, alpha=config.alpha_range_deg, beta=config.beta_range_deg,
                xshift_step=config.xshift_step_mm, alpha_step=config.alpha_step_deg,
                beta_step=config.beta_step_deg, backend=backend,
            )
            res = _backend.resample(
                projs, SDD, SOD, px,
                xshift=pose["xshift"], alpha=pose["alpha"], beta=pose["beta"],
                center_x=pose["center_x"], center_y=pose["center_y"], backend=backend,
            )

            assert math.isclose(pose["xshift"], py_pose.xshift, abs_tol=1e-12), (backend, pose, py_pose)
            assert math.isclose(pose["alpha"], py_pose.alpha, abs_tol=1e-12), (backend, pose, py_pose)
            assert math.isclose(pose["beta"], py_pose.beta, abs_tol=1e-12), (backend, pose, py_pose)
            assert math.isclose(pose["MSE"], py_pose.mse, rel_tol=1e-4), (backend, pose["MSE"], py_pose.mse)
            assert math.isclose(pose["center_x"], py_pose.center_point[0], rel_tol=1e-9, abs_tol=1e-9)
            assert math.isclose(pose["center_y"], py_pose.center_point[1], rel_tol=1e-9, abs_tol=1e-9)

            assert res["projections"].shape == py_projs.shape
            for got, want in zip((res["SDD"], res["SOD"], res["pixel_size"]), py_geo):
                assert math.isclose(got, want, rel_tol=1e-12), (backend, got, want)
            max_diff = float(np.abs(res["projections"].astype(np.float64) - py_projs).max())
            data_range = float(py_projs.max() - py_projs.min())
            assert max_diff <= RESAMPLE_REL_TOL * data_range, (backend, max_diff, data_range)

            print(f"{backend}: pose matches (MSE {pose['MSE']:.6e} vs {py_pose.mse:.6e}), "
                  f"resample max diff {max_diff:.3g} (range {data_range:.3g}), "
                  f"identical: {np.array_equal(res['projections'], py_projs)}")

    print("test_ported_backends: PASS")


if __name__ == "__main__":
    main()
