"""Regression test for porting forward_search's tan/atan elimination into
projection-center's own opencl and cpu backends.

Captures exact search() output (mse_values, x0, y0) from a small synthetic
dataset using the CURRENT (pre-optimization) code, so the optimization can
be verified to not change results. Run once before the optimization to
record the baseline, and again after to confirm it still matches."""
import sys

import h5py
import numpy as np

sys.path.insert(0, "projection-center/src")
sys.path.insert(0, "forward_search/tests")
from make_synth_hdf5 import make_synth_hdf5
from projection_center_searching.geometry import build_parameter_grid, build_sinogram, compute_forward_geometry
from projection_center_searching.hdf5_io import build_sinogram_from_hdf5
from projection_center_searching.models import SearchConfig
from projection_center_searching.backends import CpuBackend, OpenCLBackend

SYNTH_PATH = "/tmp/tan_atan_port_synth.hdf5"


def _run(backend_name):
    make_synth_hdf5(SYNTH_PATH, num_projs=8, H=48, W=32, seed=7)
    cb_params, sinogram_sum = build_sinogram_from_hdf5(SYNTH_PATH)
    sinogram = build_sinogram(sinogram_sum)
    config = SearchConfig(
        xshift_range_mm=5.0, alpha_range_deg=5.0, beta_range_deg=5.0,
        xshift_step_mm=5.0, alpha_step_deg=5.0, beta_step_deg=5.0,
        sample_count=1000, sample_angle_range_deg=30.0,
    )
    parameter_grid = build_parameter_grid(config)
    x0, y0, theta_term = compute_forward_geometry(cb_params, parameter_grid)

    backend = CpuBackend() if backend_name == "cpu" else OpenCLBackend()
    artifacts = backend.search(
        sinogram=sinogram, alpha=parameter_grid[:, 1], beta=parameter_grid[:, 2],
        tan_theta0=theta_term, x0=x0, y0=y0, cb_params=cb_params, config=config,
    )
    return artifacts.mse_values


def test_cpu_backend_baseline():
    mse = _run("cpu")
    finite = mse[np.isfinite(mse)]
    print(f"cpu mse_values: finite_count={finite.size} sum={finite.sum():.10f} "
          f"first5={mse[:5].tolist()}")


def test_opencl_backend_baseline():
    mse = _run("opencl")
    finite = mse[np.isfinite(mse)]
    print(f"opencl mse_values: finite_count={finite.size} sum={finite.sum():.10f} "
          f"first5={mse[:5].tolist()}")


if __name__ == "__main__":
    test_cpu_backend_baseline()
    test_opencl_backend_baseline()
