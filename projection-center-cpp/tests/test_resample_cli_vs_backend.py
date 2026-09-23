"""Regression check: the resample CLI binary and the pybind backend's
resample() both call the same computeResample() core, so for identical
input + pose they must produce matching corrected projections."""
import json
import math
import os
import subprocess
import sys

FORWARD_SEARCH_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, FORWARD_SEARCH_DIR)

import h5py
import numpy as np
from make_synth_hdf5 import make_synth_hdf5
from backend import _backend

SYNTH_PATH = "/tmp/forward_search_test_synth.hdf5"
POSE_PATH = "/tmp/forward_search_test_resample_pose.json"
CLI_OUT_PATH = "/tmp/forward_search_test_resample_cli_out.h5"
CLI_BIN = os.path.join(FORWARD_SEARCH_DIR, "builddir", "forward_search_resample")

DOWNSAMPLE = 2


def main():
    make_synth_hdf5(SYNTH_PATH)
    with h5py.File(SYNTH_PATH, "r") as f:
        projs = f["Projection"][()]
        SDD, SOD, pixel_size = f["SDD"][()], f["SOD"][()], f["pixelSize"][()]
        W = int(f["detector_width"][()])
        H = int(f["detector_height"][()])

    # Get a real pose from the backend's own search, so resample runs on
    # the same kind of pose it would in the actual search -> resample flow.
    pose = _backend.search(
        projs, SDD, SOD, pixel_size, W, H,
        xshift=5.0, alpha=2.0, beta=2.0,
        xshift_step=5.0, alpha_step=2.0, beta_step=2.0,
        mode="buffer",
    )

    # Same JSON shape the reference Topic_3_forwardsearching.py writes,
    # which is what the CLI's --pose expects.
    with open(POSE_PATH, "w") as f:
        json.dump({
            "xshift": pose["xshift"], "alpha": pose["alpha"], "beta": pose["beta"],
            "center_point": [pose["center_x"], pose["center_y"]],
        }, f)

    subprocess.run([
        CLI_BIN, "--data", SYNTH_PATH, "--pose", POSE_PATH,
        "--output", CLI_OUT_PATH, "--downsample", str(DOWNSAMPLE),
    ], check=True, cwd=FORWARD_SEARCH_DIR)

    with h5py.File(CLI_OUT_PATH, "r") as f:
        cli_projs = f["Projection"][()]
        cli_SDD, cli_SOD, cli_pixel = f["SDD"][()], f["SOD"][()], f["pixelSize"][()]

    res = _backend.resample(
        projs, SDD, SOD, pixel_size,
        xshift=pose["xshift"], alpha=pose["alpha"], beta=pose["beta"],
        center_x=pose["center_x"], center_y=pose["center_y"],
        downsample=DOWNSAMPLE,
    )

    assert res["projections"].shape == (projs.shape[0], H // DOWNSAMPLE, W // DOWNSAMPLE)
    assert res["projections"].shape == cli_projs.shape
    assert np.array_equal(res["projections"], cli_projs)
    assert math.isclose(res["SDD"], cli_SDD, rel_tol=1e-12)
    assert math.isclose(res["SOD"], cli_SOD, rel_tol=1e-12)
    assert math.isclose(res["pixel_size"], cli_pixel, rel_tol=1e-12)

    print("test_resample_cli_vs_backend: PASS")


if __name__ == "__main__":
    main()
