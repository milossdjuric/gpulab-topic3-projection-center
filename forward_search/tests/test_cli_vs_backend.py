"""Regression check: the CLI binary and the pybind backend both call the
same computeCOR() core, so for identical input+params they must produce
matching results."""
import math
import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import h5py
from make_synth_hdf5 import make_synth_hdf5
from backend import _backend

SYNTH_PATH = "/tmp/forward_search_test_synth.hdf5"
CLI_OUT_PATH = "/tmp/forward_search_test_cli_out.h5"
FORWARD_SEARCH_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CLI_BIN = os.path.join(FORWARD_SEARCH_DIR, "builddir", "forward_search")

ARGS = dict(xshift=5.0, alpha=2.0, beta=2.0,
            xshift_step=5.0, alpha_step=2.0, beta_step=2.0)


def main():
    make_synth_hdf5(SYNTH_PATH)

    subprocess.run([
        CLI_BIN, "--data", SYNTH_PATH,
        "--xshift", str(ARGS["xshift"]), "--alpha", str(ARGS["alpha"]), "--beta", str(ARGS["beta"]),
        "--xshift-step", str(ARGS["xshift_step"]), "--alpha-step", str(ARGS["alpha_step"]),
        "--beta-step", str(ARGS["beta_step"]),
        "--mode", "buffer", "--output", CLI_OUT_PATH,
    ], check=True, cwd=FORWARD_SEARCH_DIR)

    with h5py.File(CLI_OUT_PATH, "r") as f:
        cli_result = {
            "xshift": f["xshift"][()], "alpha": f["alpha"][()], "beta": f["beta"][()],
            "MSE": f["MSE"][()], "center_point": list(f["center_point"][()]),
        }

    with h5py.File(SYNTH_PATH, "r") as f:
        projs = f["Projection"][()]
        SDD, SOD, pixel_size = f["SDD"][()], f["SOD"][()], f["pixelSize"][()]
        W = int(f["detector_width"][()])
        H = int(f["detector_height"][()])

    backend_result = _backend.search(
        projs, SDD, SOD, pixel_size, W, H, mode="buffer", **ARGS,
    )

    assert cli_result["xshift"] == backend_result["xshift"]
    assert cli_result["alpha"] == backend_result["alpha"]
    assert cli_result["beta"] == backend_result["beta"]
    assert math.isclose(cli_result["MSE"], backend_result["MSE"], rel_tol=1e-6)
    assert math.isclose(cli_result["center_point"][0], backend_result["center_x"], rel_tol=1e-6)
    assert math.isclose(cli_result["center_point"][1], backend_result["center_y"], rel_tol=1e-6)

    print("test_cli_vs_backend: PASS")


if __name__ == "__main__":
    main()
