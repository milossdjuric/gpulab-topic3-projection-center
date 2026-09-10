"""Regression test: forward_search_pipeline (single process, search +
resample in one run, pose handed off in memory) must produce byte-identical
output to running forward_search then forward_search_resample separately
(two processes, pose round-tripped through a JSON file)."""
import json
import os
import subprocess
import sys

FORWARD_SEARCH_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(FORWARD_SEARCH_DIR, "src"))

import h5py
import numpy as np
from make_synth_hdf5 import make_synth_hdf5

SYNTH_PATH = "/tmp/forward_search_test_pipeline_synth.hdf5"
SEARCH_BIN = os.path.join(FORWARD_SEARCH_DIR, "builddir", "forward_search")
RESAMPLE_BIN = os.path.join(FORWARD_SEARCH_DIR, "builddir", "forward_search_resample")
PIPELINE_BIN = os.path.join(FORWARD_SEARCH_DIR, "builddir", "forward_search_pipeline")

OLD_POSE_H5 = "/tmp/pipeline_test_old_pose.h5"
OLD_POSE_JSON = "/tmp/pipeline_test_old_pose.json"
OLD_RESAMPLED = "/tmp/pipeline_test_old_resampled.hdf5"
NEW_POSE = "/tmp/pipeline_test_new_pose.h5"
NEW_RESAMPLED = "/tmp/pipeline_test_new_resampled.hdf5"

ARGS = ["--xshift", "5", "--alpha", "5", "--beta", "5",
        "--xshift-step", "5", "--alpha-step", "5", "--beta-step", "5"]


def main():
    make_synth_hdf5(SYNTH_PATH)

    subprocess.run([SEARCH_BIN, "--data", SYNTH_PATH, "--mode", "buffer",
                     "--output", OLD_POSE_H5, *ARGS], check=True, cwd=FORWARD_SEARCH_DIR)
    with h5py.File(OLD_POSE_H5, "r") as f:
        json.dump({
            "center_point": [float(f["center_point"][0]), float(f["center_point"][1])],
            "xshift": float(f["xshift"][()]), "alpha": float(f["alpha"][()]), "beta": float(f["beta"][()]),
        }, open(OLD_POSE_JSON, "w"))
    subprocess.run([RESAMPLE_BIN, "--data", SYNTH_PATH, "--pose", OLD_POSE_JSON,
                     "--output", OLD_RESAMPLED], check=True, cwd=FORWARD_SEARCH_DIR)

    subprocess.run([PIPELINE_BIN, "--data", SYNTH_PATH, "--mode", "buffer",
                     "--output-pose", NEW_POSE, "--output-data", NEW_RESAMPLED, *ARGS],
                    check=True, cwd=FORWARD_SEARCH_DIR)

    with h5py.File(OLD_POSE_H5, "r") as f:
        old_pose = {k: float(f[k][()]) for k in ("xshift", "alpha", "beta", "MSE")}
    with h5py.File(NEW_POSE, "r") as f:
        new_pose = {k: float(f[k][()]) for k in ("xshift", "alpha", "beta", "MSE")}
    assert old_pose == new_pose, f"pose mismatch: {old_pose} vs {new_pose}"

    with h5py.File(OLD_RESAMPLED, "r") as f:
        old_projs = f["Projection"][()]
    with h5py.File(NEW_RESAMPLED, "r") as f:
        new_projs = f["Projection"][()]
    assert np.array_equal(old_projs, new_projs), "resampled output differs between two-process and single-process runs"

    print("test_pipeline_single_process: PASS")


if __name__ == "__main__":
    main()
