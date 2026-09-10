"""Regression test for a real bug: some dataset files store detector_width/
detector_height scalars that don't match Projection's actual array shape
(confirmed on data/proj_shepplogan512.hdf5, width/height swapped relative
to the real array). loadHDF5() in both main.cpp and resample_main.cpp must
derive detector_width/detector_height from Projection's real dimensions,
not trust the file's stored scalars -- so two files with identical pixel
data but different (even wrong) stored width/height scalars must produce
identical results."""
import os
import subprocess
import sys

FORWARD_SEARCH_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import h5py
import json
import numpy as np

SEARCH_BIN = os.path.join(FORWARD_SEARCH_DIR, "builddir", "forward_search")
RESAMPLE_BIN = os.path.join(FORWARD_SEARCH_DIR, "builddir", "forward_search_resample")

CORRECT_PATH = "/tmp/forward_search_test_dims_correct.hdf5"
SWAPPED_PATH = "/tmp/forward_search_test_dims_swapped.hdf5"
POSE_FROM_CORRECT = "/tmp/forward_search_test_dims_pose_correct.h5"
POSE_FROM_SWAPPED = "/tmp/forward_search_test_dims_pose_swapped.h5"
POSE_JSON = "/tmp/forward_search_test_dims_pose.json"
RESAMPLED_CORRECT = "/tmp/forward_search_test_dims_resampled_correct.hdf5"
RESAMPLED_SWAPPED = "/tmp/forward_search_test_dims_resampled_swapped.hdf5"

# Non-square on purpose: H != W is what makes a width/height swap
# distinguishable at all.
NUM_PROJS, H, W = 4, 48, 32

SEARCH_ARGS = ["--xshift", "5", "--alpha", "5", "--beta", "5",
               "--xshift-step", "5", "--alpha-step", "5", "--beta-step", "5"]


def make_hdf5(path, store_width, store_height, projs):
    with h5py.File(path, "w") as f:
        f["voxelSize"] = 1.0
        f["Volumen_num_xz"] = float(W)
        f["Volumen_num_y"] = float(H)
        f["SDD"] = 1000.0
        f["SOD"] = 700.0
        f["pixelSize"] = 0.5
        f["num_projs"] = float(NUM_PROJS)
        f["detector_width"] = float(store_width)
        f["detector_height"] = float(store_height)
        f["Angle"] = np.linspace(0, 2 * np.pi, NUM_PROJS)
        f["Projection"] = projs


def read_pose(path):
    with h5py.File(path, "r") as f:
        return {
            "xshift": float(f["xshift"][()]), "alpha": float(f["alpha"][()]),
            "beta": float(f["beta"][()]), "MSE": float(f["MSE"][()]),
            "center_point": list(f["center_point"][()]),
        }


def main():
    rng = np.random.default_rng(42)
    projs = rng.random((NUM_PROJS, H, W)).astype(np.float32)

    # Identical pixel data. One file's scalars match the real array
    # (width=W, height=H). The other's are deliberately wrong (swapped),
    # mirroring the real proj_shepplogan512.hdf5 bug.
    make_hdf5(CORRECT_PATH, store_width=W, store_height=H, projs=projs)
    make_hdf5(SWAPPED_PATH, store_width=H, store_height=W, projs=projs)

    subprocess.run([SEARCH_BIN, "--data", CORRECT_PATH, "--mode", "buffer",
                     "--output", POSE_FROM_CORRECT, *SEARCH_ARGS],
                    check=True, cwd=FORWARD_SEARCH_DIR)
    subprocess.run([SEARCH_BIN, "--data", SWAPPED_PATH, "--mode", "buffer",
                     "--output", POSE_FROM_SWAPPED, *SEARCH_ARGS],
                    check=True, cwd=FORWARD_SEARCH_DIR)

    pose_correct = read_pose(POSE_FROM_CORRECT)
    pose_swapped = read_pose(POSE_FROM_SWAPPED)

    assert pose_correct == pose_swapped, (
        f"search result depends on the (possibly wrong) stored detector_width/"
        f"detector_height scalars instead of Projection's real shape:\n"
        f"  from correctly-labeled file: {pose_correct}\n"
        f"  from swapped-labeled file:   {pose_swapped}"
    )

    # forward_search_resample's --pose reads JSON (readPoseJSON in
    # resample_main.cpp), not the HDF5 that --output above just wrote, so
    # write out the matching JSON schema by hand here.
    with open(POSE_JSON, "w") as f:
        json.dump({
            "center_point": pose_correct["center_point"],
            "xshift": pose_correct["xshift"],
            "alpha": pose_correct["alpha"],
            "beta": pose_correct["beta"],
        }, f)

    subprocess.run([RESAMPLE_BIN, "--data", CORRECT_PATH, "--pose", POSE_JSON,
                     "--output", RESAMPLED_CORRECT],
                    check=True, cwd=FORWARD_SEARCH_DIR)
    subprocess.run([RESAMPLE_BIN, "--data", SWAPPED_PATH, "--pose", POSE_JSON,
                     "--output", RESAMPLED_SWAPPED],
                    check=True, cwd=FORWARD_SEARCH_DIR)

    with h5py.File(RESAMPLED_CORRECT, "r") as f:
        resampled_correct = f["Projection"][()]
    with h5py.File(RESAMPLED_SWAPPED, "r") as f:
        resampled_swapped = f["Projection"][()]

    assert np.array_equal(resampled_correct, resampled_swapped), (
        "resample output depends on the stored detector_width/detector_height "
        "scalars instead of Projection's real shape"
    )

    print("test_detector_dims_from_shape: PASS")


if __name__ == "__main__":
    main()
