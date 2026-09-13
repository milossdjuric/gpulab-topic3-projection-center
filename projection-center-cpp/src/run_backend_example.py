"""Example of the grading-flow interface: Python reads the input HDF5,
calls the OpenCL backend, and writes the result HDF5 itself. Run with:

    python3 run_backend_example.py --data /path/to/projs_change.hdf5
"""
import argparse
import os

import h5py
from backend import _backend


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="Path to input HDF5")
    ap.add_argument("--output", default="runs/real_cb_pose.h5", help="Path to output HDF5")
    ap.add_argument("--xshift", type=float, default=40.0)
    ap.add_argument("--alpha", type=float, default=10.0)
    ap.add_argument("--beta", type=float, default=10.0)
    ap.add_argument("--xshift-step", type=float, default=1.0)
    ap.add_argument("--alpha-step", type=float, default=1.0)
    ap.add_argument("--beta-step", type=float, default=1.0)
    ap.add_argument("--mode", default="buffer", choices=["image", "buffer"])
    args = ap.parse_args()

    with h5py.File(args.data, "r") as f:
        projs = f["Projection"][()]
        SDD, SOD, pixel_size = f["SDD"][()], f["SOD"][()], f["pixelSize"][()]
        detector_width = int(f["detector_width"][()])
        detector_height = int(f["detector_height"][()])

    result = _backend.search(
        projs, SDD, SOD, pixel_size, detector_width, detector_height,
        xshift=args.xshift, alpha=args.alpha, beta=args.beta,
        xshift_step=args.xshift_step, alpha_step=args.alpha_step, beta_step=args.beta_step,
        mode=args.mode,
    )

    out_dir = os.path.dirname(args.output)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with h5py.File(args.output, "w") as f:
        f["xshift"] = result["xshift"]
        f["alpha"]  = result["alpha"]
        f["beta"]   = result["beta"]
        f["MSE"]    = result["MSE"]
        f["center_point"] = [result["center_x"], result["center_y"]]

    print(f"Wrote {args.output}: {result}")


if __name__ == "__main__":
    main()
