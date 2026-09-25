"""Runs the pybind11 interface end to end: Python reads the input HDF5,
calls the OpenCL backend (search, then resample with the found pose), and
writes both result HDF5 files itself. Run with:

    cd projection-center-cpp
    python3 run_pybind.py --data /path/to/projs_change.hdf5

Pass --skip-resample to only run the search, and --backend to pick the
implementation: cpp (default, this directory's own OpenCL kernels),
opencl (projection-center-python-opencl/'s kernels, driven from C++) or
cpu (its NumPy CPU backend, ported to C++).

Prints how long each step took, in the same form as the other
implementations (search total, resample total, total search + resample),
plus the one-off cost of importing torch and loading the module.
"""
import time

_T_START = time.perf_counter()

import argparse
import math
import os
import sys

import h5py
from backend import _backend, private_tmp_dir

_IMPORT_MS = (time.perf_counter() - _T_START) * 1000


# Flush every line: the C++ side prints its own progress straight to stderr,
# and without this Python's buffered lines could show up out of order
# relative to it when the output is piped or redirected.
sys.stdout.reconfigure(line_buffering=True)


def _ms(t0):
    return (time.perf_counter() - t0) * 1000


def _timing_line(read_ms, compute_ms, write_ms):
    # Same line, same format, as every other implementation in this project
    # prints (projection-center-cpp's binaries, the projection-center CLI).
    return (f"timing (excluding imports): read {read_ms:.0f} ms | compute {compute_ms:.0f} ms"
            f" | write {write_ms:.0f} ms | I/O {read_ms + write_ms:.0f} ms"
            f" | compute+I/O {read_ms + compute_ms + write_ms:.0f} ms")


def _writable_output(path):
    """Returns path if the file can be written there (creating its folder if
    needed), otherwise the same file name in a private folder under the
    system temp dir. Checked before any work starts, so a read-only project folder
    doesn't crash the run after the search has already taken its time."""
    target = path if os.path.exists(path) else os.path.dirname(os.path.abspath(path))
    while not os.path.exists(target):
        target = os.path.dirname(target)
    if os.access(target, os.W_OK):
        return path
    # A per-user folder only this user can write (see private_tmp_dir()),
    # not a shared name another user could have prepared.
    fallback = os.path.join(private_tmp_dir("projection_center_runs"), os.path.basename(path))
    print(f"{path}: location is not writable, writing to {fallback} instead")
    return fallback


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
    ap.add_argument("--mode", default="buffer", choices=["image", "buffer"], help="cpp backend only")
    ap.add_argument("--backend", default="cpp", choices=["cpp", "opencl", "cpu"])
    ap.add_argument("--resample-output", default="runs/projs_resample.h5",
                    help="Path to the resampled-projections output HDF5")
    ap.add_argument("--downsample", type=int, default=1, help="Detector downsample factor for resample")
    ap.add_argument("--skip-resample", action="store_true", help="Only run the search")
    args = ap.parse_args()

    print(f"import (torch + pybind module load): {_IMPORT_MS:.0f} ms")

    args.output = _writable_output(args.output)
    if not args.skip_resample:
        args.resample_output = _writable_output(args.resample_output)

    t0 = time.perf_counter()
    with h5py.File(args.data, "r") as f:
        projs = f["Projection"][()]
        # Passed straight through to the resampled output unchanged.
        passthrough = {k: f[k][()] for k in ("voxelSize", "Volumen_num_xz", "Volumen_num_y", "num_projs", "Angle")}
        SDD, SOD, pixel_size = f["SDD"][()], f["SOD"][()], f["pixelSize"][()]
    # Detector size from the array's own shape, not the file's
    # detector_width/detector_height scalars: some dataset files
    # (proj_shepplogan512.hdf5) store those two swapped.
    detector_height, detector_width = projs.shape[1:]
    read_ms = _ms(t0)
    print(f"Loaded {projs.shape[0]} projections, {detector_width}x{detector_height} ({read_ms:.0f} ms)")

    t_search = time.perf_counter()
    result = _backend.search(
        projs, SDD, SOD, pixel_size, detector_width, detector_height,
        xshift=args.xshift, alpha=args.alpha, beta=args.beta,
        xshift_step=args.xshift_step, alpha_step=args.alpha_step, beta_step=args.beta_step,
        mode=args.mode, backend=args.backend,
    )
    search_ms = _ms(t_search)

    print(f"MSE:    {result['MSE']}")
    print(f"xshift: {result['xshift'] * 1000:.6g} mm")
    print(f"alpha:  {math.degrees(result['alpha']):.6g} deg")
    print(f"beta:   {math.degrees(result['beta']):.6g} deg")
    if result["kernel_ms"] is not None:
        print(f"search kernel: {result['kernel_ms']} ms")
    print(f"search total: {search_ms:.0f} ms ({args.backend} backend)")

    t_write = time.perf_counter()
    out_dir = os.path.dirname(args.output)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with h5py.File(args.output, "w") as f:
        f["xshift"] = result["xshift"]
        f["alpha"]  = result["alpha"]
        f["beta"]   = result["beta"]
        f["MSE"]    = result["MSE"]
        f["center_point"] = [result["center_x"], result["center_y"]]
    write_ms = _ms(t_write)

    print(f"Wrote {args.output}: {result}")

    if args.skip_resample:
        print(_timing_line(read_ms, search_ms, write_ms))
        print(f"total run: {_ms(_T_START) / 1000:.1f} s (including import and file I/O)")
        return

    # The pose goes straight from search() into resample(), in memory, in
    # the same units search() returned it (meters/radians).
    t_resample = time.perf_counter()
    res = _backend.resample(
        projs, SDD, SOD, pixel_size,
        xshift=result["xshift"], alpha=result["alpha"], beta=result["beta"],
        center_x=result["center_x"], center_y=result["center_y"],
        downsample=args.downsample, backend=args.backend,
    )
    resample_ms = _ms(t_resample)
    print(f"resample total: {resample_ms:.0f} ms ({args.backend} backend)")

    t_write = time.perf_counter()
    out_dir = os.path.dirname(args.resample_output)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    # Same field names the reference Topic_3_resampling.py and the C++
    # resample CLI write, so any of them can read the others' output.
    with h5py.File(args.resample_output, "w") as f:
        for k, v in passthrough.items():
            f[k] = v
        f["SDD"]             = res["SDD"]
        f["SOD"]             = res["SOD"]
        f["pixelSize"]       = res["pixel_size"]
        f["detector_width"]  = float(res["detector_width"])
        f["detector_height"] = float(res["detector_height"])
        f["Projection"]      = res["projections"]
    write_ms += _ms(t_write)

    print(f"Wrote {args.resample_output}: projections {res['projections'].shape}")
    print(f"total (search + resample): {search_ms + resample_ms:.0f} ms")
    print(_timing_line(read_ms, search_ms + resample_ms, write_ms))
    print(f"total run: {_ms(_T_START) / 1000:.1f} s (including import and file I/O)")


if __name__ == "__main__":
    main()
