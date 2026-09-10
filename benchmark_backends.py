#!/usr/bin/env python3
"""Run the untouched reference and all three projection-center backends
against the same dataset, one after another, and print a single timing/
result comparison table.

Runs (all default search range, no flags overridden, so results are
directly comparable):
  1. reference/Topic_3_forwardsearching.py -- the RAW, unmodified reference,
     search. This is "the original code," known bug included.
  2. reference/Topic_3_resampling.py's resample, using the reference's own
     get_cb_para()/get_rotation_matrix()/get_real_projection() functions
     called directly, in-process -- Topic_3_resampling.py's own resampling()
     wrapper writes to a hardcoded lab-network path and can't complete off
     that network (see RUNNING.md Limitations), so this calls its
     computation functions directly and writes the result under this
     script's own output directory instead. Every line of actual pixel
     computation is still the reference's own unmodified code; only the
     hardcoded write path is bypassed. The .py file on disk is never
     touched, same as validate_forward_search.py's --fix-ref.
  3. projection-center pipeline --backend opencl  (search + resample)
  4. projection-center pipeline --backend cpu     (search + resample)
  5. projection-center pipeline --backend cpp     (search + resample)

WARNING: the reference resample step is extremely slow (nested pure-Python
pixel loop, no vectorization) -- expect on the order of an hour on the real
dataset. Use --skip-reference-resample to keep the (much faster, ~10-27 min)
reference search comparison without paying for this. Use --skip-reference
to skip the reference entirely.

Usage:
    python3 benchmark_backends.py --data data/projs_change.hdf5
    python3 benchmark_backends.py --data data/projs_change.hdf5 --skip-reference-resample
    python3 benchmark_backends.py --data data/projs_change.hdf5 --skip-reference
"""
import argparse
import json
import math
import os
import subprocess
import sys
import time

import h5py
import numpy as np

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))


def run_timed(cmd, cwd=None):
    t0 = time.time()
    result = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)
    elapsed = time.time() - t0
    return elapsed, result


def load_cb_para(data_path):
    with h5py.File(data_path, "r") as f:
        num_projs, height, width = f["Projection"].shape
        return {
            "num_projs": num_projs,
            "SDD": f["SDD"][()],
            "SOD": f["SOD"][()],
            "pixel_size": f["pixelSize"][()],
            "voxelSize": f["voxelSize"][()],
            "Volumen_num_xz": int(f["Volumen_num_xz"][()]),
            "Volumen_num_y": int(f["Volumen_num_y"][()]),
            # Derived from Projection's real shape, not trusted from the
            # file's detector_width/detector_height scalars: some dataset
            # files (data/proj_shepplogan512.hdf5) store those two swapped
            # relative to the actual array.
            "detector_width": width,
            "detector_height": height,
            "angles": f["Angle"][()],
        }


def run_reference_search(data_path, out_dir, xshift=40.0, alpha=10.0, beta=10.0,
                          xshift_step=1.0, alpha_step=1.0, beta_step=1.0):
    # Runs reference/Topic_3_forwardsearching.py's own Compute_COR in-process,
    # with its two known bugs (f_theta/f_rtheta not zero-initialized on an
    # out-of-bounds ray, detector_width/detector_height trusted from file
    # metadata instead of Projection's real shape) corrected at call time --
    # the same approach validate_forward_search.py's --fix-ref already uses.
    # reference/Topic_3_forwardsearching.py itself is never modified on disk.
    print("=== reference (zero-init + detector-dims fixed, run in-process) -- search ===")
    from validate_forward_search import run_corrected_reference

    os.makedirs(out_dir, exist_ok=True)
    pose_path = os.path.join(out_dir, "pose.json")
    t0 = time.time()
    run_corrected_reference(data_path, pose_path, xshift, alpha, beta,
                             xshift_step, alpha_step, beta_step)
    elapsed = time.time() - t0
    with open(pose_path) as f:
        pose = json.load(f)
    print(f"  search done in {elapsed:.1f}s")
    return elapsed, pose


def run_reference_resample(data_path, cb_para, pose, out_dir):
    print("=== reference (raw, unmodified Topic_3_resampling.py functions) -- resample ===")
    print("  (this is the slow part -- nested pure-Python pixel loop, no vectorization)")
    reference_dir = os.path.join(REPO_ROOT, "reference")
    if reference_dir not in sys.path:
        sys.path.insert(0, reference_dir)
    import Topic_3_resampling as ref_mod

    with h5py.File(data_path, "r") as f:
        projs = f["Projection"][()]

    t0 = time.time()
    real_cb_para = ref_mod.get_cb_para(cb_para, pose, downsample_factor=1)
    rotation_matrix = ref_mod.get_rotation_matrix(cb_para, pose)
    real_projs = np.zeros_like(projs)
    for i_proj in range(projs.shape[0]):
        real_projs[i_proj] = ref_mod.get_real_projection(cb_para, real_cb_para, rotation_matrix, projs[i_proj])
    elapsed = time.time() - t0

    out_path = os.path.join(out_dir, "resampled.hdf5")
    with h5py.File(out_path, "w") as file:
        file.create_dataset("pixelSize", dtype=np.float64, data=real_cb_para["pixel_size"])
        file.create_dataset("SDD", dtype=np.float64, data=real_cb_para["SDD"])
        file.create_dataset("SOD", dtype=np.float64, data=real_cb_para["SOD"])
        file.create_dataset("voxelSize", dtype=np.float64, data=cb_para["voxelSize"])
        file.create_dataset("Volumen_num_xz", dtype=np.float64, data=cb_para["Volumen_num_xz"])
        file.create_dataset("Volumen_num_y", dtype=np.float64, data=cb_para["Volumen_num_y"])
        file.create_dataset("num_projs", dtype=np.float64, data=real_cb_para["num_projs"])
        file.create_dataset("detector_width", dtype=np.float64, data=real_cb_para["detector_width"])
        file.create_dataset("detector_height", dtype=np.float64, data=real_cb_para["detector_height"])
        file.create_dataset("Angle", dtype=np.float64, data=cb_para["angles"])
        projection = file.create_dataset(
            "Projection", dtype=np.float32,
            shape=(int(real_cb_para["num_projs"]), int(real_cb_para["detector_height"]), int(real_cb_para["detector_width"])),
        )
        projection[:, :, :] = real_projs

    print(f"  resample done in {elapsed:.1f}s")
    return elapsed, out_path


def run_reference(data_path, out_dir, skip_resample):
    search_elapsed, pose = run_reference_search(data_path, out_dir)
    if skip_resample:
        return search_elapsed, pose, False

    cb_para = load_cb_para(data_path)
    resample_elapsed, _ = run_reference_resample(data_path, cb_para, pose, out_dir)
    return search_elapsed + resample_elapsed, pose, True


def run_backend_pipeline(backend, data_path, out_dir):
    print(f"=== projection-center pipeline --backend {backend} -- search + resample ===")
    pose_path = os.path.join(out_dir, "pose.json")
    data_out_path = os.path.join(out_dir, "resampled.hdf5")
    cmd = [
        "projection-center", "pipeline", "--backend", backend,
        "--data", data_path,
        "--output-pose", pose_path, "--output-data", data_out_path,
    ]
    elapsed, result = run_timed(cmd, cwd=REPO_ROOT)
    if result.returncode != 0:
        print(result.stderr[-3000:])
        raise RuntimeError(f"{backend} pipeline failed")
    with open(pose_path) as f:
        pose = json.load(f)
    print(f"  done in {elapsed:.1f}s")
    return elapsed, elapsed, pose


def fmt_pose(pose):
    xshift_mm = pose["xshift"] * 1000
    alpha_deg = math.degrees(pose["alpha"])
    beta_deg = math.degrees(pose["beta"])
    mse = pose.get("MSE", pose.get("mse"))
    kernel_ms = pose.get("kernel_ms")
    kernel_str = f"  kernel={kernel_ms:.2f}ms" if kernel_ms is not None else ""
    return f"xshift={xshift_mm:+.3f}mm  alpha={alpha_deg:+.3f}deg  beta={beta_deg:+.3f}deg  MSE={mse:.6e}{kernel_str}"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default="data/projs_change.hdf5")
    ap.add_argument("--output-dir", default="runs/benchmark_all")
    ap.add_argument("--skip-reference", action="store_true",
                     help="Skip the raw reference entirely (search + resample).")
    ap.add_argument("--skip-reference-resample", action="store_true",
                     help="Run the reference search (~10-27 min) but skip its very slow "
                          "resample step (on the order of an hour on the real dataset).")
    args = ap.parse_args()

    data_path = os.path.abspath(args.data)
    out_root = os.path.join(REPO_ROOT, args.output_dir)

    rows = []  # (label, elapsed, pose, had_resample)

    if not args.skip_reference:
        d = os.path.join(out_root, "reference")
        os.makedirs(d, exist_ok=True)
        try:
            elapsed, pose, had_resample = run_reference(data_path, d, args.skip_reference_resample)
            label = "reference (search + resample)" if had_resample else "reference (search only)"
            rows.append((label, elapsed, pose))
        except RuntimeError as e:
            print(f"  reference FAILED: {e}")
            print("  This can be the known out-of-bounds bug crashing outright (not just")
            print("  silently misbehaving) on a small/edge-of-range dataset -- see")
            print("  RUNNING.md Limitations. Continuing with the other backends.")
            rows.append(("reference (FAILED, see above)", None, None))

    for backend in ("opencl", "cpu", "cpp", "hybrid"):
        d = os.path.join(out_root, backend)
        os.makedirs(d, exist_ok=True)
        elapsed, _, pose = run_backend_pipeline(backend, data_path, d)
        rows.append((f"{backend} (search + resample)", elapsed, pose))

    print()
    print("=" * 78)
    print(f"  {'Run':<32} {'Time':>10}   Pose")
    print("=" * 78)
    for label, elapsed, pose in rows:
        if elapsed is None:
            print(f"  {label:<32} {'--':>7}          (no result -- see error above)")
            continue
        mins = elapsed / 60
        print(f"  {label:<32} {elapsed:>7.1f}s ({mins:>4.1f}m)   {fmt_pose(pose)}")
    print("=" * 78)
    if any(label.endswith("(search only)") for label, _, _ in rows):
        print("  Note: the reference row is search-only; the backend rows are")
        print("  search+resample. Not an apples-to-apples speedup ratio -- see")
        print("  each row's own label.")
        print("=" * 78)
    print()
    print(f"Full pose/output files saved under {os.path.relpath(out_root, REPO_ROOT)}/")


if __name__ == "__main__":
    main()
