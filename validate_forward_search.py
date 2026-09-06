#!/usr/bin/env python3
"""
Compare GPU forward search output against the Python reference.

Usage:
    # Run Python reference first, then GPU binary, then (--gpu/--ref default
    # to runs/real_cb_pose.h5 / runs/ref_cb_pose.json):
    python3 validate_forward_search.py

    # Or run the Python reference automatically:
    python3 validate_forward_search.py --run-ref \
        --data /lgrp/edu-2026-1-gpulab/projs_change.hdf5

    # Or validate against a bug-fixed reference instead (recommended -- see
    # KNOWN REFERENCE BUG below):
    python3 validate_forward_search.py --run-ref \
        --use-corrected-ref --data /lgrp/edu-2026-1-gpulab/projs_change.hdf5

KNOWN REFERENCE BUG:
    Topic_3_forwardsearching.get_linear_interpolate_MSE declares f_theta and
    f_rtheta outside the per-ray bounds check and never resets them to 0, so
    a ray that falls outside the detector silently reuses the *previous*
    ray's interpolated value instead of contributing 0. On projs_change.hdf5
    this hits ~47% of rays at some grid points and biases the reference's
    MSE landscape unevenly -- confirmed (via a fine 0.1-degree sweep) to
    shift its reported optimum by as much as 1.8 degrees in beta, not merely
    an adjacent grid bin. --use-corrected-ref runs the reference's own
    Compute_COR search in-process with that one function monkey-patched to
    zero-init on out-of-bounds, without ever modifying
    Topic_3_forwardsearching.py on disk, and validates against that
    demonstrably-correct baseline instead of relying on a tolerance heuristic.
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


def load(path):
    if path.endswith(".h5") or path.endswith(".hdf5"):
        with h5py.File(path, "r") as f:
            return {
                "xshift": f["xshift"][()],
                "alpha": f["alpha"][()],
                "beta": f["beta"][()],
                "MSE": f["MSE"][()],
                "center_point": list(f["center_point"][()]),
            }
    with open(path) as f:
        return json.load(f)


def run_reference(data_path, out_path, xshift, alpha, beta,
                  xshift_step, alpha_step, beta_step):
    script = os.path.join(os.path.dirname(__file__), "reference", "Topic_3_forwardsearching.py")
    cmd = [
        sys.executable, script,
        "--data", data_path,
        "--xshift",      str(xshift),
        "--alpha",       str(alpha),
        "--beta",        str(beta),
        "--xshift_step", str(xshift_step),
        "--alpha_step",  str(alpha_step),
        "--beta_step",   str(beta_step),
    ]
    print(f"Running reference: {' '.join(cmd)}")
    t0 = time.time()
    subprocess.run(cmd, check=True)
    elapsed = time.time() - t0
    print(f"Reference finished in {elapsed:.1f}s")
    # Python script always writes to ./real_cb_pose.json — move it
    import shutil
    out_dir = os.path.dirname(out_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    shutil.move("real_cb_pose.json", out_path)


def _corrected_get_linear_interpolate_MSE(N, sino_input, SDD, detector_width,
                                          detector_height, pixel_size, alpha,
                                          beta, theta_0, nearest_theta):
    """Drop-in replacement for Topic_3_forwardsearching.get_linear_interpolate_MSE
    with f_theta/f_rtheta zero-initialized per ray, instead of carrying
    forward the previous ray's value on out-of-bounds. See KNOWN REFERENCE
    BUG at the top of this file."""
    pixel_MSE = np.zeros((N,), dtype=np.float64)
    for idx in range(nearest_theta.shape[0]):
        theta = nearest_theta[idx] + theta_0
        reflect_theta = -nearest_theta[idx] + theta_0
        tan_t = math.tan(theta)
        tan_rt = math.tan(reflect_theta)

        sin_a = math.sin(alpha)
        cos_a = math.cos(alpha)
        sin_b = math.sin(beta)
        cos_b = math.cos(beta)

        temp = SDD / (tan_t * sin_a * sin_b + cos_b)
        rtemp = SDD / (tan_rt * sin_a * sin_b + cos_b)

        y = temp * (-tan_t * sin_a * cos_b + sin_b)
        ry = rtemp * (-tan_rt * sin_a * cos_b + sin_b)
        x = -temp * tan_t * cos_a
        rx = -rtemp * tan_rt * cos_a

        y = (detector_height - 1) / 2 - y / pixel_size
        ry = (detector_height - 1) / 2 - ry / pixel_size
        x = x / pixel_size + (detector_width - 1) / 2
        rx = rx / pixel_size + (detector_width - 1) / 2

        f_theta = 0.0
        f_rtheta = 0.0
        if 0 <= x < detector_width - 1 and 0 <= y < detector_height - 1:
            idx_x = int(x)
            idx_y = int(y)
            dx = x - idx_x
            dy = y - idx_y
            f_theta = (1 - dx) * ((1 - dy) * sino_input[idx_y][idx_x] + dy * sino_input[idx_y + 1][idx_x]) \
                    +       dx  * ((1 - dy) * sino_input[idx_y][idx_x + 1] + dy * sino_input[idx_y + 1][idx_x + 1])
        if 0 <= rx < detector_width - 1 and 0 <= ry < detector_height - 1:
            idx_x = int(rx)
            idx_y = int(ry)
            dx = rx - idx_x
            dy = ry - idx_y
            f_rtheta = (1 - dx) * ((1 - dy) * sino_input[idx_y][idx_x] + dy * sino_input[idx_y + 1][idx_x]) \
                     +       dx  * ((1 - dy) * sino_input[idx_y][idx_x + 1] + dy * sino_input[idx_y + 1][idx_x + 1])

        pixel_MSE[idx] = (f_theta - f_rtheta) ** 2
    return pixel_MSE


def run_corrected_reference(data_path, out_path, xshift, alpha, beta,
                            xshift_step, alpha_step, beta_step):
    """Runs Topic_3_forwardsearching.Compute_COR in-process with
    get_linear_interpolate_MSE monkey-patched to the zero-init-on-oob
    version above. The file on disk is never written to."""
    reference_dir = os.path.join(os.path.dirname(__file__), "reference")
    if reference_dir not in sys.path:
        sys.path.insert(0, reference_dir)
    import Topic_3_forwardsearching as ref_mod

    orig_fn = ref_mod.get_linear_interpolate_MSE
    ref_mod.get_linear_interpolate_MSE = _corrected_get_linear_interpolate_MSE
    try:
        with h5py.File(data_path, "r") as f:
            cb_para = {
                "num_projs": int(f["num_projs"][()]),
                "SDD": f["SDD"][()],
                "SOD": f["SOD"][()],
                "pixel_size": f["pixelSize"][()],
                "voxelSize": f["voxelSize"][()],
                "Volumen_num_xz": int(f["Volumen_num_xz"][()]),
                "Volumen_num_y": int(f["Volumen_num_y"][()]),
                "detector_width": int(f["detector_width"][()]),
                "detector_height": int(f["detector_height"][()]),
                "angles": f["Angle"][()],
            }
            projs = f["Projection"][()][:]

        class _Args:
            pass
        a = _Args()
        a.xshift, a.alpha, a.beta = xshift, alpha, beta
        a.xshift_step, a.alpha_step, a.beta_step = xshift_step, alpha_step, beta_step

        print(f"Running corrected reference (in-process, zero-init on out-of-bounds)")
        t0 = time.time()
        real_cb_pose = ref_mod.Compute_COR(cb_para, projs, a)
        elapsed = time.time() - t0
        print(f"Corrected reference finished in {elapsed:.1f}s")

        real_cb_pose["_corrected_reference"] = True
        out_dir = os.path.dirname(out_path)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(real_cb_pose, f)
    finally:
        ref_mod.get_linear_interpolate_MSE = orig_fn


def fmt_angle(rad):
    return f"{math.degrees(rad):.6f} deg"

def fmt_mm(m):
    return f"{m * 1000:.6f} mm"


def compare(gpu, ref, xshift_step_m=None, alpha_step_rad=None, beta_step_rad=None,
           ref_is_corrected=False):
    # Grid-step tolerance: absorbs float32-vs-float64 / native_trig noise, which
    # can occasionally tip the argmin by one bin even when both implementations
    # agree. It does NOT reliably absorb the reference's out-of-bounds bug (see
    # KNOWN REFERENCE BUG at the top of this file) -- that bug's effect on the
    # argmin is landscape-dependent and confirmed to reach 1.8 degrees in beta
    # on this dataset, well beyond one grid step. Use --use-corrected-ref for a
    # rigorous check that isn't relying on this tolerance to paper over it.
    passed = True
    adjacent_bin_notes = []
    rows = []

    # --- MSE ---
    mse_rel = abs(gpu["MSE"] - ref["MSE"]) / max(abs(ref["MSE"]), 1e-12)
    mse_ok = mse_rel < 0.01  # 1% tolerance (float32 vs float64 + native_trig)
    rows.append(("MSE (GPU)",    f"{gpu['MSE']:.8e}"))
    rows.append(("MSE (ref)",    f"{ref['MSE']:.8e}"))
    rows.append(("MSE rel err",  f"{mse_rel*100:.4f}%  {'OK' if mse_ok else 'FAIL (>1%)'}"))

    rows.append(("", ""))

    # --- Best combo (xshift, alpha, beta) ---
    steps = {"xshift": xshift_step_m, "alpha": alpha_step_rad, "beta": beta_step_rad}
    combo_mismatch = False
    for key, fmt in [("xshift", fmt_mm), ("alpha", fmt_angle), ("beta", fmt_angle)]:
        gv = gpu[key]
        rv = ref[key]
        diff = abs(gv - rv)
        step = steps[key]
        # "Exact" absorbs float32-vs-float64 rounding noise (diff << one grid
        # step); "adjacent" is reserved for an actual one-bin argmin
        # disagreement, which is orders of magnitude larger than that noise.
        noise_floor = max(1e-9, step * 1e-4) if step is not None else 1e-9
        exact = diff < noise_floor
        adjacent = (not exact) and step is not None and diff < step * 1.01
        rows.append((f"{key} (GPU)", fmt(gv)))
        rows.append((f"{key} (ref)", fmt(rv)))
        if exact:
            status = "OK"
        elif adjacent:
            status = "OK (adjacent grid bin)"
            adjacent_bin_notes.append(key)
        else:
            status = f"MISMATCH  diff={diff:.2e}"
            combo_mismatch = True
        rows.append((f"{key} match", status))
        rows.append(("", ""))

    if combo_mismatch:
        passed = False
    elif not mse_ok and not adjacent_bin_notes:
        # MSE off by >1% with no adjacent-bin explanation on any param -- real mismatch.
        passed = False

    # --- Center point ---
    # center_point is derived from (xshift, alpha, beta), so it necessarily
    # diverges when the argmin lands on an adjacent bin -- not scored
    # independently, just reported for context.
    cx_gpu, cy_gpu = gpu["center_point"]
    cx_ref, cy_ref = ref["center_point"]
    cx_ok = abs(cx_gpu - cx_ref) < 1e-6
    cy_ok = abs(cy_gpu - cy_ref) < 1e-6
    rows.append(("center_x (GPU)", f"{cx_gpu:.8f}"))
    rows.append(("center_x (ref)", f"{cx_ref:.8f}"))
    rows.append(("center_x match", "OK" if cx_ok else f"diff={abs(cx_gpu-cx_ref):.2e}{'  (expected: adjacent bin)' if adjacent_bin_notes else ''}"))
    rows.append(("", ""))
    rows.append(("center_y (GPU)", f"{cy_gpu:.8f}"))
    rows.append(("center_y (ref)", f"{cy_ref:.8f}"))
    rows.append(("center_y match", "OK" if cy_ok else f"diff={abs(cy_gpu-cy_ref):.2e}{'  (expected: adjacent bin)' if adjacent_bin_notes else ''}"))

    # Print table
    col = max(len(r[0]) for r in rows) + 2
    print()
    print("=" * 60)
    print("  VALIDATION RESULTS")
    print("=" * 60)
    for label, value in rows:
        if label == "":
            print()
        else:
            print(f"  {label:<{col}} {value}")
    print("=" * 60)
    print(f"  OVERALL: {'PASS' if passed else 'FAIL'}")
    if ref_is_corrected:
        print()
        print(f"  Reference run with --use-corrected-ref: get_linear_interpolate_MSE's")
        print(f"  out-of-bounds handling was patched in-process (zero-init instead of")
        print(f"  stale carry-over) before running the search. This is the rigorous")
        print(f"  check -- a mismatch here is a real GPU-side issue, not the known")
        print(f"  reference bug.")
    elif adjacent_bin_notes:
        print()
        print(f"  NOTE: {', '.join(adjacent_bin_notes)} landed on an adjacent grid bin vs the")
        print(f"  unmodified reference. This MAY be the reference's known out-of-bounds bug")
        print(f"  (see KNOWN REFERENCE BUG at the top of this file) tipping its argmin by one")
        print(f"  step, or it may be genuine float32-vs-float64 noise -- a coarse grid can't")
        print(f"  tell these apart. Re-run with --use-corrected-ref for a rigorous check that")
        print(f"  doesn't depend on this tolerance.")
    print("=" * 60)
    print()
    return passed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu",  default="runs/real_cb_pose.h5",  help="GPU output HDF5")
    ap.add_argument("--ref",  default="runs/ref_cb_pose.json",   help="Reference JSON")
    ap.add_argument("--run-ref", action="store_true",        help="Run Python reference first")
    ap.add_argument("--use-corrected-ref", action="store_true",
                    help="With --run-ref, run the reference in-process with its "
                         "out-of-bounds bug patched (zero-init instead of stale "
                         "carry-over) instead of the unmodified script as a subprocess. "
                         "Topic_3_forwardsearching.py on disk is never modified.")
    ap.add_argument("--data", default="/lgrp/edu-2026-1-gpulab/projs_change.hdf5")
    ap.add_argument("--xshift",      type=float, default=40.0)
    ap.add_argument("--alpha",       type=float, default=10.0)
    ap.add_argument("--beta",        type=float, default=10.0)
    ap.add_argument("--xshift-step", type=float, default=1.0)
    ap.add_argument("--alpha-step",  type=float, default=1.0)
    ap.add_argument("--beta-step",   type=float, default=1.0)
    args = ap.parse_args()

    if args.run_ref:
        if args.use_corrected_ref:
            run_corrected_reference(args.data, args.ref,
                                    args.xshift, args.alpha, args.beta,
                                    args.xshift_step, args.alpha_step, args.beta_step)
        else:
            run_reference(args.data, args.ref,
                          args.xshift, args.alpha, args.beta,
                          args.xshift_step, args.alpha_step, args.beta_step)

    gpu = load(args.gpu)
    ref = load(args.ref)
    # A ref JSON produced by run_corrected_reference is self-tagged, so a
    # compare-only invocation (no --run-ref) still gets the right label.
    ref_is_corrected = args.use_corrected_ref or bool(ref.get("_corrected_reference", False))

    ok = compare(
        gpu, ref,
        xshift_step_m=args.xshift_step / 1000.0,
        alpha_step_rad=math.radians(args.alpha_step),
        beta_step_rad=math.radians(args.beta_step),
        ref_is_corrected=ref_is_corrected,
    )
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
