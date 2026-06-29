#!/usr/bin/env python3
"""
Compare GPU forward search output against the Python reference.

Usage:
    # Run Python reference first, then GPU binary, then:
    python3 validate_forward_search.py --gpu real_cb_pose.json --ref ref_cb_pose.json

    # Or run the Python reference automatically:
    python3 validate_forward_search.py --gpu real_cb_pose.json --run-ref \
        --data /lgrp/edu-2026-1-gpulab/projs_change.hdf5
"""

import argparse
import json
import math
import subprocess
import sys
import time


def load(path):
    with open(path) as f:
        return json.load(f)


def run_reference(data_path, out_path, xshift, alpha, beta,
                  xshift_step, alpha_step, beta_step):
    import os
    script = os.path.join(os.path.dirname(__file__), "Topic_3_forwardsearching.py")
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
    shutil.move("real_cb_pose.json", out_path)


def fmt_angle(rad):
    return f"{math.degrees(rad):.6f} deg"

def fmt_mm(m):
    return f"{m * 1000:.6f} mm"


def compare(gpu, ref):
    passed = True
    rows = []

    # --- MSE ---
    mse_rel = abs(gpu["MSE"] - ref["MSE"]) / max(abs(ref["MSE"]), 1e-12)
    mse_ok = mse_rel < 0.01  # 1% tolerance (float32 vs float64 + native_trig)
    rows.append(("MSE (GPU)",    f"{gpu['MSE']:.8e}"))
    rows.append(("MSE (ref)",    f"{ref['MSE']:.8e}"))
    rows.append(("MSE rel err",  f"{mse_rel*100:.4f}%  {'OK' if mse_ok else 'FAIL (>1%)'}"))
    if not mse_ok:
        passed = False

    rows.append(("", ""))

    # --- Best combo (xshift, alpha, beta) ---
    for key, fmt in [("xshift", fmt_mm), ("alpha", fmt_angle), ("beta", fmt_angle)]:
        gv = gpu[key]
        rv = ref[key]
        match = abs(gv - rv) < 1e-9
        rows.append((f"{key} (GPU)", fmt(gv)))
        rows.append((f"{key} (ref)", fmt(rv)))
        rows.append((f"{key} match", "OK" if match else f"MISMATCH  diff={abs(gv-rv):.2e}"))
        if not match:
            passed = False
        rows.append(("", ""))

    # --- Center point ---
    cx_gpu, cy_gpu = gpu["center_point"]
    cx_ref, cy_ref = ref["center_point"]
    cx_ok = abs(cx_gpu - cx_ref) < 1e-6
    cy_ok = abs(cy_gpu - cy_ref) < 1e-6
    rows.append(("center_x (GPU)", f"{cx_gpu:.8f}"))
    rows.append(("center_x (ref)", f"{cx_ref:.8f}"))
    rows.append(("center_x match", "OK" if cx_ok else f"MISMATCH  diff={abs(cx_gpu-cx_ref):.2e}"))
    rows.append(("", ""))
    rows.append(("center_y (GPU)", f"{cy_gpu:.8f}"))
    rows.append(("center_y (ref)", f"{cy_ref:.8f}"))
    rows.append(("center_y match", "OK" if cy_ok else f"MISMATCH  diff={abs(cy_gpu-cy_ref):.2e}"))

    if not cx_ok or not cy_ok:
        passed = False

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
    print("=" * 60)
    print()
    return passed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu",  default="real_cb_pose.json",  help="GPU output JSON")
    ap.add_argument("--ref",  default="ref_cb_pose.json",   help="Reference JSON")
    ap.add_argument("--run-ref", action="store_true",        help="Run Python reference first")
    ap.add_argument("--data", default="/lgrp/edu-2026-1-gpulab/projs_change.hdf5")
    ap.add_argument("--xshift",      type=float, default=40.0)
    ap.add_argument("--alpha",       type=float, default=10.0)
    ap.add_argument("--beta",        type=float, default=10.0)
    ap.add_argument("--xshift-step", type=float, default=1.0)
    ap.add_argument("--alpha-step",  type=float, default=1.0)
    ap.add_argument("--beta-step",   type=float, default=1.0)
    args = ap.parse_args()

    if args.run_ref:
        run_reference(args.data, args.ref,
                      args.xshift, args.alpha, args.beta,
                      args.xshift_step, args.alpha_step, args.beta_step)

    gpu = load(args.gpu)
    ref = load(args.ref)

    ok = compare(gpu, ref)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
