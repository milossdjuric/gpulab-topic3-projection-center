#!/usr/bin/env python3
"""Run the untouched reference (search, then resample) as a single pipeline
command, with the same --data/--output-pose/--output-data flags as
`projection-center pipeline`, so it can sit next to the opencl/cpp
projection-center launch configs as a uniform "full pipeline" entry point.

Topic_3_resampling.py's own resampling() wrapper writes to a hardcoded
lab-network path and can't complete off that network, so this calls the
reference's own unmodified search/resample
functions directly, in-process, via benchmark_backends.run_reference() --
the same technique validate_forward_search.py's --fix-ref already uses.
Every line of actual computation is still the reference's own unmodified
code; only the hardcoded write path is bypassed. The reference .py files on
disk are never touched.
"""
import argparse
import os
import shutil
import sys
import tempfile

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from benchmark_backends import run_reference


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True)
    ap.add_argument("--output-pose", required=True)
    ap.add_argument("--output-data", required=True)
    args = ap.parse_args()

    output_pose = os.path.abspath(args.output_pose)
    output_data = os.path.abspath(args.output_data)
    os.makedirs(os.path.dirname(output_pose) or ".", exist_ok=True)
    os.makedirs(os.path.dirname(output_data) or ".", exist_ok=True)

    with tempfile.TemporaryDirectory() as tmp_dir:
        elapsed, pose, had_resample = run_reference(os.path.abspath(args.data), tmp_dir, False)
        shutil.move(os.path.join(tmp_dir, "pose.json"), output_pose)
        shutil.move(os.path.join(tmp_dir, "resampled.hdf5"), output_data)

    print(f"search+resample done in {elapsed:.1f}s")
    print(f"output pose: {output_pose}")
    print(f"output hdf5: {output_data}")


if __name__ == "__main__":
    main()
