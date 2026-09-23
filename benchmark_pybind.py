#!/usr/bin/env python3
"""Times every backend N times (default 10) on each dataset and prints the
mean/stdev per backend, in the same form as the earlier
runs/session2_full_10x_results.jsonl benchmark: one fresh process per rep,
doing the full flow (read HDF5 -> search -> resample -> write HDF5), and
"elapsed" is that process's wall time.

Backends timed:
  pybind-cpp / pybind-opencl / pybind-cpu
      projection-center-cpp/'s pybind11 interface (_backend.search +
      _backend.resample, backend="cpp"/"opencl"/"cpu"), run through
      projection-center-cpp/run_pybind.py's flow. Each rep also
      reports its own stage times (import incl. torch, HDF5 read, search,
      resample, HDF5 write), so the fixed cost of `import torch` is visible
      separately from the actual work.
  cli-cpp / cli-opencl / cli-cpu
      `projection-center pipeline --backend ...`, same as the earlier
      benchmark, for a like-for-like comparison with its numbers.

Usage:
    python3 benchmark_pybind.py
    python3 benchmark_pybind.py --reps 10 --datasets 128 512 real --backends pybind-cpp cli-cpp
"""
import argparse
import json
import os
import statistics
import subprocess
import sys
import tempfile
import time

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
CPP_DIR = os.path.join(REPO_ROOT, "projection-center-cpp")

DATASETS = {
    "128": "data/proj_shepplogan128.hdf5",
    "512": "data/proj_shepplogan512.hdf5",
    "real": "data/projs_change.hdf5",
}
BACKENDS = ["pybind-cpp", "pybind-opencl", "pybind-cpu", "cli-cpp", "cli-opencl", "cli-cpu"]

# Runs in a fresh interpreter per rep: the same flow as run_pybind.py
# (default search range, downsample 1), timing each stage, printing one JSON
# line on stdout.
PYBIND_WORKER = r"""
import json, sys, time
t0 = time.perf_counter()
import h5py
from backend import _backend
t_import = time.perf_counter() - t0
data, out_dir, backend = sys.argv[1], sys.argv[2], sys.argv[3]

t = time.perf_counter()
with h5py.File(data, "r") as f:
    projs = f["Projection"][()]
    SDD, SOD, px = f["SDD"][()], f["SOD"][()], f["pixelSize"][()]
    passthrough = {k: f[k][()] for k in ("voxelSize", "Volumen_num_xz", "Volumen_num_y", "num_projs", "Angle")}
H, W = projs.shape[1:]
t_read = time.perf_counter() - t

t = time.perf_counter()
pose = _backend.search(projs, SDD, SOD, px, W, H, xshift=40.0, alpha=10.0, beta=10.0,
                       xshift_step=1.0, alpha_step=1.0, beta_step=1.0, backend=backend)
t_search = time.perf_counter() - t

t = time.perf_counter()
res = _backend.resample(projs, SDD, SOD, px, xshift=pose["xshift"], alpha=pose["alpha"], beta=pose["beta"],
                        center_x=pose["center_x"], center_y=pose["center_y"], backend=backend)
t_resample = time.perf_counter() - t

t = time.perf_counter()
with h5py.File(out_dir + "/pose.h5", "w") as f:
    for k in ("xshift", "alpha", "beta", "MSE"):
        f[k] = pose[k]
    f["center_point"] = [pose["center_x"], pose["center_y"]]
with h5py.File(out_dir + "/resampled.h5", "w") as f:
    for k, v in passthrough.items():
        f[k] = v
    f["SDD"], f["SOD"], f["pixelSize"] = res["SDD"], res["SOD"], res["pixel_size"]
    f["detector_width"], f["detector_height"] = float(res["detector_width"]), float(res["detector_height"])
    f["Projection"] = res["projections"]
t_write = time.perf_counter() - t

print(json.dumps({"pose": pose, "stages": {"import": t_import, "read": t_read, "search": t_search,
                                           "resample": t_resample, "write": t_write}}))
"""


def run_rep(backend, data_path, tmp):
    if backend.startswith("pybind-"):
        cmd = [sys.executable, "-c", PYBIND_WORKER, data_path, tmp, backend.split("-", 1)[1]]
        cwd = CPP_DIR
    else:
        cmd = ["projection-center", "pipeline", "--backend", backend.split("-", 1)[1],
               "--data", data_path, "--output-pose", os.path.join(tmp, "pose.json"),
               "--output-data", os.path.join(tmp, "resampled.h5")]
        cwd = REPO_ROOT
    t0 = time.perf_counter()
    proc = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)
    elapsed = time.perf_counter() - t0
    if proc.returncode != 0:
        raise RuntimeError(f"{backend} failed:\n{proc.stdout}\n{proc.stderr}")

    if backend.startswith("pybind-"):
        info = json.loads(proc.stdout.strip().splitlines()[-1])
    else:
        with open(os.path.join(tmp, "pose.json")) as f:
            info = {"pose": json.load(f), "stages": None}
    return elapsed, info


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=10)
    ap.add_argument("--datasets", nargs="+", default=list(DATASETS), choices=list(DATASETS))
    ap.add_argument("--backends", nargs="+", default=BACKENDS, choices=BACKENDS)
    ap.add_argument("--output", default="runs/pybind_10x_results.jsonl")
    args = ap.parse_args()

    os.makedirs(os.path.dirname(os.path.join(REPO_ROOT, args.output)), exist_ok=True)
    summary = []
    with open(os.path.join(REPO_ROOT, args.output), "a") as log:
        for ds in args.datasets:
            data_path = os.path.join(REPO_ROOT, DATASETS[ds])
            for backend in args.backends:
                times, stages = [], []
                for rep in range(args.reps):
                    with tempfile.TemporaryDirectory() as tmp:
                        elapsed, info = run_rep(backend, data_path, tmp)
                    times.append(elapsed)
                    if info["stages"]:
                        stages.append(info["stages"])
                    log.write(json.dumps({"dataset": ds, "backend": backend, "rep": rep,
                                          "elapsed": elapsed, **info}) + "\n")
                    log.flush()
                    print(f"{ds:>4} {backend:<14} rep {rep}: {elapsed:.2f}s", flush=True)
                pose = info["pose"]
                summary.append((ds, backend, times, stages, pose))

    print("\n=== summary (mean ± stdev over", args.reps, "reps, wall time per full run) ===")
    for ds, backend, times, stages, pose in summary:
        mse = pose["MSE"]
        line = (f"{ds:>4} {backend:<14} {statistics.mean(times):8.2f}s ± {statistics.stdev(times) if len(times) > 1 else 0:.2f}"
                f"   pose xshift={pose['xshift'] * 1000:.1f}mm alpha={pose['alpha'] * 57.29577951308232:.1f}° "
                f"beta={pose['beta'] * 57.29577951308232:.1f}° MSE={mse:.4e}")
        if stages:
            m = {k: statistics.mean(s[k] for s in stages) for k in stages[0]}
            line += (f"\n{'':20}stages: import {m['import']:.2f}s, read {m['read']:.2f}s, "
                     f"search {m['search']:.2f}s, resample {m['resample']:.2f}s, write {m['write']:.2f}s")
        print(line)


if __name__ == "__main__":
    main()
