# Projection Center Searching and Resampling

This package implements the two reference tasks in the repository:

- projection center searching
- projection resampling

Fetched from `github.com/tanya1019/projection-center` and merged into this
repository as its unified CLI. The implementation is organized as a
reusable Python package with:

- an OpenCL backend for GPU execution
- a NumPy CPU fallback for environments without OpenCL
- **this repository's own C++/OpenCL implementation** (`forward_search/`),
  reached as a third backend
- HDF5 input and output compatible with the provided reference scripts
- a CLI that runs on Windows, Linux, and macOS

Unlike the source repository, this copy does **not** expose four root
wrapper scripts — see "Root Script Usage" below for why, and the CLI
equivalents to use instead.

## What Changed

The original reference code spends most of its time in Python loops:

- center searching loops over every parameter candidate and every sampled angle
- resampling loops over every detector pixel of every projection

This version moves both expensive parts to GPU kernels:

- `center_search_reduce`: one OpenCL work-group per parameter candidate, with parallel reduction across sampled angles
- `resample_projections`: one OpenCL work-item per output pixel per projection

That structure is substantially more parallel than the reference Python implementation and matches the requirement for a higher-grade solution with better parallelization.

This repository's own C++/OpenCL implementation (`forward_search/`) uses the
same shape independently, reachable here as `--backend cpp`.

## Repository Layout

```text
.
|-- data/                        (gitignored -- populate locally)
|   |-- projs_change.hdf5
|   `-- proj_shepplogan128.hdf5
|-- Topic_3_forwardsearching.py  (untouched CPU reference -- repo root, not here)
|-- Topic_3_resampling.py        (untouched CPU reference -- repo root, not here)
|-- forward_search/               (this repo's C++/OpenCL implementation -- reached via --backend cpp)
|-- projection-center/            (this package)
|   |-- pyproject.toml
|   |-- README.md
|   `-- src/
|       `-- projection_center_searching/
|           |-- __init__.py
|           |-- backends.py
|           |-- cli.py
|           |-- geometry.py
|           |-- hdf5_io.py
|           |-- models.py
|           `-- pipeline.py
`-- docs/ARCHITECTURE.md          (full design writeup, gitignored)
```

## Requirements

- Python 3.10 or newer
- `numpy`
- `h5py`
- `pyopencl`
- `mako` (required by `pyopencl` at runtime; not declared upstream — see "Fixes Applied In This Copy")
- an installed OpenCL runtime

Python dependencies are declared in `pyproject.toml`.

## OpenCL Setup

You need both the Python package and a system OpenCL runtime.

### Windows

Install:

- NVIDIA driver with OpenCL support, if using an NVIDIA GPU
- AMD Adrenalin driver or ROCm-compatible OpenCL runtime, if using an AMD GPU
- Intel Graphics Driver or Intel oneAPI OpenCL runtime, if using Intel GPU or CPU OpenCL

Then install the Python package:

```powershell
py -m pip install -e .
```

If `py` is not available on your machine, use your Python executable directly:

```powershell
python -m pip install -e .
```

### Linux

Install your vendor OpenCL loader and runtime first. Common packages include:

- Ubuntu or Debian: `ocl-icd-opencl-dev`, plus vendor runtime packages
- Fedora: `ocl-icd`, plus vendor runtime packages
- Arch: `ocl-icd`, plus vendor runtime packages

Then install Python dependencies:

```bash
cd projection-center
python3 -m pip install -e .
```

### macOS

`pyopencl` can still be used, but macOS OpenCL support is deprecated and depends on Apple's system framework. For reliable GPU execution, Linux or Windows is preferred.

Install:

```bash
python3 -m pip install -e .
```

If OpenCL is unavailable on your macOS machine, use the CPU backend:

```bash
projection-center pipeline --backend cpu --data data/projs_change.hdf5
```

## Input Data

Datasets are expected under `data/` (repo root), but that directory is
ignored by Git and must be populated locally:

- `data/projs_change.hdf5`
- `data/proj_shepplogan128.hdf5`

The reader expects the same HDF5 keys used by the reference scripts:
`Projection`, `pixelSize`, `SDD`, `SOD`, `voxelSize`, `Volumen_num_xz`,
`Volumen_num_y`, `num_projs`, `detector_width`, `detector_height`, `Angle`.

## Root Script Usage

The source repository documented four root wrapper scripts
(`Topic_3_forwardsearching.py`, `_cpu.py`, `Topic_3_resampling.py`,
`_cpu.py`) as an alternative to the CLI. **They are not present here.**
Their filenames collide with this repository's canonical, untouched CPU
reference scripts at the repo root — those must stay byte-for-byte
unmodified per the course rubric, and in the source repo the same filenames
had been overwritten with GPU-calling entrypoints instead, losing the
original reference. Use the CLI equivalents instead:

| Source repo script | CLI equivalent here |
|---|---|
| `python Topic_3_forwardsearching.py --data ...` | `projection-center search --backend opencl --data ...` |
| `python Topic_3_forwardsearching_cpu.py --data ...` | `projection-center search --backend cpu --data ...` |
| `python Topic_3_resampling.py --data ... --pose ...` | `projection-center resample --backend opencl --data ... --pose ...` |
| `python Topic_3_resampling_cpu.py --data ... --pose ...` | `projection-center resample --backend cpu --data ... --pose ...` |

This repository additionally has no equivalent for `forward_search/`'s own
CLI (`--backend cpp`) in the source repo, since that implementation didn't
exist there — see "Backend Selection" below.

## CLI Usage

After installation, the package exposes the command:

```text
projection-center
```

### 1. List OpenCL Devices

```bash
projection-center devices
```

Example output:

```text
platform=0 device=0 name=... type=GPU
```

### 2. Search Only

```bash
projection-center search --data data/projs_change.hdf5 --output-pose real_cb_pose.json
```

Useful search parameters:

- `--xshift` search half-range in mm
- `--alpha` search half-range in degrees
- `--beta` search half-range in degrees
- `--xshift-step` search step in mm
- `--alpha-step` search step in degrees
- `--beta-step` search step in degrees
- `--sample-count` number of angular samples per candidate
- `--sample-angle-range` search sample span in degrees

Example with explicit ranges:

```bash
projection-center search --data data/projs_change.hdf5 --xshift 40 --alpha 10 --beta 10 --xshift-step 1 --alpha-step 1 --beta-step 1 --sample-count 1000 --sample-angle-range 30 --output-pose real_cb_pose.json
```

### 3. Resample Only

```bash
projection-center resample --data data/projs_change.hdf5 --pose real_cb_pose.json --output-data projs_resample.hdf5 --downsample 1 --batch-size 16
```

### 4. Run the Full Pipeline

```bash
projection-center pipeline --data data/projs_change.hdf5 --output-pose real_cb_pose.json --output-data projs_resample.hdf5
```

### Backend Selection

The default backend is OpenCL:

```bash
projection-center pipeline --data data/projs_change.hdf5
```

To force CPU execution:

```bash
projection-center pipeline --backend cpu --data data/projs_change.hdf5
```

To choose a specific OpenCL device:

```bash
projection-center pipeline --data data/projs_change.hdf5 --platform-index 0 --device-index 0
```

**This repository adds a third backend, `cpp`**, delegating to
`forward_search/`'s own C++/OpenCL implementation instead of this
package's kernels:

```bash
projection-center search --backend cpp --data data/projs_change.hdf5 --output-pose real_cb_pose.json
```

`--backend cpp` only implements search, not resample (`forward_search/` has
no resampling implementation, on purpose — resampling stays exclusively
this package's). It also has one extra flag, `--cpp-mode {image,buffer}`
(forwarding to `forward_search/`'s own `--mode`, default `buffer`), and
doesn't support `--platform-index`/`--device-index` or non-default
`--sample-count`/`--sample-angle-range` (see "Limitations").

## Output Files

### Pose JSON

Search writes a JSON file like:

```json
{
  "center_point": [0.0, 0.0],
  "xshift": 0.0,
  "alpha": 0.0,
  "beta": 0.0,
  "MSE": 0.0
}
```

### Resampled HDF5

Resampling writes a new HDF5 file containing:

- updated `pixelSize`
- updated `SDD`
- updated `SOD`
- preserved angle and volume metadata
- the resampled `Projection` stack

## Parallelization Details

### Center Search

The center search kernel is organized so that:

- each candidate `(xshift, alpha, beta)` uses one OpenCL work-group
- work-items inside the work-group process different angular samples in parallel
- a local-memory reduction produces one MSE value per candidate

That avoids launching one Python loop iteration per candidate and keeps the expensive interpolation work on the GPU.

### Resampling

The resampling kernel is organized so that:

- each work-item computes one output detector pixel
- the third global dimension is the projection index inside the current batch
- projections are processed in batches to control GPU memory use

This scales well for large detector sizes and large projection stacks.

The HDF5 pipeline is also streamed in batches during search and resampling, so the implementation does not need to load the full projection stack into memory before computation starts.

## Notes About Accuracy

- The GPU kernels use `float32` arithmetic for portability and speed.
- The host-side geometry setup uses `float64`.
- The CPU fallback uses NumPy vectorization and is intended for portability and debugging, not peak performance.
- All three backends (`opencl`, `cpu`, `cpp`) agree on the found pose to
  float32 precision on both datasets in this repository, reproducibly
  across reruns.

If you need strict numerical comparison against the original scripts, compare:

- the selected pose JSON
- the output HDF5 metadata
- slices or summary statistics from the resampled projection stack

## Example End-to-End Commands

```bash
projection-center pipeline --data data/projs_change.hdf5 --output-pose outputs/real_pose.json --output-data outputs/real_resampled.hdf5

projection-center pipeline --data data/proj_shepplogan128.hdf5 --output-pose outputs/shepp_pose.json --output-data outputs/shepp_resampled.hdf5
```

## Development Notes

- The new code does not use hardcoded Linux-only paths.
- Paths are handled through CLI arguments and `pathlib`.
- The package is suitable for installation on Windows, Linux, and macOS.
- `--backend cpp` is this repository's own addition, delegating to
  `forward_search/`'s compiled binaries via subprocess rather than
  reimplementing its kernels here.

## Verification

Recommended checks after installation:

1. `projection-center devices`
2. `projection-center search --data data/projs_change.hdf5`
3. `projection-center resample --data data/projs_change.hdf5 --pose real_cb_pose.json`
4. Inspect the generated JSON and HDF5 outputs
5. Repeat 2-3 with `--backend cpp` and `--backend cpu`, confirm the same pose

## Limitations

- This repository does not vendor GPU drivers or OpenCL runtimes.
- macOS OpenCL support is deprecated by Apple and may fall back to CPU-only workflows in practice.
- Runtime performance depends heavily on the installed OpenCL implementation and device memory.
- `--backend cpp` only implements search; use `opencl`/`cpu` for resample or pipeline.
- `--backend cpp` doesn't support `--platform-index`/`--device-index` (always uses the first GPU found) or non-default `--sample-count`/`--sample-angle-range` (`forward_search.cpp` hardcodes these).
- On the small `proj_shepplogan128.hdf5` dataset, the found `xshift` is weakly determined (a documented limitation of the algorithm itself, not this implementation) — see `docs/ARCHITECTURE.md` §12.

## Fixes Applied In This Copy

Not present upstream; found and fixed while integrating this package into
the merged repository. All verified to leave `projs_change.hdf5`'s result
unchanged.

1. **Missing `mako` dependency.** `pyopencl` needs it at runtime; neither
   `pyproject.toml` nor `requirements.txt` declared it, so
   `pip install -e .` succeeded but importing `pyopencl` raised
   `ModuleNotFoundError: mako`. Added `mako>=1.2` to both files.
2. **`OpenCLBackend.resample()` was 12x slower than necessary** (48s → 3.9s
   on the real dataset) — it recreated the rotation buffer, re-fetched the
   kernel, and reallocated GPU buffers on every one of the ~12 streamed
   batches instead of reusing them. Fixed by caching all three on the
   backend instance.
3. **MSE metric normalized by a fixed sample count**, letting a candidate
   pose fake a low MSE by pushing rays off-detector or into background —
   harmless on the large real dataset, but produced a degenerate MSE=0.0
   boundary result on the small `proj_shepplogan128.hdf5` dataset. Now
   normalizes by the count of valid, signal-bearing pairs instead.
   `Topic_3_forwardsearching.py` has the identical property and was not
   touched (confirmed it reproduces the same numbers).

Full write-ups, including a fourth fix that was tried and deliberately
reverted (it regressed agreement with the Python reference), are in
`docs/ARCHITECTURE.md` §11-§12.
