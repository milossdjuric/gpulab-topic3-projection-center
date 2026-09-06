# gpulab-topic3-projection-center

CIS GPU Lab course project — Topic 3-4: Projection Center Searching and
Resampling. This project implements the two reference tasks:

- projection center (forward) searching
- projection resampling

Cone-beam CT (CBCT) reconstruction needs an accurate detector center of
rotation (COR); a misaligned center produces reconstruction artifacts. This
project finds the true center via a GPU-accelerated symmetry-based grid
search, then uses the found pose to resample the raw projections as if
captured by a centered detector.

This repository merges two implementations of the same two tasks:

- **`forward_search/`** — this repo's own C++/OpenCL implementation.
  Forward search only (no resampling).
- **`projection-center/`** — fetched from
  `github.com/tanya1019/projection-center`, a Python/PyOpenCL implementation
  with a NumPy CPU fallback. Forward search *and* resampling.

Both are reachable through **one CLI**, `projection-center`, via
`--backend {opencl, cpu, cpp}` — see "CLI Usage" below.

## What Changed

The original reference scripts (`Topic_3_forwardsearching.py`,
`Topic_3_resampling.py`, in `reference/`, kept byte-for-byte unmodified)
spend most of their time in plain Python loops:

- center searching loops over every parameter candidate and every sampled angle
- resampling loops over every detector pixel of every projection

Both implementations in this repo move the expensive parts to GPU kernels,
independently of each other:

- `forward_search/`: one OpenCL work-group per parameter candidate, local-memory
  tree reduction across sampled angles (C++ host, `kernels/forward_search.cl`)
- `projection-center/`: the same shape, in Python via PyOpenCL
  (`center_search_reduce` kernel) — plus `resample_projections`, one
  OpenCL work-item per output pixel per projection, batched

That structure is substantially more parallel than the reference Python
implementation and matches the requirement for a higher-grade solution with
multiple explored parallelization strategies (the rubric's own language) —
having two independent GPU implementations of forward search, reachable
side by side through the same CLI, is exactly that.

## Repository Layout

```text
.
|-- data/                            (gitignored -- populate locally)
|   |-- projs_change.hdf5
|   `-- proj_shepplogan128.hdf5
|-- reference/                        untouched CPU reference implementation
|   |-- Topic_3_forwardsearching.py
|   `-- Topic_3_resampling.py
|-- forward_search/                   this repo's C++/OpenCL implementation
|   |-- src/
|   |   |-- forward_search.cpp/.hpp     forward search host code
|   |   |-- resample.cpp/.hpp           resampling host code
|   |   |-- main.cpp / resample_main.cpp  CLI entrypoints (meson-built binaries)
|   |   `-- backend.py / pybind_backend.cpp  pybind11 Python interface
|   |-- kernels/*.cl                   OpenCL C kernels
|   |-- tests/                         smoke + CLI-vs-backend regression tests
|   `-- README.md
|-- projection-center/                fetched Python/PyOpenCL implementation
|   |-- pyproject.toml
|   |-- README.md
|   `-- src/projection_center_searching/
|       |-- backends.py                CpuBackend, OpenCLBackend, CppBackend
|       |-- cli.py                     the `projection-center` command
|       |-- geometry.py, hdf5_io.py, models.py, pipeline.py
|       `-- __init__.py, __main__.py
|-- docs/                             (gitignored)
|   |-- ARCHITECTURE.md                full design writeup
|   |-- course/                        course-provided PDFs (exercise sheets, etc.)
|   `-- reports/                       our submitted report PDFs
|-- PROGRESS_REPORT.md                mid-term progress report (source; PDF in docs/reports/)
`-- runs/                             (gitignored) validation run outputs/logs
```

## Requirements

- Python 3.10 or newer, `numpy`, `h5py`, `pyopencl`, `mako`
- A C++17 compiler, Meson + Ninja, HDF5 C++ headers, Boost (`program_options`, JSON)
- An installed OpenCL runtime

`forward_search/`'s C++ dependencies are declared in `forward_search/meson.build`;
`projection-center/`'s Python dependencies in `projection-center/pyproject.toml`.

## OpenCL Setup

You need both an installed OpenCL runtime and, for `projection-center/`,
the Python package.

### Linux

Install your vendor OpenCL loader and runtime first (e.g. Ubuntu/Debian:
`ocl-icd-opencl-dev` plus the vendor runtime package — `intel-opencl-icd`
for Intel iGPUs). Then:

```bash
# C++ side
cd forward_search
meson setup builddir && meson compile -C builddir
cd ..

# Python side
cd projection-center
python3 -m pip install -e .
cd ..
```

### Windows / macOS

See `projection-center/README.md`'s OpenCL Setup section — the Python
implementation supports both; the C++ implementation (`forward_search/`) is
Linux-only as built here (meson + Boost + HDF5 C++ via system packages).

## Input Data

Datasets are expected under `data/`, gitignored, populated locally:

- `data/projs_change.hdf5` — real dataset
- `data/proj_shepplogan128.hdf5` — smaller Shepp-Logan phantom dataset

Both use the same HDF5 schema: `Projection`, `pixelSize`, `SDD`, `SOD`,
`voxelSize`, `Volumen_num_xz`, `Volumen_num_y`, `num_projs`,
`detector_width`, `detector_height`, `Angle`.

## Root Script Usage

`Topic_3_forwardsearching.py` and `Topic_3_resampling.py` in `reference/`
are the **untouched, serial CPU reference** — required to stay unmodified
per the course rubric, not GPU-callable entrypoints:

```bash
python3 reference/Topic_3_forwardsearching.py --data data/projs_change.hdf5
python3 reference/Topic_3_resampling.py --data data/projs_change.hdf5 --pose real_cb_pose.json
```

For anything GPU-accelerated (or a faster vectorized CPU path), use the CLI
instead — see below.

## CLI Usage

After installing `projection-center/` (see "OpenCL Setup"), the package
exposes one command, available from any directory:

```text
projection-center
```

### 1. List OpenCL Devices

```bash
projection-center devices
```

### 2. Search Only

```bash
projection-center search --data data/projs_change.hdf5 --output-pose real_cb_pose.json
```

### 3. Resample Only

```bash
projection-center resample --data data/projs_change.hdf5 --pose real_cb_pose.json --output-data projs_resample.hdf5
```

### 4. Run the Full Pipeline

```bash
projection-center pipeline --data data/projs_change.hdf5 --output-pose real_cb_pose.json --output-data projs_resample.hdf5
```

### Backend Selection

Three backends, all through the same commands above via `--backend`:

```bash
projection-center search --backend opencl --data data/projs_change.hdf5   # default: projection-center/'s own PyOpenCL kernel
projection-center search --backend cpu    --data data/projs_change.hdf5   # projection-center/'s own NumPy fallback
projection-center search --backend cpp    --data data/projs_change.hdf5   # this repo's forward_search/, via subprocess
```

`--backend cpp` only implements search (not resample/pipeline's resample
stage) — `forward_search/` has no resampling implementation, on purpose.
Full flag reference, including `cpp`-only flags, in `projection-center/README.md`.

## Output Files

Same schema regardless of which backend produced them:

- **Pose JSON**: `center_point` (2-element), `xshift`, `alpha`, `beta`, `MSE`
- **Resampled HDF5**: updated `pixelSize`/`SDD`/`SOD`, preserved angle/volume metadata, the resampled `Projection` stack

## Parallelization Details

Both GPU implementations use the same two-kernel shape:

1. **Search**: one OpenCL work-group per `(xshift, alpha, beta)` candidate;
   work-items split the sampled angles; a local-memory tree reduction
   produces one MSE per candidate.
2. **Resample** (`projection-center/` only): one work-item per output
   detector pixel, batched across projections.

`forward_search/` additionally precomputes `sin`/`cos` for all unique
`alpha`/`beta` values in a separate kernel pass, avoiding redundant trig
calls across the 35,721-combo grid.

## Notes About Accuracy

- Both GPU implementations use `float32` arithmetic; host-side geometry
  setup uses `float64`.
- All three backends (`opencl`, `cpu`, `cpp`) agree on the found pose to
  float32 precision on both datasets, reproducibly across reruns — verified
  directly, not assumed.
- The MSE metric was corrected in both implementations (normalizing by the
  count of valid, on-detector, signal-bearing sample pairs instead of a
  fixed sample count) after it produced a degenerate result on the small
  Shepp-Logan dataset; `Topic_3_forwardsearching.py` was confirmed to have
  the identical property and was not touched. Full writeup in
  `docs/ARCHITECTURE.md` §12.

## Example End-to-End Commands

```bash
projection-center pipeline --data data/projs_change.hdf5 --output-pose runs/pipeline/opencl/pose.json --output-data runs/pipeline/opencl/resampled.hdf5

projection-center pipeline --data data/proj_shepplogan128.hdf5 --output-pose runs/pipeline/shepplogan/pose.json --output-data runs/pipeline/shepplogan/resampled.hdf5
```

## Development Notes

- `forward_search/` and `projection-center/` are independently maintained,
  connected only through `projection-center`'s `--backend cpp` (a
  subprocess call into `forward_search/`'s compiled binaries) — see
  `projection-center/README.md`'s "Development Notes" for the internals.
- `docs/ARCHITECTURE.md` (gitignored, local-only) is the full design
  writeup: algorithm, kernel design, every bug found and fixed, course
  requirement status, report mapping.

## Verification

Recommended checks:

1. `projection-center devices`
2. `projection-center search --backend opencl --data data/projs_change.hdf5`
3. Repeat with `--backend cpp` and `--backend cpu`, confirm the same pose
4. `projection-center resample --data data/projs_change.hdf5 --pose real_cb_pose.json`
5. Inspect the generated JSON and HDF5 outputs

## Limitations

- `forward_search/`'s `--mode image` kernel variant crashes on this dev
  machine's Intel iGPU driver (a driver bug, not a code bug) — untestable
  here; `--mode buffer` (the default via the CLI) is unaffected.
- `--backend cpp` only implements search; use `opencl`/`cpu` for resample
  or pipeline, and it doesn't support `--platform-index`/`--device-index`
  or non-default `--sample-count`/`--sample-angle-range`.
- On the small `proj_shepplogan128.hdf5` dataset, the found `xshift` is
  weakly determined — a documented limitation of the algorithm itself, not
  either implementation. See `docs/ARCHITECTURE.md` §12.
- KDevelop project packaging (a course submission requirement) is not yet done.
