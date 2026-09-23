# Forward Search — GPU (OpenCL) Cone-Beam CT Center-of-Rotation Search

C++/OpenCL port of `Topic_3_forwardsearching.py`. Finds the projection center
`(xshift, alpha, beta)` that minimizes a symmetry-based MSE metric over a
grid search, using the GPU for the compute-heavy inner loop.

## Dependencies

- A C++17 compiler (g++ or clang++)
- [Meson](https://mesonbuild.com/) + Ninja
- An OpenCL ICD loader + a GPU driver providing an OpenCL platform
  (`ocl-icd-opencl-dev`, plus a vendor driver — e.g. `intel-opencl-icd` for
  Intel iGPUs)
- HDF5 C++ headers/libs (`libhdf5-dev`)
- Boost `program_options` (`libboost-program-options-dev`)

On Ubuntu:

```bash
sudo apt install meson ninja-build ocl-icd-opencl-dev opencl-clhpp-headers libhdf5-dev libboost-program-options-dev libboost-json-dev
```

Check that a GPU OpenCL platform is visible before building:

```bash
clinfo -l   # should list at least one Device Type: GPU
```

## Build

```bash
cd projection-center-cpp
meson setup builddir      # first time only
meson compile -C builddir # or: ninja -C builddir
```

`meson.build` sets `buildtype=release` by default (fixed 2026-08-25 — it
previously set nothing, which meant meson's own default of `buildtype=debug`
i.e. `-O0`; `buildSinogram()`'s CPU loop alone took ~11.4s at `-O0` vs
~0.3-0.8s at `-O3`, ~240x more than the GPU search kernel's own ~48ms). If you already have
an existing `builddir/` configured before this fix, `meson setup` won't
retroactively apply the new default — reconfigure it once:

```bash
meson configure builddir --buildtype=release
meson compile -C builddir
```

The binary now also prints stage timing (`[+Xms] ...` lines on stderr) —
useful for spotting where time is actually going if this ever regresses
again.

## Run

```bash
./builddir/forward_search --data /path/to/projs_change.hdf5
```

### Options

| Flag | Default | Meaning |
|---|---|---|
| `--data` | `/lgrp/edu-2026-1-gpulab/projs_change.hdf5` | Path to input HDF5 projections |
| `--xshift` | `40.0` | Search range for xshift, ± mm |
| `--alpha` | `10.0` | Search range for alpha, ± degrees |
| `--beta` | `10.0` | Search range for beta, ± degrees |
| `--xshift-step` | `1.0` | Grid step for xshift, mm |
| `--alpha-step` | `1.0` | Grid step for alpha, degrees |
| `--beta-step` | `1.0` | Grid step for beta, degrees |
| `--kernel` | `kernels/forward_search.cl` | Path to the OpenCL kernel source |
| `--mode` | `image` | Sinogram data format: `image` (Image2D+sampler) or `buffer` (manual bilinear) |
| `--output` | `runs/forward_search_cli/real_cb_pose.h5` | Output HDF5 path |

Run outputs, logs, and local dataset copies aren't meant to be committed —
`--output` defaults into `runs/` (created automatically if missing), which is
gitignored, same as `data/` for local `.hdf5` input copies.

Note: the CLI defaults to `--mode image`, while the Python interface below defaults
to `mode="buffer"` — deliberately different, because `image` mode crashes on this
dev machine's driver (see the caveat at the end of the Python interface section).
Pass `--mode buffer` to the CLI explicitly if testing on this machine.

### Input HDF5 schema

Datasets expected in the input file (same as the Python reference):
`voxelSize`, `Volumen_num_xz`, `Volumen_num_y`, `SDD`, `SOD`, `pixelSize`,
`num_projs`, `detector_width`, `detector_height`, `Angle` (per-projection
array), `Projection` (shape `num_projs × detector_height × detector_width`).

### Output HDF5 schema

`xshift`, `alpha`, `beta`, `MSE` as scalar datasets (radians for angles,
meters for xshift — same units as internal computation), plus
`center_point` as a 2-element dataset `[x_0, y_0]`.

## Validating against the Python reference

```bash
python3 ../validate_forward_search.py --gpu runs/forward_search_cli/real_cb_pose.h5 --run-ref \
    --data /path/to/projs_change.hdf5
```

This runs the Python reference, compares its JSON output against the GPU's
HDF5 output (MSE within 1% tolerance, pose parameters within tight absolute
tolerance, or within one grid step — see caveat below), and prints a
pass/fail table.

**Already done once, real data, this machine:** `--mode buffer` against the
real `projs_change.hdf5` (180×1024×1024) — Python reference 10m33s vs GPU
~40-49s (~13-16x speedup), MSE relative diff 6.5e-6, identical winning pose.
See `runs/` for the raw logs/outputs. `--mode image` hasn't been run against real data yet (see the
driver caveat below).

**Known reference bug (not a GPU defect):** `get_linear_interpolate_MSE` in
`Topic_3_forwardsearching.py` declares `f_theta`/`f_rtheta` outside the
per-ray bounds check and never resets them to 0, so whenever a ray falls
outside the detector it silently reuses the *previous* ray's interpolated
value instead of contributing 0. On `projs_change.hdf5` this hits ~47% of
rays at some grid points, which biases the reference's MSE and can shift its
argmin by one grid step in flat regions of the search landscape (confirmed:
patching the reference to zero-init those variables reproduces the GPU's MSE
to 5 significant figures). The GPU kernel already implements the correct
zero-on-out-of-bounds behavior (`bilinear_buffer` in the `.cl` file), so
`validate_forward_search.py` treats an adjacent-bin match as a pass with a
note rather than a hard failure.

## Architecture

- **Kernel 1** (`precompute_trig`): 1D kernel, precomputes `sin`/`cos` for
  all unique `alpha`/`beta` values once, avoiding redundant trig calls
  across all parameter combos.
- **Kernel 2** (`forward_search_mse`): 2D ND-range, global `(P, WG_SIZE)`,
  local `(1, WG_SIZE)` — one work group per `(xshift, alpha, beta)` combo.
  Each work item handles a slice of the 1000 theta steps, accumulates a
  partial MSE sum, and a local-memory tree reduction combines partial sums
  into one MSE per combo. `WG_SIZE` is chosen at runtime from
  `CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE`.
- Sinogram is uploaded as an OpenCL `Image2D` with a `CLK_FILTER_LINEAR`
  sampler, so the GPU's texture units do the bilinear interpolation in one
  `read_imagef()` call instead of a manual 10-line bilinear block
  (`--mode image`, the CLI default).
- A second kernel, `forward_search_mse_buffer` (`--mode buffer`), does the
  same computation against a plain `cl::Buffer` sinogram with a manual
  bilinear interpolation function (`bilinear_buffer` in the `.cl` file)
  instead of the hardware sampler — satisfying the course's requirement for
  both an opencl-image and an opencl-buffer implementation, and the only
  mode validated on this dev machine (see the Python interface section).

## Python interface (pybind11 / torch.utils.cpp_extension)

Per course requirement, forward search and resample are both callable from
Python, without going through the CLI at all. Python reads the data, passes the
arrays in, and gets the results back.

Layout follows the course's `pybindextension.zip` example: `backend.py` at the
top of `projection-center-cpp/`, the C++ sources it compiles under `src/`.

```python
# run from projection-center-cpp/, or add it to sys.path first
from backend import _backend

result = _backend.search(
    projections,       # numpy array, shape (num_projs, H, W), float32
    SDD, SOD, pixel_size,
    detector_width, detector_height,
    xshift=40.0, alpha=10.0, beta=10.0,          # ± search range, mm / degrees
    xshift_step=1.0, alpha_step=1.0, beta_step=1.0,
    mode="buffer",      # or "image" — see the driver caveat below (cpp backend only)
    backend="cpp",      # "cpp" | "opencl" | "cpu" — see the table below
)
# result: {"xshift", "alpha", "beta", "MSE", "center_x", "center_y", "kernel_ms"}
#         (xshift in meters, alpha/beta in radians)

corrected = _backend.resample(
    projections,       # same array as above
    SDD, SOD, pixel_size,
    xshift=result["xshift"], alpha=result["alpha"], beta=result["beta"],
    center_x=result["center_x"], center_y=result["center_y"],
    downsample=1, batch_size=16,
    backend="cpp",      # same choices as search()
)
# corrected: {"projections" (numpy, num_projs x out_H x out_W), "SDD", "SOD",
#             "pixel_size", "detector_width", "detector_height"}
```

`resample()` takes the pose exactly as `search()` returns it, so the two chain
directly in memory. Both read the projections straight from the NumPy array's
memory, and `resample()` writes straight into the NumPy array it returns, so
the ~750MB real dataset is never copied on the way in or out.

`backend=` picks one of the project's three implementations, all behind the
same interface:

| backend | what runs | source |
|---|---|---|
| `"cpp"` (default) | this directory's OpenCL kernels, same code as the CLI binaries | `kernels/forward_search.cl`, `kernels/resample.cl`, `src/forward_search.cpp`, `src/resample.cpp` |
| `"opencl"` | `projection-center-python-opencl/`'s OpenCL kernels, driven from C++ instead of PyOpenCL | `kernels/python_opencl_port.cl` (verbatim copy of its `KERNEL_SOURCE`), `src/ported_backends.cpp` |
| `"cpu"` | `projection-center-python-opencl/`'s NumPy CPU backend, ported to plain C++ (no GPU) | `src/ported_backends.cpp` |

`"opencl"` and `"cpu"` follow the Python package's `geometry.py`/`backends.py`
step for step, and `tests/test_ported_backends.py` checks they give the same
pose and bit-identical resampled output as the Python originals.

The first call JIT-compiles `src/pybind_backend.cpp` + `src/forward_search.cpp` + `src/resample.cpp` + `src/ported_backends.cpp` via
`torch.utils.cpp_extension.load()` (cached afterwards, so later calls are fast).
No separate build step, no meson — this path is entirely independent of the
CLI's `builddir/`, including its optimization flags: `backend.py` passes
`-O3` explicitly in `extra_cflags` (added 2026-08-25 — `torch.utils.cpp_extension`
sets no optimization flag by default, so this path had the same `-O0`
slowdown the CLI's missing `meson.build buildtype` caused). If you built the extension before this fix,
clear the JIT cache once to pick it up:
`rm -rf ~/.cache/torch_extensions/*/forward_search_backend`.

`run_pybind.py` runs the full read-HDF5 → search → resample → write-HDF5 flow,
matching the flow the course requires (Python owns all I/O; the backend is pure
compute).

`import torch` itself takes ~5s on this machine, once per Python process
(`load()` of the cached extension then takes ~0.03s). That's a fixed cost of
the required `torch.utils.cpp_extension` interface, not of the computation.

Tests: `tests/test_backend_smoke.py` (backend runs and returns sane output),
`tests/test_cli_vs_backend.py` (backend and CLI search agree on identical input),
`tests/test_resample_cli_vs_backend.py` (backend and CLI resample produce
bit-identical output) and `tests/test_ported_backends.py` (`backend="opencl"`/`"cpu"`
match the Python package's own opencl/cpu backends) — run them with
`python3 tests/<name>.py` from the `projection-center-cpp/` directory.

Timing: `benchmark_pybind.py` at the repo root runs every backend (pybind and
the unified CLI) 10x per dataset and prints mean ± stdev.

**Known driver caveat (this dev machine only):** `mode="image"` crashes on this
machine's Intel NEO OpenCL driver — a driver bug unrelated to this project's code
(see `env-opencl-gpu-setup` memory / `docs/superpowers/specs/2026-06-26-forward-search-design.md`).
Use `mode="buffer"` for local testing here; test `mode="image"` on the actual lab
GPU machine before relying on it for the report.
