# Python Interface (pybind11) — How to Run

This is the Python interface the course requires: the OpenCL code is compiled
into a Python module with `torch.utils.cpp_extension` + pybind11, Python reads
all the data, passes it into the module, and gets the results back. It follows
the layout of the course's `pybindextension.zip` example (`backend.py` at the
top, the C++ sources under `src/`), and is used the same way:

```python
from backend import _backend
```

Everything below is run from inside `projection-center-cpp/`:

```bash
cd projection-center-cpp          # from the repo root (Topic_3/)
```

To stay in the repo root instead, put the folder in front of the script and
drop the `../` from the data path:

```bash
python3 projection-center-cpp/run_pybind.py --data data/proj_shepplogan128.hdf5
```

Running `python3 run_pybind.py ...` from the repo root fails with
`can't open file '.../Topic_3/run_pybind.py'`: the script is one folder down.

---

## 1. Requirements (once)

| What | Why | Check |
|---|---|---|
| Python 3.10+ with `torch`, `numpy`, `h5py` | `torch` provides `torch.utils.cpp_extension` **and** the pybind11 headers (no separate pybind11 install) | `python3 -c "import torch, numpy, h5py"` |
| `g++` (C++17) and `ninja` | `torch.utils.cpp_extension` compiles the C++ with them | `which g++ ninja` |
| OpenCL C++ header `CL/opencl.hpp` + ICD loader (`-lOpenCL`) and a GPU OpenCL driver. Ubuntu/Debian: `sudo apt install opencl-clhpp-headers ocl-icd-opencl-dev` + vendor driver (e.g. `intel-opencl-icd`) | the C++ code includes `CL/opencl.hpp`; the `cpp` and `opencl` backends run on the GPU | `ls /usr/include/CL/opencl.hpp` and `clinfo -l` |

A CPU-only PyTorch build is enough. The warning
`No CUDA runtime is found, using CUDA_HOME=...` is harmless: this project uses
OpenCL, not CUDA.

No build step is needed. The first `from backend import _backend` compiles the
module (about 40 s on the dev machine) and caches it in `~/.cache/torch_extensions/`;
after that it loads in well under a second.

---

## 2. Quick start: the full flow in one command

`run_pybind.py` does exactly what the course describes: Python reads
the HDF5 file, calls `_backend.search()`, passes the found pose to
`_backend.resample()`, and writes both results as HDF5.

```bash
cd projection-center-cpp
python3 run_pybind.py --data ../data/proj_shepplogan128.hdf5
```

Output:

- `runs/real_cb_pose.h5`: `xshift`, `alpha`, `beta`, `MSE`, `center_point`
- `runs/projs_resample.h5`: the corrected projections (`Projection`) plus the
  corrected geometry, same field names as the input file

Options:

| Option | Default | Meaning |
|---|---|---|
| `--data` | *(required)* | input HDF5 file |
| `--backend` | `cpp` | which implementation to run, see section 4 |
| `--xshift`, `--alpha`, `--beta` | `40`, `10`, `10` | search range ± (mm, degrees, degrees) |
| `--xshift-step`, `--alpha-step`, `--beta-step` | `1`, `1`, `1` | search step (mm, degrees, degrees) |
| `--mode` | `buffer` | `buffer` or `image` sinogram (`cpp` backend only) |
| `--downsample` | `1` | detector downsample factor for resample |
| `--output` | `runs/real_cb_pose.h5` | pose output file |
| `--resample-output` | `runs/projs_resample.h5` | resampled projections output file |
| `--skip-resample` | off | only run the search |

Every run writes to the same two output files by default, so each run
overwrites the previous one's results. To keep several, give each run its own
names:

```bash
python3 run_pybind.py --data ../data/proj_shepplogan512.hdf5 \
    --output runs/pose_512.h5 --resample-output runs/resampled_512.h5
```

What a run prints, the same on every backend (`--backend opencl` shown):

```text
import (torch + pybind module load): 6321 ms          # once per Python process
Loaded 500 projections, 128x128 (41 ms)
Device: Intel(R) UHD Graphics
  [+443ms] device picked
  [+443ms] context+queue created
  [+464ms] sinogram built (CPU)
Sinogram: 128x128 Buffer uploaded
  [+465ms] sinogram uploaded
Grid: 81x21x21 = 35721 combos
  [+479ms] program built (JIT compile)
WG_SIZE: 256
Forward search done
  [+519ms] search kernel done
Search kernel device time: 21.9132 ms
  [+519ms] results read back
MSE:    0.0006593746365979314
xshift: 40 mm
alpha:  -3 deg
beta:   4 deg
search kernel: 21.913166 ms
search total: 522 ms (opencl backend)
Wrote runs/real_cb_pose.h5: {...}
Device: Intel(R) UHD Graphics
Resampled 500 projections
resample total: 48 ms (opencl backend)
Wrote runs/projs_resample.h5: projections (500, 128, 128)
total (search + resample): 570 ms
timing (excluding imports): read 41 ms | compute 570 ms | write 71 ms | I/O 112 ms | compute+I/O 682 ms
total run: 7.0 s (including import and file I/O)
```

- `timing` splits the run into **read** (HDF5 input), **compute** (search +
  resample, including device setup and kernel compile), **write** (output
  files), **I/O** (read + write) and **compute+I/O**, all without the import.
  The `projection-center` CLI and the C++ binaries print the same line.
- `total run` is the whole process, import included.
- Lines a backend has no step for are left out: `cpp` also prints `Trig
  precomputation done`; `cpu` prints `Device: CPU (no OpenCL)` and no GPU
  lines (`WG_SIZE`, kernel times).
- In the `Wrote runs/real_cb_pose.h5` line the pose is in meters/radians
  (`0.04` = 40 mm, `-0.0524` = -3°).

Examples:

```bash
# the real dataset
python3 run_pybind.py --data ../data/projs_change.hdf5

# the same flow on each implementation
python3 run_pybind.py --data ../data/proj_shepplogan128.hdf5 --backend cpp
python3 run_pybind.py --data ../data/proj_shepplogan128.hdf5 --backend opencl
python3 run_pybind.py --data ../data/proj_shepplogan128.hdf5 --backend cpu

# search only
python3 run_pybind.py --data ../data/proj_shepplogan128.hdf5 --skip-resample
```

---

## 3. Calling it from your own Python code

```python
import h5py
from backend import _backend      # run from projection-center-cpp/, or put it on sys.path

# 1. Python reads the data
with h5py.File("../data/proj_shepplogan128.hdf5", "r") as f:
    projs = f["Projection"][()]                  # float32 array (num_projs, H, W)
    SDD, SOD, pixel_size = f["SDD"][()], f["SOD"][()], f["pixelSize"][()]
H, W = projs.shape[1:]

# 2. pass it into the interface: forward search
pose = _backend.search(projs, SDD, SOD, pixel_size, W, H,
                       xshift=40.0, alpha=10.0, beta=10.0,
                       xshift_step=1.0, alpha_step=1.0, beta_step=1.0)
print(pose)   # {'xshift': ..., 'alpha': ..., 'beta': ..., 'MSE': ..., 'center_x': ..., 'center_y': ..., 'kernel_ms': ...}

# 3. resample with the found pose
out = _backend.resample(projs, SDD, SOD, pixel_size,
                        xshift=pose["xshift"], alpha=pose["alpha"], beta=pose["beta"],
                        center_x=pose["center_x"], center_y=pose["center_y"])
corrected = out["projections"]                   # float32 array (num_projs, H, W)
```

### `_backend.search(...)` → `dict`

| Argument | Meaning |
|---|---|
| `projections` | NumPy array, shape `(num_projs, H, W)`, converted to float32 if needed |
| `SDD`, `SOD`, `pixel_size` | scan geometry, same units as the HDF5 file |
| `detector_width`, `detector_height` | must equal `W`, `H` of the array |
| `xshift`, `alpha`, `beta` | search range ± in **mm**, **degrees**, **degrees** |
| `xshift_step`, `alpha_step`, `beta_step` | search step in **mm**, **degrees**, **degrees** |
| `mode="buffer"` | `"buffer"` or `"image"`; `cpp` backend only |
| `backend="cpp"` | `"cpp"`, `"opencl"` or `"cpu"` |

Returns `xshift` (**meters**), `alpha`, `beta` (**radians**), `MSE`,
`center_x`, `center_y` (the found projection center on the detector) and
`kernel_ms` (GPU time of the search kernel; `None` for `cpu`).

### `_backend.resample(...)` → `dict`

| Argument | Meaning |
|---|---|
| `projections`, `SDD`, `SOD`, `pixel_size` | same as for `search()` |
| `xshift`, `alpha`, `beta`, `center_x`, `center_y` | the pose, exactly as `search()` returned it (meters/radians) |
| `downsample=1` | detector downsample factor |
| `batch_size=16` | projections sent to the GPU at once |
| `backend="cpp"` | `"cpp"`, `"opencl"` or `"cpu"` |

Returns `projections` (NumPy float32 array, `(num_projs, out_H, out_W)`) and the
corrected geometry: `SDD`, `SOD`, `pixel_size`, `detector_width`,
`detector_height`.

---

## 4. The three backends

All three sit behind the same two functions; only `backend=` changes.

| `backend=` | What runs | Where |
|---|---|---|
| `"cpp"` (default) | our own OpenCL kernels, the same code as the C++ command-line programs | `kernels/forward_search.cl`, `kernels/resample.cl`, `src/forward_search.cpp`, `src/resample.cpp` |
| `"opencl"` | `projection-center-python-opencl/`'s OpenCL kernels, run from C++ instead of PyOpenCL | `kernels/python_opencl_port.cl`, `src/ported_backends.cpp` |
| `"cpu"` | `projection-center-python-opencl/`'s NumPy CPU backend, rewritten in plain C++; no GPU | `src/ported_backends.cpp` |

All three find the same pose on every dataset. `opencl` and `cpu` give
bit-identical results to the Python code they were ported from (checked by
`tests/test_ported_backends.py`).

`kernels/python_opencl_port.cl` is an exact copy of `KERNEL_SOURCE` in
`projection-center-python-opencl/src/projection_center_searching/backends.py`.
If that kernel is ever changed, re-copy it:

```bash
cd ..   # repo root
python3 -c "import sys; sys.path.insert(0, 'projection-center-python-opencl/src'); \
from projection_center_searching.backends import KERNEL_SOURCE; \
open('projection-center-cpp/kernels/python_opencl_port.cl', 'w').write(KERNEL_SOURCE)"
```

---

## 5. Tests

From `projection-center-cpp/` (the CLI tests need the meson build in
`builddir/` first: `meson setup builddir && meson compile -C builddir`):

```bash
python3 tests/test_backend_smoke.py              # module builds, loads, returns a sane pose
python3 tests/test_cli_vs_backend.py             # pybind search == CLI search
python3 tests/test_resample_cli_vs_backend.py    # pybind resample == CLI resample, bit-identical
python3 tests/test_ported_backends.py            # backend="opencl"/"cpu" == the Python originals (~40 s;
                                                 # needs projection-center-python-opencl installed)
```

Each prints `... PASS` at the end.

---

## 6. Timing

From the repo root:

```bash
python3 benchmark_pybind.py                                   # everything: 10 reps × 6 backends × 3 datasets
python3 benchmark_pybind.py --reps 10 --datasets 128 --backends pybind-cpp pybind-opencl pybind-cpu
```

Each rep is a fresh process running the full flow (read → search → resample →
write), the same method as the earlier benchmark report. Prints mean ± stdev
per backend, plus the per-stage times of the pybind runs. Raw results are
appended to `runs/pybind_10x_results.jsonl`.

Note: `import torch` takes about 5 s, once per Python process. That's a fixed
cost of the required `torch.utils.cpp_extension` interface, not of the
computation, and is listed separately as the `import` stage.

---

## 7. Troubleshooting

| Problem | Fix |
|---|---|
| `ModuleNotFoundError: No module named 'backend'` | run from `projection-center-cpp/`, or `sys.path.insert(0, "<repo>/projection-center-cpp")` first |
| `No CUDA runtime is found` warning | harmless, ignore |
| Changed a `.cpp`/`.hpp` file but nothing changes | `load()` rebuilds changed sources automatically; if it doesn't, clear the cache: `rm -rf ~/.cache/torch_extensions/*/forward_search_backend` |
| Changed a `.cl` kernel | nothing to rebuild; kernels are compiled at run time from `kernels/` |
| `mode="image"` crashes | known Intel iGPU driver bug on the dev machine; use `mode="buffer"` (the default) |
| `No GPU device found` | no OpenCL GPU driver visible (`clinfo -l`); `backend="cpu"` still works |

---

## 8. Files

```text
projection-center-cpp/
|-- backend.py                 torch.utils.cpp_extension.load(...) -> _backend
|-- run_pybind.py              full read -> search -> resample -> write
|-- src/
|   |-- pybind_backend.cpp     the pybind11 module: search(), resample()
|   |-- forward_search.cpp     backend="cpp" search
|   |-- resample.cpp           backend="cpp" resample
|   `-- ported_backends.cpp    backend="opencl" and backend="cpu"
|-- kernels/
|   |-- forward_search.cl, resample.cl   cpp kernels
|   `-- python_opencl_port.cl            opencl kernels (copy of the Python package's)
`-- tests/                     see section 5
```
