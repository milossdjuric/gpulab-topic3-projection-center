# gpulab-topic3-projection-center

CIS GPU Lab course project — Topic 3-4: Projection Center Searching and
Resampling. Cone-beam CT (CBCT) reconstruction needs an accurate detector
center of rotation (COR); this project finds it via a GPU-accelerated grid
search and (eventually) uses it to resample the raw projections.

The project has two halves:

1. **Forward search** — grid-searches the detector misalignment
   `(xshift, alpha, beta)` for the pose that minimizes a symmetry-based MSE
   metric. Implemented: serial Python reference (`Topic_3_forwardsearching.py`,
   untouched) plus a C++/OpenCL GPU port (`forward_search/`) with both
   `opencl-buffer` and `opencl-image` kernel variants, HDF5 I/O, and a
   pybind11/`torch.utils.cpp_extension` Python interface. Validated against
   the real dataset on this machine's Intel iGPU: ~13-16x speedup, MSE
   within 6.5e-6 relative of the reference. See `docs/ARCHITECTURE.md` for
   the full design writeup and `forward_search/README.md` for build/run
   instructions.
2. **Resampling** — uses the found pose to re-sample every projection as if
   captured by a centered detector. The serial Python reference
   (`Topic_3_resampling.py`) is untouched, as required; the GPU
   implementation lives in `projection-center/` (Python/PyOpenCL, merged in
   from a teammate's separate repo — see below).

This repo merges two previously separate repos: this one (C++/OpenCL
forward search) and `github.com/tanya1019/projection-center` (Python/PyOpenCL
forward search + resampling). **`projection-center/` — the imported
package — is the single user interface for the whole project now**; this
repo's own contribution is connected in at exactly one point, forward
search's runtime execution, reachable as one more `--backend` choice
(`cpp`) alongside their own `opencl`/`cpu` backends. Nothing else changed
about their CLI, commands, or package layout. Resampling only ever runs
through their own `opencl`/`cpu` kernels — `forward_search/` has none.

```bash
projection-center search --backend cpp    --data data/projs_change.hdf5  # this repo's C++/OpenCL, via forward_search/
projection-center search --backend opencl --data data/projs_change.hdf5  # the imported package's own PyOpenCL kernel
projection-center search --backend cpu    --data data/projs_change.hdf5  # the imported package's own NumPy fallback
```

All three have been run end-to-end against the real dataset, reproducibly
(rerun twice each), and agree on results — see `docs/ARCHITECTURE.md` §11
and `projection-center/README.md` for the full walkthrough of how the `cpp`
backend is wired in and why.

## Where to look

- `docs/ARCHITECTURE.md` — full design doc: algorithm, GPU kernel design,
  Python interface, testing/validation, known issues, course requirement
  status, report mapping. Covers the C++/OpenCL forward search in depth;
  `projection-center/README.md` covers the merged-in Python package.
- `forward_search/README.md` — build/run quick reference for the C++/OpenCL
  CLI and its Python (pybind11) backend.
- `projection-center/README.md` — the project's CLI reference: all three
  `--backend` choices (`opencl`, `cpu`, `cpp`), what each does, and how the
  `cpp` one is wired to `forward_search/`.
- `PROGRESS_REPORT.md` / `PROGRESS_REPORT.pdf` — the mid-term progress
  report submitted for the forward-search half (predates both the real-data
  validation results in `docs/ARCHITECTURE.md` §6.4 and this repo merge).
- `runs/` — logs and outputs from real validation runs (gitignored).
- `data/` — local copy of the real input dataset (gitignored).
