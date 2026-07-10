# Forward Search Progress

We've implemented the center-of-rotation forward search on the GPU in
C++/OpenCL, covering what the assignment asks for on the code side:

- GPU implementation alongside the CPU reference (we use the provided
  Python script as-is for the CPU side, which is what's expected)
- both the buffer and image data formats, as two separate kernels
- HDF5 output
- a commented, documented codebase, README included
- a Python interface (pybind11 + torch's C++ extension loader) that calls
  the exact same search function as the command-line tool, so there's no
  second implementation to keep in sync. We checked both entry points give
  identical results on the same input.

The catch is that all of this has only been run and checked against
synthetic, randomly generated data we made ourselves, shaped to match the
real dataset's format. So what we can currently say is that the pipeline
runs end to end and the buffer-mode kernel produces consistent, sensible
output on synthetic data. Real correctness, the kind the report needs,
still depends on running it against the actual projection data.

Things still genuinely open on our side:

- **We've only tested on a synthesized dataset, not the given one.** We
  generated a small synthetic dataset ourselves, matching the shape of the
  real file, just to prove the code runs and produces sensible output. We
  still need to run it on the actual dataset we were given to know if the
  results are really correct.
- **Resampling**, the other half of the project, hasn't been started yet
  as far as we know on our side.
