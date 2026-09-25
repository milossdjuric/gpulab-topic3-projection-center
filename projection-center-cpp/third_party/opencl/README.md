# Bundled OpenCL headers

Unmodified copies of the official Khronos OpenCL headers, release
**v2025.07.22** of both repositories:

- `CL/opencl.hpp` from https://github.com/KhronosGroup/OpenCL-CLHPP
- `CL/*.h` (the C headers `opencl.hpp` includes) from https://github.com/KhronosGroup/OpenCL-Headers

Both are Apache-2.0 licensed; the license texts are next to this file.

Why they are bundled: the build (pybind11 via `backend.py`, and both
`meson.build` files) must not depend on which version of these headers the
machine has installed. Older system copies of `CL/opencl.hpp` (for example
Debian 12's `opencl-clhpp-headers`, release 2023.02) fail to compile this
project with `'getContextPlatformVersion' is not a member of 'cl::detail'`,
and some systems only ship the older `cl2.hpp` or no C++ header at all.
These folders are searched before the system's, so every machine compiles
against exactly this tested version. Only the OpenCL library itself
(`libOpenCL.so`, e.g. Ubuntu/Debian `ocl-icd-opencl-dev`) and a GPU driver
come from the system.

To update: download both files sets from the same new release tag and
replace them here.
