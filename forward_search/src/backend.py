import os
from torch.utils.cpp_extension import load

current_dir = os.path.dirname(os.path.abspath(__file__))
kernel_dir = os.path.join(os.path.dirname(current_dir), "kernels")

_backend = load(
    name="forward_search_backend",
    sources=[
        os.path.join(current_dir, "pybind_backend.cpp"),
        os.path.join(current_dir, "forward_search.cpp"),
    ],
    extra_include_paths=[current_dir],
    extra_ldflags=["-lOpenCL"],
    # The quotes must survive shell parsing: ninja runs each build command
    # through `sh -c`, which would otherwise strip an unescaped '"',
    # leaving KERNEL_DIR expanding to a bare (unquoted) path and breaking
    # the string-literal concatenation in pybind_backend.cpp.
    #
    # -O3: torch.utils.cpp_extension.load() sets no optimization flag by
    # default (confirmed via the generated build.ninja -- cflags had no -O*
    # at all, i.e. the compiler's own default of -O0). forward_search.cpp's
    # buildSinogram() is a plain double-accumulation loop that depends on
    # auto-vectorization to be fast; at -O0 it took ~11.4s on the real
    # dataset, ~240x the actual GPU kernel's ~48ms. See forward_search.cpp's
    # comment on buildSinogram() and docs/ARCHITECTURE.md for the writeup
    # (found via the CLI/meson build, which had the same bug, fixed there by
    # setting buildtype=release in meson.build).
    extra_cflags=[f'-DKERNEL_DIR=\\"{kernel_dir}\\"', "-O3"],
)

__all__ = ["_backend"]
