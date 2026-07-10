import os
from torch.utils.cpp_extension import load

current_dir = os.path.dirname(os.path.abspath(__file__))
kernel_dir = os.path.join(current_dir, "kernels")

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
    extra_cflags=[f'-DKERNEL_DIR=\\"{kernel_dir}\\"'],
)

__all__ = ["_backend"]
