import glob
import os
import shutil
import stat
import subprocess
import sys
import sysconfig
import tempfile

from torch.utils.cpp_extension import load

current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.join(current_dir, "src")
kernel_dir = os.path.join(current_dir, "kernels")
# Bundled Khronos OpenCL headers (CL/opencl.hpp + C headers), searched
# before the system's: older system copies of opencl.hpp don't compile with
# this project's settings. See third_party/opencl/README.md.
opencl_include_dir = os.path.join(current_dir, "third_party", "opencl")


def _opencl_link_flags():
    """Linker flags for the OpenCL library (the ICD loader).

    A plain -lOpenCL needs the unversioned libOpenCL.so, which only the
    development package installs (e.g. ocl-icd-opencl-dev). Many GPU
    machines only have the runtime libOpenCL.so.1 that comes with the
    driver, or have the library outside the linker's default path (e.g.
    CUDA's lib64), and -lOpenCL then fails with "cannot find -lOpenCL".
    So find the library file and link it directly, with an rpath to its
    folder so it is also found at import time. OPENCL_LIBRARY=/path/to/lib
    overrides the search.
    """
    override = os.environ.get("OPENCL_LIBRARY")
    candidates = [override] if override else []
    for ldconfig in ("/sbin/ldconfig", "/usr/sbin/ldconfig", "ldconfig"):
        try:
            out = subprocess.run([ldconfig, "-p"], capture_output=True, text=True).stdout
        except OSError:
            continue
        for line in out.splitlines():
            name, _, path = line.partition("=>")
            # skip 32-bit copies on a 64-bit Python
            if "libOpenCL.so" in name and (sys.maxsize <= 2**32 or "64" in name):
                candidates.append(path.strip())
        break
    folders = [os.path.join(os.environ.get("CUDA_HOME", "/usr/local/cuda"), "lib64"),
               "/usr/local/lib", "/opt/rocm/lib"]
    if not candidates:
        # No ldconfig cache (some containers): look in the system library
        # folders directly. /usr/lib itself is left out, since on some
        # distros it holds the 32-bit libraries.
        multiarch = sysconfig.get_config_var("MULTIARCH")
        if multiarch:
            folders += [os.path.join("/usr/lib", multiarch), os.path.join("/lib", multiarch)]
        folders += ["/usr/lib64", "/lib64"]
    for folder in folders:
        candidates += sorted(glob.glob(os.path.join(folder, "libOpenCL.so*")))
    candidates = [c for c in candidates if c and os.path.isfile(c)]
    if not candidates:
        return ["-lOpenCL"]  # nothing found: let the linker report it
    # prefer the unversioned development symlink, then the shortest name
    # (libOpenCL.so.1 over libOpenCL.so.1.0.0); the override always wins
    if not override:
        candidates.sort(key=lambda c: (not c.endswith(".so"), len(os.path.basename(c))))
    lib = candidates[0]
    return [lib, "-Wl,-rpath," + os.path.dirname(os.path.realpath(lib))]


# torch looks for ninja on PATH. When a venv's python is run directly,
# without activating the venv, a ninja pip-installed into that venv is not
# on PATH, and load() fails with "Ninja is required to load C++
# extensions". In that case (and only then), put this interpreter's own
# bin folder first, which is what activating the venv would do.
_python_bin = os.path.dirname(sys.executable)
if shutil.which("ninja") is None and shutil.which("ninja", path=_python_bin):
    os.environ["PATH"] = _python_bin + os.pathsep + os.environ.get("PATH", "")


def private_tmp_dir(name):
    """A folder under the system temp dir that only this user can write:
    <tmp>/<name>_<uid>, created with mode 0700. The temp dir is shared by
    every user, so a folder with a predictable name could have been created
    (or symlinked) by someone else first, letting them plant files this
    process then loads, or redirect what it writes. If the folder isn't a
    real directory owned by us and closed to others, a fresh private one
    (tempfile.mkdtemp) is used instead."""
    uid = os.getuid() if hasattr(os, "getuid") else None
    path = os.path.join(tempfile.gettempdir(), f"{name}_{uid if uid is not None else 'user'}")
    try:
        os.makedirs(path, mode=0o700, exist_ok=True)
        st = os.lstat(path)
        if stat.S_ISDIR(st.st_mode) and (uid is None or st.st_uid == uid) and not st.st_mode & 0o022:
            return path
    except OSError:
        pass
    return tempfile.mkdtemp(prefix=name + "_")


def _writable_dir(path):
    """True if path exists and can be written to, or can be created."""
    while not os.path.exists(path):
        parent = os.path.dirname(path)
        if parent == path:
            return False
        path = parent
    return os.access(path, os.W_OK)


# torch compiles into ~/.cache/torch_extensions (or $XDG_CACHE_HOME). If
# that isn't writable (no home folder, a read-only or quota-full home), use
# a folder in the system temp dir instead. A TORCH_EXTENSIONS_DIR set by
# the user always wins.
if not os.environ.get("TORCH_EXTENSIONS_DIR"):
    cache_home = os.environ.get("XDG_CACHE_HOME") or os.path.join(os.path.expanduser("~"), ".cache")
    if not _writable_dir(os.path.join(cache_home, "torch_extensions")):
        fallback = private_tmp_dir("torch_extensions")
        print(f"[backend] {cache_home} is not writable, building the module in {fallback} "
              f"(set TORCH_EXTENSIONS_DIR to choose another folder)", file=sys.stderr)
        os.environ["TORCH_EXTENSIONS_DIR"] = fallback

_backend = load(
    name="forward_search_backend",
    sources=[
        os.path.join(src_dir, "pybind_backend.cpp"),
        os.path.join(src_dir, "forward_search.cpp"),
        os.path.join(src_dir, "resample.cpp"),
        os.path.join(src_dir, "ported_backends.cpp"),
    ],
    extra_include_paths=[opencl_include_dir, src_dir],
    extra_ldflags=_opencl_link_flags(),
    # -O3: torch.utils.cpp_extension.load() sets no optimization flag by
    # default (confirmed via the generated build.ninja -- cflags had no -O*
    # at all, i.e. the compiler's own default of -O0). forward_search.cpp's
    # buildSinogram() is a plain double-accumulation loop that depends on
    # auto-vectorization to be fast; at -O0 it took ~11.4s on the real
    # dataset, ~240x the actual GPU kernel's ~48ms. See forward_search.cpp's
    # comment on buildSinogram() (found via the CLI/meson build, which had
    # the same bug, fixed there by setting buildtype=release in meson.build).
    extra_cflags=["-O3"],
)
# The kernel folder is handed over at run time rather than compiled in, so
# the build works for any project path, including one with spaces.
_backend._set_kernel_dir(kernel_dir)

__all__ = ["_backend"]
