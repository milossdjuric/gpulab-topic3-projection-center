"""Read/write time accounting, so every run can print the same timing line
as the other implementations in this project (projection-center-cpp's
binaries and run_pybind.py):

    timing (excluding imports): read R ms | compute C ms | write W ms | I/O R+W ms | compute+I/O C+R+W ms

hdf5_io.py adds the time of every HDF5/pose-file read and write to IO;
the cpp backend adds the read/write time its C++ binaries report. cli.py
times the whole command and treats everything that isn't read or write as
compute (device setup, kernel compile, sinogram, search, resample).
"""
from __future__ import annotations

import functools
import re
import time
from contextlib import contextmanager

# The line projection-center-cpp's binaries print (see its src/stage_timing.hpp).
_NATIVE_TIMING = re.compile(r"timing \(excluding imports\): read (\d+) ms \| compute (\d+) ms \| write (\d+) ms")


class _IOClock:
    def __init__(self) -> None:
        self.read = 0.0   # seconds
        self.write = 0.0  # seconds
        self._depth = 0   # only the outermost timed block counts, so a read
                          # helper calling another one isn't counted twice

    def reset(self) -> None:
        self.read = 0.0
        self.write = 0.0

    @contextmanager
    def _timed(self, kind: str):
        self._depth += 1
        t0 = time.perf_counter()
        try:
            yield
        finally:
            self._depth -= 1
            if self._depth == 0:
                setattr(self, kind, getattr(self, kind) + time.perf_counter() - t0)

    def reading(self):
        return self._timed("read")

    def writing(self):
        return self._timed("write")

    def timed_read(self, fn):
        """Decorator: the whole function counts as read time."""
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            with self.reading():
                return fn(*args, **kwargs)
        return wrapper

    def timed_write(self, fn):
        """Decorator: the whole function counts as write time."""
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            with self.writing():
                return fn(*args, **kwargs)
        return wrapper

    def add_native(self, output: str) -> None:
        """Adds the read/write time from a projection-center-cpp binary's
        timing line (found anywhere in its captured output)."""
        for match in _NATIVE_TIMING.finditer(output):
            self.read += int(match.group(1)) / 1000.0
            self.write += int(match.group(3)) / 1000.0


IO = _IOClock()


def format_timing_line(read_ms: float, compute_ms: float, write_ms: float) -> str:
    return (f"timing (excluding imports): read {read_ms:.0f} ms | compute {compute_ms:.0f} ms"
            f" | write {write_ms:.0f} ms | I/O {read_ms + write_ms:.0f} ms"
            f" | compute+I/O {read_ms + compute_ms + write_ms:.0f} ms")
