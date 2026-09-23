// The one timing line every implementation in this project prints at the
// end of a run, in exactly this format, so runs can be compared (and
// benchmark_pybind.py can read it back from any of them):
//
//   timing (excluding imports): read R ms | compute C ms | write W ms | I/O R+W ms | compute+I/O C+R+W ms
//
// read  = reading the HDF5 input (and pose file, if any)
// write = writing the output files
// compute = everything else the run does (device setup, kernel compile,
//           sinogram, search, resample)
// "excluding imports" matters for the Python implementations, whose module
// imports (torch, pyopencl, ...) happen before any of this; a C++ binary
// has none.
#pragma once
#include <chrono>
#include <cstdio>

inline double msSince(std::chrono::steady_clock::time_point t0) {
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
}

inline void printTimingLine(double read_ms, double compute_ms, double write_ms) {
    std::fprintf(stderr,
                 "timing (excluding imports): read %.0f ms | compute %.0f ms | write %.0f ms"
                 " | I/O %.0f ms | compute+I/O %.0f ms\n",
                 read_ms, compute_ms, write_ms, read_ms + write_ms, read_ms + compute_ms + write_ms);
}
