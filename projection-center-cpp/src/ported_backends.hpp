// C++ ports of the two other implementations in this project, the ones in
// projection-center-python-opencl/ (the teammate's package), so they can be
// called through the same pybind11 interface as this directory's own
// OpenCL code:
//
//   "opencl" -- their OpenCL kernels (center_search_reduce /
//               resample_projections, copied verbatim into
//               kernels/python_opencl_port.cl), driven from C++ host code
//               here instead of from PyOpenCL.
//   "cpu"    -- their NumPy CpuBackend, rewritten as plain C++ loops. Runs
//               entirely on the CPU, no OpenCL.
//
// Both follow projection-center-python-opencl/.../geometry.py and
// backends.py step for step (same parameter grid, same forward geometry,
// same resample rotation matrix), so they give the same answers as the
// Python versions, not just similar ones.
#pragma once
#include <string>
#include "forward_search.hpp"

// The search range and sampling, in the units the Python package's
// SearchConfig uses (mm and degrees), since its parameter grid is built
// from those units directly.
struct PortSearchConfig {
    double xshift_range_mm, alpha_range_deg, beta_range_deg;
    double xshift_step_mm, alpha_step_deg, beta_step_deg;
    int    sample_count = 1000;
    double sample_angle_range_deg = 30.0;
};

// Everything resample needs from a pose: the corrected output geometry and
// the 3x3 rotation matrix (row-major), as geometry.py's
// compute_resample_geometry() returns them.
struct PortResampleGeometry {
    CbPara out;
    float rotation[9];
};

// Forward search. projs is num_projs x detector_height x detector_width.
// kernel_ms is the device time of the search kernel (opencl) or 0 (cpu).
CbPose searchPortedOpenCL(const CbPara& para, const float* projs,
                          const PortSearchConfig& config, const std::string& kernel_path);
CbPose searchPortedCPU(const CbPara& para, const float* projs, const PortSearchConfig& config);

// geometry.py's compute_resample_geometry(). pose is in meters/radians,
// exactly as either search above returns it.
PortResampleGeometry portedResampleGeometry(const CbPara& para, const CbPose& pose, int downsample_factor);

// Resample every projection into out (num_projs x out.detector_height x
// out.detector_width floats, sizes from portedResampleGeometry()).
void resamplePortedOpenCL(const CbPara& para, const float* projs, const PortResampleGeometry& geom,
                          int batch_size, const std::string& kernel_path, float* out);
void resamplePortedCPU(const CbPara& para, const float* projs, const PortResampleGeometry& geom, float* out);
