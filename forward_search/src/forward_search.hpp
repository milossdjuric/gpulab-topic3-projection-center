#pragma once
#include <string>
#include <vector>

struct CbPara {
    int num_projs;
    double SDD, SOD, pixel_size, voxel_size;
    int detector_width, detector_height;
    int volumen_num_xz, volumen_num_y;
    std::vector<double> angles;
};

struct SearchArgs {
    double xshift, alpha, beta;
    double xshift_step, alpha_step, beta_step;
};

struct CbPose {
    double center_x, center_y;
    double xshift, alpha, beta, mse;
    // Device-side execution time of the main search kernel, in milliseconds
    // (via OpenCL event profiling -- CL_PROFILING_COMMAND_START/END), not a
    // host-side wall-clock stage timestamp. Whole-program wall time is
    // dominated by HDF5 I/O and CPU-side sinogram building, so this is the
    // only honest way to measure a kernel-level optimization's effect.
    double kernel_ms;
};

CbPose computeCOR(const CbPara& para,
                  const std::vector<float>& projs,
                  const SearchArgs& args,
                  const std::string& kernel_path,
                  const std::string& mode = "image");
