#pragma once
#include <string>
#include <vector>
#include "forward_search.hpp"

// The pose found by forward search (either implementation -- this repo's C++
// one or pyopencl_backend's), read from a pose JSON. MSE isn't needed for
// resampling, only the geometry.
struct ResamplePose {
    double center_x, center_y;
    double xshift, alpha, beta;
};

struct ResampleOutput {
    CbPara para;               // corrected geometry: SDD/SOD/pixel_size/detector dims updated,
                                // num_projs/voxel_size/volumen_*/angles carried through unchanged
    std::vector<float> projs;  // resampled projections, num_projs x detector_height x detector_width
};

ResampleOutput computeResample(const CbPara& para,
                               const std::vector<float>& projs,
                               const ResamplePose& pose,
                               int downsample_factor,
                               const std::string& kernel_path,
                               int batch_size = 16);
