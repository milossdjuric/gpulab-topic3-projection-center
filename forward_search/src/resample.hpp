// The data types used by the resample side of this project: the pose to
// correct for (ResamplePose) and the corrected result (ResampleOutput).
// computeResample(), at the bottom, is the function that actually does
// the correcting, on the GPU (in resample.cpp).
#pragma once
#include <string>
#include <vector>
#include "forward_search.hpp"

// The pose that search already found, either read from a pose file or
// handed over directly in memory. We only need the geometry here, not the
// error score, since resample just needs to know where the true center is.
struct ResamplePose {
    double center_x, center_y;
    double xshift, alpha, beta;
};

// What computeResample() hands back: the corrected geometry plus the
// corrected pixels themselves.
struct ResampleOutput {
    CbPara para;               // corrected geometry: SDD, SOD, pixel_size and detector size are
                                // updated; everything else is copied through unchanged
    std::vector<float> projs;  // the corrected images, num_projs x detector_height x detector_width
};

// Runs resample: applies pose to every image in projs so the result looks
// like it was captured by a centered detector. downsample_factor can
// shrink the output size (1 means keep it the same). kernel_path is which
// .cl file to compile. batch_size is how many images get sent to the GPU
// at once.
ResampleOutput computeResample(const CbPara& para,
                               const std::vector<float>& projs,
                               const ResamplePose& pose,
                               int downsample_factor,
                               const std::string& kernel_path,
                               int batch_size = 16);
