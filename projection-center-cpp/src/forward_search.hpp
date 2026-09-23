// The data types used by the search side of this project: what a dataset
// looks like (CbPara), what range to search (SearchArgs), and the pose
// that gets found (CbPose). computeCOR(), at the bottom, is the function
// that actually runs the search on the GPU (in forward_search.cpp).
#pragma once
#include <string>
#include <vector>

// PI isn't standard C++ everywhere (Linux has it built in, Windows does
// not), so it's just defined once here for everyone to use.
constexpr double PI = 3.14159265358979323846;

// Everything about one dataset, read from its HDF5 file. Not the pose
// we're searching for, just facts about the scan itself: how many images,
// how big they are, and the real-world distances involved.
struct CbPara {
    int num_projs;                          // how many projection images the scan has
    double SDD, SOD, pixel_size, voxel_size; // source-detector distance, source-object distance,
                                             // detector pixel size, reconstruction voxel size
    int detector_width, detector_height;    // detector size in pixels
    int volumen_num_xz, volumen_num_y;      // reconstruction volume size, unchanged throughout
    std::vector<double> angles;             // one rotation angle per projection
};

// How wide a range to search, and how big a step to take between tries,
// for each of xshift/alpha/beta. computeCOR() turns this into the actual
// grid of candidate poses.
struct SearchArgs {
    double xshift, alpha, beta;                  // how far to search on each side of zero
    double xshift_step, alpha_step, beta_step;    // step size between candidates
};

// The best pose search found, how good it was (mse), and how long the GPU
// took to find it (kernel_ms).
struct CbPose {
    double center_x, center_y;
    double xshift, alpha, beta, mse;
    // How long the GPU itself spent on the search kernel, in
    // milliseconds, measured by OpenCL directly. This is not the whole
    // program's runtime; most of that actually goes to loading the file
    // and building the sinogram, not this kernel.
    double kernel_ms;
};

// Runs the search. Builds every candidate pose from args, tries them all
// on the GPU, and returns whichever one had the lowest error. kernel_path
// is which .cl file to compile. mode picks between two ways of reading
// the sinogram on the GPU, see forward_search.cpp for why both exist.
CbPose computeCOR(const CbPara& para,
                  const std::vector<float>& projs,
                  const SearchArgs& args,
                  const std::string& kernel_path,
                  const std::string& mode = "image");

// Same as above, but reads the projections straight from a raw pointer
// (num_projs x detector_height x detector_width floats) instead of a
// vector. The pybind11 interface uses this one so it can hand over the
// NumPy array's own memory without copying ~750MB into a vector first.
CbPose computeCOR(const CbPara& para,
                  const float* projs,
                  const SearchArgs& args,
                  const std::string& kernel_path,
                  const std::string& mode = "image");

// Adds every projection together into one image and scales it to 0..1.
// Shared by every search implementation in this project (see
// forward_search.cpp for the details).
std::vector<float> buildSinogram(const float* projs, int num_projs, int H, int W);
