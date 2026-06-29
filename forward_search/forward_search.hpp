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
};

CbPose computeCOR(const CbPara& para,
                  const std::vector<float>& projs,
                  const SearchArgs& args,
                  const std::string& kernel_path);
