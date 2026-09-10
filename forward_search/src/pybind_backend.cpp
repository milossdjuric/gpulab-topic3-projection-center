#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include "forward_search.hpp"
#include <cmath>
#include <stdexcept>
#include <string>
#include <vector>

namespace py = pybind11;

#ifndef KERNEL_DIR
#error "KERNEL_DIR must be defined by the build (set in backend.py via extra_cflags)"
#endif

static py::dict search(
    py::array_t<float, py::array::c_style | py::array::forcecast> projections,
    double SDD, double SOD, double pixel_size,
    int detector_width, int detector_height,
    double xshift, double alpha, double beta,
    double xshift_step, double alpha_step, double beta_step,
    std::string mode
) {
    if (projections.ndim() != 3)
        throw std::runtime_error("projections must have shape (num_projs, H, W)");

    int num_projs = static_cast<int>(projections.shape(0));
    int H         = static_cast<int>(projections.shape(1));
    int W         = static_cast<int>(projections.shape(2));
    if (H != detector_height || W != detector_width)
        throw std::runtime_error("projections shape does not match detector_height/detector_width");

    const float* data = projections.data();
    std::vector<float> projs(data, data + static_cast<size_t>(num_projs) * H * W);

    CbPara para{};
    para.num_projs       = num_projs;
    para.SDD             = SDD;
    para.SOD             = SOD;
    para.pixel_size      = pixel_size;
    para.detector_width  = detector_width;
    para.detector_height = detector_height;

    SearchArgs args{};
    args.xshift      = xshift      / 1000.0;
    args.alpha       = alpha       / 180.0 * PI;
    args.beta         = beta        / 180.0 * PI;
    args.xshift_step = xshift_step / 1000.0;
    args.alpha_step  = alpha_step  / 180.0 * PI;
    args.beta_step   = beta_step   / 180.0 * PI;

    std::string kernel_path = std::string(KERNEL_DIR) + "/forward_search.cl";
    // No explicit cl::Error translator needed: cl2.hpp/opencl.hpp's cl::Error
    // derives from std::exception here, so pybind11's default translation
    // already surfaces OpenCL failures as Python exceptions with what()'s text.
    CbPose pose = computeCOR(para, projs, args, kernel_path, mode);

    py::dict result;
    result["xshift"]   = pose.xshift;
    result["alpha"]    = pose.alpha;
    result["beta"]     = pose.beta;
    result["MSE"]      = pose.mse;
    result["center_x"] = pose.center_x;
    result["center_y"] = pose.center_y;
    result["kernel_ms"] = pose.kernel_ms;
    return result;
}

PYBIND11_MODULE(forward_search_backend, m) {
    m.doc() = "OpenCL cone-beam CT forward search backend";
    m.def("search", &search,
          py::arg("projections"),
          py::arg("SDD"), py::arg("SOD"), py::arg("pixel_size"),
          py::arg("detector_width"), py::arg("detector_height"),
          py::arg("xshift"), py::arg("alpha"), py::arg("beta"),
          py::arg("xshift_step"), py::arg("alpha_step"), py::arg("beta_step"),
          py::arg("mode") = "buffer",
          "Run the OpenCL forward search and return the best pose as a dict");
}
