// Lets Python call search and resample directly, without going through a
// subprocess or a file. Projections come in as a NumPy array; search hands
// the found pose back as a Python dict, resample hands the corrected
// images back as a NumPy array (plus the corrected geometry).
//
// Both functions take a backend= choice, one per implementation in this
// project:
//   "cpp"    -- this directory's own OpenCL kernels (forward_search.cl,
//               resample.cl), the same code every CLI binary runs.
//   "opencl" -- projection-center-python-opencl/'s OpenCL kernels, driven
//               from C++ instead of PyOpenCL (ported_backends.cpp).
//   "cpu"    -- projection-center-python-opencl/'s NumPy CPU backend,
//               rewritten as plain C++ (ported_backends.cpp). No GPU.
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include "forward_search.hpp"
#include "opencl_utils.hpp"  // cl::Error and clErrorName(), for the exception translator
#include "ported_backends.hpp"
#include "resample.hpp"
#include <cmath>
#include <stdexcept>
#include <string>

namespace py = pybind11;

// Folder holding the .cl kernel files. backend.py sets it right after
// loading this module (_set_kernel_dir), instead of compiling the path in:
// a path baked into the build flags breaks the build when it contains a
// space, since the compiler command is split at spaces.
static std::string g_kernel_dir;

static void setKernelDir(std::string dir) { g_kernel_dir = std::move(dir); }

static std::string kernelPath(const char* file) {
    if (g_kernel_dir.empty())
        throw std::runtime_error("kernel folder not set: import the module through backend.py "
                                 "(from backend import _backend)");
    return g_kernel_dir + "/" + file;
}

using FloatArray = py::array_t<float, py::array::c_style | py::array::forcecast>;

static void checkBackend(const std::string& backend) {
    if (backend != "cpp" && backend != "opencl" && backend != "cpu")
        throw std::runtime_error("Unknown backend '" + backend + "' (expected cpp|opencl|cpu)");
}

// Geometry for one dataset. Detector size comes from the array's own
// shape, same as the CLI's loadHDF5() does, so the two can never disagree.
static CbPara makePara(const FloatArray& projections, double SDD, double SOD, double pixel_size) {
    if (projections.ndim() != 3)
        throw std::runtime_error("projections must have shape (num_projs, H, W)");
    CbPara para{};
    para.num_projs       = static_cast<int>(projections.shape(0));
    para.detector_height = static_cast<int>(projections.shape(1));
    para.detector_width  = static_cast<int>(projections.shape(2));
    para.SDD             = SDD;
    para.SOD             = SOD;
    para.pixel_size      = pixel_size;
    return para;
}

// The search function Python calls. Search ranges come in as mm/degrees;
// the pose comes back in meters/radians, ready to pass to resample().
// The projections are read straight from the NumPy array's memory, not
// copied.
static py::dict search(
    FloatArray projections,
    double SDD, double SOD, double pixel_size,
    int detector_width, int detector_height,
    double xshift, double alpha, double beta,
    double xshift_step, double alpha_step, double beta_step,
    std::string mode, std::string backend
) {
    checkBackend(backend);
    // A step of 0 or inf would make the candidate grid endless (a hang,
    // then out of memory); a negative or NaN range gives an empty grid.
    auto positive    = [](double v) { return std::isfinite(v) && v > 0; };
    auto nonnegative = [](double v) { return std::isfinite(v) && v >= 0; };
    if (!(positive(xshift_step) && positive(alpha_step) && positive(beta_step)))
        throw std::invalid_argument("xshift_step, alpha_step and beta_step must be finite and > 0");
    if (!(nonnegative(xshift) && nonnegative(alpha) && nonnegative(beta)))
        throw std::invalid_argument("xshift, alpha and beta ranges must be finite and >= 0");
    // Candidates per parameter, as the grid builders count them; keeps the
    // total inside what the kernels' int indexes and memory can hold.
    double grid = (2 * xshift / xshift_step + 1) * (2 * alpha / alpha_step + 1) * (2 * beta / beta_step + 1);
    if (grid > 1e8)
        throw std::invalid_argument("search grid too large (" + std::to_string(static_cast<long long>(grid))
                                    + " candidates, limit 1e8): use a smaller range or a larger step");
    CbPara para = makePara(projections, SDD, SOD, pixel_size);
    if (para.detector_height != detector_height || para.detector_width != detector_width)
        throw std::runtime_error("projections shape does not match detector_height/detector_width");
    const float* projs = projections.data();

    CbPose pose;
    {
        // The NumPy array stays alive (we hold a reference), so the GIL
        // can be dropped while the long-running C++/OpenCL work happens.
        py::gil_scoped_release release;
        if (backend == "cpp") {
            SearchArgs args{};
            args.xshift      = xshift      / 1000.0;
            args.alpha       = alpha       / 180.0 * PI;
            args.beta        = beta        / 180.0 * PI;
            args.xshift_step = xshift_step / 1000.0;
            args.alpha_step  = alpha_step  / 180.0 * PI;
            args.beta_step   = beta_step   / 180.0 * PI;
            // OpenCL failures (cl::Error) reach Python as a RuntimeError with the
            // error code, via the translator registered in PYBIND11_MODULE below.
            pose = computeCOR(para, projs, args, kernelPath("forward_search.cl"), mode);
        } else {
            PortSearchConfig config{};
            config.xshift_range_mm = xshift;
            config.alpha_range_deg = alpha;
            config.beta_range_deg  = beta;
            config.xshift_step_mm  = xshift_step;
            config.alpha_step_deg  = alpha_step;
            config.beta_step_deg   = beta_step;
            pose = backend == "opencl"
                ? searchPortedOpenCL(para, projs, config, kernelPath("python_opencl_port.cl"))
                : searchPortedCPU(para, projs, config);
        }
    }

    py::dict result;
    result["xshift"]   = pose.xshift;
    result["alpha"]    = pose.alpha;
    result["beta"]     = pose.beta;
    result["MSE"]      = pose.mse;
    result["center_x"] = pose.center_x;
    result["center_y"] = pose.center_y;
    // The cpu backend has no GPU kernel to time.
    if (backend == "cpu") result["kernel_ms"] = py::none();
    else                  result["kernel_ms"] = pose.kernel_ms;
    return result;
}

// The resample function Python calls. Takes the pose exactly as search()
// returns it (meters/radians, center point on the detector) and returns
// the corrected images as a (num_projs, out_H, out_W) array next to the
// corrected geometry. The output array is allocated once up front and the
// C++ code writes straight into it, no intermediate copies.
static py::dict resample(
    FloatArray projections,
    double SDD, double SOD, double pixel_size,
    double xshift, double alpha, double beta,
    double center_x, double center_y,
    int downsample, int batch_size, std::string backend
) {
    checkBackend(backend);
    if (downsample < 1)
        throw std::runtime_error("downsample must be >= 1");
    if (batch_size < 1)
        throw std::runtime_error("batch_size must be >= 1");
    CbPara para = makePara(projections, SDD, SOD, pixel_size);
    const float* projs = projections.data();

    CbPose pose{};
    pose.center_x = center_x;
    pose.center_y = center_y;
    pose.xshift   = xshift;
    pose.alpha    = alpha;
    pose.beta     = beta;

    ResamplePose rpose{ center_x, center_y, xshift, alpha, beta };
    PortResampleGeometry port_geo{};
    CbPara out_para = backend == "cpp"
        ? resampleOutputGeometry(para, rpose, downsample)
        : (port_geo = portedResampleGeometry(para, pose, downsample)).out;

    int out_h = out_para.detector_height;
    int out_w = out_para.detector_width;
    py::array_t<float> corrected({para.num_projs, out_h, out_w});
    float* out = corrected.mutable_data();

    {
        py::gil_scoped_release release;
        if (backend == "cpp")
            computeResampleInto(para, projs, rpose, downsample,
                                kernelPath("resample.cl"), batch_size, out);
        else if (backend == "opencl")
            resamplePortedOpenCL(para, projs, port_geo, batch_size,
                                 kernelPath("python_opencl_port.cl"), out);
        else
            resamplePortedCPU(para, projs, port_geo, out);
    }

    py::dict result;
    result["projections"]     = corrected;
    result["SDD"]             = out_para.SDD;
    result["SOD"]             = out_para.SOD;
    result["pixel_size"]      = out_para.pixel_size;
    result["detector_width"]  = out_w;
    result["detector_height"] = out_h;
    return result;
}

// Registers this module as forward_search_backend, with two functions,
// search() and resample(), so Python can import it and call them like any
// other package.
PYBIND11_MODULE(forward_search_backend, m) {
    m.doc() = "Cone-beam CT forward search + resample backend (cpp / opencl / cpu)";
    // cl::Error's what() is only the name of the OpenCL call that failed
    // ("clEnqueueNDRangeKernel"). Add the error code and its name, so a
    // failure on another machine says why, e.g. "... failed:
    // CL_OUT_OF_RESOURCES (-5)".
    py::register_exception_translator([](std::exception_ptr p) {
        try {
            if (p) std::rethrow_exception(p);
        } catch (const cl::Error& e) {
            PyErr_SetString(PyExc_RuntimeError,
                            ("OpenCL error: " + std::string(e.what()) + " failed: " + clErrorName(e.err())
                             + " (" + std::to_string(e.err()) + ")").c_str());
        }
    });
    m.def("_set_kernel_dir", &setKernelDir, py::arg("path"),
          "Internal: set the folder holding the .cl kernel files (done by backend.py)");
    m.def("search", &search,
          py::arg("projections"),
          py::arg("SDD"), py::arg("SOD"), py::arg("pixel_size"),
          py::arg("detector_width"), py::arg("detector_height"),
          py::arg("xshift"), py::arg("alpha"), py::arg("beta"),
          py::arg("xshift_step"), py::arg("alpha_step"), py::arg("beta_step"),
          py::arg("mode") = "buffer", py::arg("backend") = "cpp",
          "Run the forward search and return the best pose as a dict. "
          "backend: 'cpp' (default), 'opencl' or 'cpu'. mode ('buffer'/'image') applies to 'cpp' only.");
    m.def("resample", &resample,
          py::arg("projections"),
          py::arg("SDD"), py::arg("SOD"), py::arg("pixel_size"),
          py::arg("xshift"), py::arg("alpha"), py::arg("beta"),
          py::arg("center_x"), py::arg("center_y"),
          py::arg("downsample") = 1, py::arg("batch_size") = 16, py::arg("backend") = "cpp",
          "Resample projections with a pose from search(); returns the corrected "
          "projections and geometry as a dict. backend: 'cpp' (default), 'opencl' or 'cpu'.");
}
