// See ported_backends.hpp. Everything here mirrors a specific function in
// projection-center-python-opencl/src/projection_center_searching/, named
// in the comment above it. Precision follows the Python code too: where
// NumPy works in float64 this does too, where it works in float32 (the
// resample coordinate grid) this uses float.
#define CL_HPP_ENABLE_EXCEPTIONS
#define CL_HPP_MINIMUM_OPENCL_VERSION 120
#define CL_HPP_TARGET_OPENCL_VERSION 300
#include <CL/opencl.hpp>
#include "ported_backends.hpp"
#include "opencl_utils.hpp"
#include <algorithm>
#include <cfloat>
#include <chrono>
#include <cmath>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <vector>

namespace {

// backends.py's _MIN_SIGNAL (and MIN_SIGNAL in the kernel file).
constexpr double MIN_SIGNAL = 0.01;

// np.arange(start, stop, step): ceil((stop - start) / step) values, each
// start + i * step.
std::vector<double> npArange(double start, double stop, double step) {
    long n = static_cast<long>(std::ceil((stop - start) / step));
    std::vector<double> v(std::max(0L, n));
    for (long i = 0; i < n; ++i) v[i] = start + i * step;
    return v;
}

double deg2rad(double deg) { return deg * (PI / 180.0); }

// One row per candidate pose, plus the forward geometry every candidate
// needs, as geometry.py's build_parameter_grid() and
// compute_forward_geometry() produce them.
struct CandidateGrid {
    int nx = 0, na = 0, nb = 0;                // grid size per parameter
    std::vector<double> xshift, alpha, beta;   // one entry per candidate
    std::vector<double> x0, y0, tan_theta0;    // one entry per candidate
};

// Prints how many milliseconds have passed since the stage log started,
// in the same "  [+Xms] label" form forward_search.cpp's logStage() uses,
// so every backend's output reads the same way.
struct StageLog {
    std::chrono::steady_clock::time_point t0 = std::chrono::steady_clock::now();
    void operator()(const char* label) const {
        auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::steady_clock::now() - t0).count();
        std::cerr << "  [+" << ms << "ms] " << label << "\n";
    }
};

// geometry.py's build_parameter_grid() (meshgrid with indexing="ij", so
// xshift varies slowest and beta fastest) followed by
// compute_forward_geometry().
CandidateGrid buildCandidates(const CbPara& para, const PortSearchConfig& c) {
    std::vector<double> xs = npArange(-c.xshift_range_mm / 1000.0,
                                      c.xshift_range_mm / 1000.0 + c.xshift_step_mm / 2000.0,
                                      c.xshift_step_mm / 1000.0);
    std::vector<double> as = npArange(-c.alpha_range_deg, c.alpha_range_deg + c.alpha_step_deg / 2.0, c.alpha_step_deg);
    std::vector<double> bs = npArange(-c.beta_range_deg,  c.beta_range_deg  + c.beta_step_deg  / 2.0, c.beta_step_deg);
    for (double& a : as) a = deg2rad(a);
    for (double& b : bs) b = deg2rad(b);

    CandidateGrid g;
    g.nx = static_cast<int>(xs.size());
    g.na = static_cast<int>(as.size());
    g.nb = static_cast<int>(bs.size());
    size_t P = xs.size() * as.size() * bs.size();
    g.xshift.reserve(P); g.alpha.reserve(P); g.beta.reserve(P);
    for (double x : xs)
        for (double a : as)
            for (double b : bs) {
                g.xshift.push_back(x);
                g.alpha.push_back(a);
                g.beta.push_back(b);
            }

    const double sod = para.SOD, sdd = para.SDD;
    g.x0.resize(P); g.y0.resize(P); g.tan_theta0.resize(P);
    for (size_t i = 0; i < P; ++i) {
        double xshift = g.xshift[i];
        double sin_a = std::sin(g.alpha[i]), cos_a = std::cos(g.alpha[i]);
        double sin_b = std::sin(g.beta[i]),  cos_b = std::cos(g.beta[i]);

        double b0 = sdd * cos_a * sin_b;
        double b1 = sdd * cos_a * xshift * cos_b;

        double det_a = -sin_a * (-xshift * cos_a * sin_b + sod * sin_a) - cos_a * cos_b * sod * cos_a * cos_b;
        double inv_a00 = 0.0, inv_a01 = 0.0, inv_a10 = 0.0, inv_a11 = 0.0;
        if (std::abs(det_a) > 1.0e-12) {
            inv_a00 = (-xshift * cos_a * sin_b + sod * sin_a) / det_a;
            inv_a01 = (-cos_a * cos_b) / det_a;
            inv_a10 = (-sod * cos_a * cos_b) / det_a;
            inv_a11 = (-sin_a) / det_a;
        }
        double x0 = inv_a00 * b0 + inv_a01 * b1;
        double y0 = inv_a10 * b0 + inv_a11 * b1;

        double r00 = cos_a, r10 = sin_a * cos_b, r20 = sin_a * sin_b;
        double r02 = 0.0,   r12 = -sin_b,        r22 = cos_b;
        double x_p = r00 * x0 + r10 * y0 - r20 * sdd;
        double z_p = r02 * x0 + r12 * y0 - r22 * sdd;

        g.x0[i] = x0;
        g.y0[i] = y0;
        g.tan_theta0[i] = x_p / z_p;
    }
    return g;
}

// tan() of every sampled angle offset, identical for every candidate --
// backends.py computes this once per search the same way.
std::vector<double> tanDtheta(const PortSearchConfig& c) {
    double step = deg2rad(c.sample_angle_range_deg) / c.sample_count;
    std::vector<double> t(c.sample_count);
    for (int i = 0; i < c.sample_count; ++i) t[i] = std::tan(i * step);
    return t;
}

// Picks the winning candidate (first lowest MSE, like np.argmin) and turns
// it into a pose.
CbPose bestPose(const CandidateGrid& g, const std::vector<float>& mse, double kernel_ms) {
    size_t best = std::min_element(mse.begin(), mse.end()) - mse.begin();
    CbPose pose;
    pose.center_x  = g.x0[best];
    pose.center_y  = g.y0[best];
    pose.xshift    = g.xshift[best];
    pose.alpha     = g.alpha[best];
    pose.beta      = g.beta[best];
    pose.mse       = mse[best];
    pose.kernel_ms = kernel_ms;
    return pose;
}

// backends.py's _valid_mask_numpy() for a single point.
inline bool inBounds(double x, double y, int w, int h) {
    return x >= 0.0 && x < w - 1 && y >= 0.0 && y < h - 1;
}

// backends.py's _bilinear_sample_numpy() for a single point: float64
// math on float32 pixels, result rounded to float32, 0 outside the image.
inline float bilinearCPU(const float* img, int w, int h, double x, double y) {
    if (!inBounds(x, y, w, h)) return 0.0f;
    long ix = static_cast<long>(std::floor(x));
    long iy = static_cast<long>(std::floor(y));
    double dx = x - ix, dy = y - iy;
    double v00 = img[iy * w + ix];
    double v10 = img[iy * w + ix + 1];
    double v01 = img[(iy + 1) * w + ix];
    double v11 = img[(iy + 1) * w + ix + 1];
    return static_cast<float>((1.0 - dx) * ((1.0 - dy) * v00 + dy * v01) + dx * ((1.0 - dy) * v10 + dy * v11));
}

// Same as backends.py's OpenCLBackend: at most 256 work-items per group
// (the kernel's __local scratch arrays are sized 256), rounded down to a
// power of two.
size_t workGroupSize(const cl::Device& dev) {
    size_t wg = std::min<size_t>(256, dev.getInfo<CL_DEVICE_MAX_WORK_GROUP_SIZE>());
    wg = std::max<size_t>(1, wg);
    size_t p = 1;
    while (p * 2 <= wg) p *= 2;
    return p;
}

} // namespace

// backends.py's OpenCLBackend.search(), with pipeline.py's run_search()
// around it (sinogram, grid, forward geometry, argmin).
CbPose searchPortedOpenCL(const CbPara& para, const float* projs,
                          const PortSearchConfig& config, const std::string& kernel_path) {
    StageLog log;
    const int W = para.detector_width, H = para.detector_height;

    cl::Device dev = pickGPU();
    std::cerr << "Device: " << dev.getInfo<CL_DEVICE_NAME>() << "\n";
    log("device picked");
    cl::Context ctx({dev});
    cl::CommandQueue queue(ctx, dev, CL_QUEUE_PROFILING_ENABLE);
    log("context+queue created");

    std::vector<float> sino = buildSinogram(projs, para.num_projs, H, W);
    log("sinogram built (CPU)");
    cl::Buffer sino_buf(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sino.size() * sizeof(float), sino.data());
    std::cerr << "Sinogram: " << W << "x" << H << " Buffer uploaded\n";
    log("sinogram uploaded");

    CandidateGrid g = buildCandidates(para, config);
    const size_t P = g.xshift.size();
    std::cerr << "Grid: " << g.nx << "x" << g.na << "x" << g.nb << " = " << P << " combos\n";

    std::vector<float> alpha(P), beta(P), tan_theta0(P);
    for (size_t i = 0; i < P; ++i) {
        alpha[i] = static_cast<float>(g.alpha[i]);
        beta[i]  = static_cast<float>(g.beta[i]);
        tan_theta0[i] = static_cast<float>(g.tan_theta0[i]);
    }
    std::vector<double> tdd = tanDtheta(config);
    std::vector<float> tan_dtheta(tdd.begin(), tdd.end());

    cl::Program prog = buildProgram(ctx, dev, kernel_path);
    log("program built (JIT compile)");

    auto ro = [&](std::vector<float>& v) {
        return cl::Buffer(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, v.size() * sizeof(float), v.data());
    };
    cl::Buffer alpha_buf = ro(alpha), beta_buf = ro(beta);
    cl::Buffer tan_theta0_buf = ro(tan_theta0), tan_dtheta_buf = ro(tan_dtheta);
    cl::Buffer mse_buf(ctx, CL_MEM_WRITE_ONLY, P * sizeof(float));

    cl::Kernel k(prog, "center_search_reduce");
    k.setArg(0, sino_buf);
    k.setArg(1, alpha_buf);
    k.setArg(2, beta_buf);
    k.setArg(3, tan_theta0_buf);
    k.setArg(4, tan_dtheta_buf);
    k.setArg(5, mse_buf);
    k.setArg(6, static_cast<float>(para.SDD));
    k.setArg(7, W);
    k.setArg(8, H);
    k.setArg(9, static_cast<float>(para.pixel_size));
    k.setArg(10, config.sample_count);

    size_t wg = workGroupSize(dev);
    std::cerr << "WG_SIZE: " << wg << "\n";
    cl::Event ev;
    queue.enqueueNDRangeKernel(k, cl::NullRange, cl::NDRange(wg, P), cl::NDRange(wg, 1), nullptr, &ev);
    queue.finish();
    std::cerr << "Forward search done\n";
    log("search kernel done");
    double kernel_ms = (ev.getProfilingInfo<CL_PROFILING_COMMAND_END>() -
                        ev.getProfilingInfo<CL_PROFILING_COMMAND_START>()) / 1.0e6;
    std::cerr << "Search kernel device time: " << kernel_ms << " ms\n";

    std::vector<float> mse(P);
    queue.enqueueReadBuffer(mse_buf, CL_TRUE, 0, P * sizeof(float), mse.data());
    log("results read back");
    return bestPose(g, mse, kernel_ms);
}

// backends.py's CpuBackend.search(), with pipeline.py's run_search()
// around it. Every candidate, every sample pair, one after another on the
// CPU.
CbPose searchPortedCPU(const CbPara& para, const float* projs, const PortSearchConfig& config) {
    StageLog log;
    std::cerr << "Device: CPU (no OpenCL)\n";
    const int W = para.detector_width, H = para.detector_height;
    std::vector<float> sino = buildSinogram(projs, para.num_projs, H, W);
    log("sinogram built (CPU)");
    CandidateGrid g = buildCandidates(para, config);
    std::vector<double> tan_dtheta = tanDtheta(config);
    const size_t P = g.xshift.size();
    std::cerr << "Grid: " << g.nx << "x" << g.na << "x" << g.nb << " = " << P << " combos\n";
    const double sdd = para.SDD, pixel_size = para.pixel_size;
    const double cx = (W - 1) / 2.0, cy = (H - 1) / 2.0;

    std::vector<float> mse(P);
    for (size_t idx = 0; idx < P; ++idx) {
        double sin_a = std::sin(g.alpha[idx]), cos_a = std::cos(g.alpha[idx]);
        double sin_b = std::sin(g.beta[idx]),  cos_b = std::cos(g.beta[idx]);
        double t0 = g.tan_theta0[idx];

        double sum = 0.0;
        long count = 0;
        for (double td : tan_dtheta) {
            double tan_t  = (t0 + td) / (1.0 - t0 * td);
            double tan_rt = (t0 - td) / (1.0 + t0 * td);
            double temp  = sdd / (tan_t  * sin_a * sin_b + cos_b);
            double rtemp = sdd / (tan_rt * sin_a * sin_b + cos_b);
            double y  = temp  * (-tan_t  * sin_a * cos_b + sin_b);
            double ry = rtemp * (-tan_rt * sin_a * cos_b + sin_b);
            double x  = -temp  * tan_t  * cos_a;
            double rx = -rtemp * tan_rt * cos_a;
            x  = x  / pixel_size + cx;
            rx = rx / pixel_size + cx;
            y  = cy - y  / pixel_size;
            ry = cy - ry / pixel_size;

            if (!inBounds(x, y, W, H) || !inBounds(rx, ry, W, H)) continue;
            float f_t  = bilinearCPU(sino.data(), W, H, x, y);
            float f_rt = bilinearCPU(sino.data(), W, H, rx, ry);
            if (f_t >= MIN_SIGNAL || f_rt >= MIN_SIGNAL) {
                float d = f_t - f_rt;       // float32 subtraction and square, like
                sum += d * d;               // np.square() on the float32 samples
                ++count;
            }
        }
        mse[idx] = count > 0 ? static_cast<float>(sum / count) : std::numeric_limits<float>::infinity();
    }
    std::cerr << "Forward search done\n";
    log("search done (CPU)");
    return bestPose(g, mse, 0.0);
}

// geometry.py's compute_resample_geometry().
PortResampleGeometry portedResampleGeometry(const CbPara& para, const CbPose& pose, int downsample) {
    if (downsample < 1)
        throw std::runtime_error("downsample_factor must be >= 1");
    if (para.detector_width % downsample != 0 || para.detector_height % downsample != 0)
        throw std::runtime_error("downsample_factor must evenly divide detector_width and detector_height");

    auto norm = [](double x, double y, double z) { return std::sqrt(x * x + y * y + z * z); };
    auto cross = [](const double a[3], const double b[3], double out[3]) {
        out[0] = a[1] * b[2] - a[2] * b[1];
        out[1] = a[2] * b[0] - a[0] * b[2];
        out[2] = a[0] * b[1] - a[1] * b[0];
    };

    PortResampleGeometry geo;
    geo.out = para;
    geo.out.pixel_size      = para.pixel_size * downsample;
    geo.out.detector_width  = para.detector_width / downsample;
    geo.out.detector_height = para.detector_height / downsample;

    double x0 = pose.center_x, y0 = pose.center_y;
    double sin_a = std::sin(pose.alpha), cos_a = std::cos(pose.alpha);
    double sin_b = std::sin(pose.beta),  cos_b = std::cos(pose.beta);

    geo.out.SDD = norm(x0, y0, para.SDD);
    double axis[3]   = { -sin_a, cos_a * cos_b, cos_a * sin_b };
    double offset[3] = { pose.xshift, 0.0, -para.SOD };
    double c[3];
    cross(offset, axis, c);
    geo.out.SOD = norm(c[0], c[1], c[2]) / norm(axis[0], axis[1], axis[2]);

    double n = norm(axis[0], axis[1], axis[2]);
    double y_axis[3] = { axis[0] / n, axis[1] / n, axis[2] / n };
    double z_axis[3] = { -x0, -y0, para.SDD };
    n = norm(z_axis[0], z_axis[1], z_axis[2]);
    for (double& v : z_axis) v /= n;
    double x_axis[3];
    cross(y_axis, z_axis, x_axis);
    n = norm(x_axis[0], x_axis[1], x_axis[2]);
    for (double& v : x_axis) v /= n;
    cross(x_axis, y_axis, z_axis);
    n = norm(z_axis[0], z_axis[1], z_axis[2]);
    for (double& v : z_axis) v /= n;

    // np.column_stack([x_axis, y_axis, z_axis]): row r is (x[r], y[r], z[r]).
    for (int r = 0; r < 3; ++r) {
        geo.rotation[r * 3 + 0] = static_cast<float>(x_axis[r]);
        geo.rotation[r * 3 + 1] = static_cast<float>(y_axis[r]);
        geo.rotation[r * 3 + 2] = static_cast<float>(z_axis[r]);
    }
    return geo;
}

// backends.py's OpenCLBackend.resample(): one set of device buffers,
// reused for every batch.
void resamplePortedOpenCL(const CbPara& para, const float* projs, const PortResampleGeometry& geo,
                          int batch_size, const std::string& kernel_path, float* out) {
    if (batch_size < 1) throw std::runtime_error("batch_size must be >= 1");
    const int in_w = para.detector_width, in_h = para.detector_height;
    const int out_w = geo.out.detector_width, out_h = geo.out.detector_height;
    const size_t in_px = static_cast<size_t>(in_w) * in_h, out_px = static_cast<size_t>(out_w) * out_h;

    cl::Device dev = pickGPU();
    std::cerr << "Device: " << dev.getInfo<CL_DEVICE_NAME>() << "\n";
    cl::Context ctx({dev});
    cl::CommandQueue queue(ctx, dev);
    cl::Program prog = buildProgram(ctx, dev, kernel_path);
    cl::Kernel k(prog, "resample_projections");

    cl::Buffer rot_buf(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, 9 * sizeof(float),
                       const_cast<float*>(geo.rotation));
    cl::Buffer in_buf(ctx, CL_MEM_READ_ONLY, batch_size * in_px * sizeof(float));
    cl::Buffer out_buf(ctx, CL_MEM_WRITE_ONLY, batch_size * out_px * sizeof(float));

    for (int start = 0; start < para.num_projs; start += batch_size) {
        int count = std::min(batch_size, para.num_projs - start);
        queue.enqueueWriteBuffer(in_buf, CL_TRUE, 0, count * in_px * sizeof(float), projs + start * in_px);
        k.setArg(0, in_buf);
        k.setArg(1, out_buf);
        k.setArg(2, rot_buf);
        k.setArg(3, count);
        k.setArg(4, in_w);
        k.setArg(5, in_h);
        k.setArg(6, static_cast<float>(para.pixel_size));
        k.setArg(7, static_cast<float>(para.SDD));
        k.setArg(8, out_w);
        k.setArg(9, out_h);
        k.setArg(10, static_cast<float>(geo.out.pixel_size));
        k.setArg(11, static_cast<float>(geo.out.SDD));
        queue.enqueueNDRangeKernel(k, cl::NullRange, cl::NDRange(out_w, out_h, count), cl::NullRange);
        queue.enqueueReadBuffer(out_buf, CL_TRUE, 0, count * out_px * sizeof(float), out + start * out_px);
    }
    std::cerr << "Resampled " << para.num_projs << " projections\n";
}

// backends.py's CpuBackend.resample(). The output pixel -> source pixel
// mapping is the same for every projection, so it is worked out once
// (in float32, like the NumPy grid), then every projection is sampled
// through it.
void resamplePortedCPU(const CbPara& para, const float* projs, const PortResampleGeometry& geo, float* out) {
    std::cerr << "Device: CPU (no OpenCL)\n";
    const int in_w = para.detector_width, in_h = para.detector_height;
    const int out_w = geo.out.detector_width, out_h = geo.out.detector_height;
    const size_t in_px = static_cast<size_t>(in_w) * in_h, out_px = static_cast<size_t>(out_w) * out_h;
    const float* R = geo.rotation;

    // Python floats mixed into float32 NumPy arrays are rounded to float32
    // first (NumPy 2 scalar promotion), so every constant here is too.
    const float out_px_size = static_cast<float>(geo.out.pixel_size);
    const float out_cx = static_cast<float>((out_w - 1) / 2.0);
    const float out_cy = static_cast<float>((out_h - 1) / 2.0);
    const float z = static_cast<float>(-geo.out.SDD);
    const float neg_in_sdd = static_cast<float>(-para.SDD);
    const float in_px_size = static_cast<float>(para.pixel_size);
    const float in_cx = static_cast<float>((in_w - 1) / 2.0);
    const float in_cy = static_cast<float>((in_h - 1) / 2.0);

    std::vector<float> src_x(out_px), src_y(out_px);
    for (int yi = 0; yi < out_h; ++yi) {
        float gy = (out_cy - static_cast<float>(yi)) * out_px_size;
        for (int xi = 0; xi < out_w; ++xi) {
            float gx = (static_cast<float>(xi) - out_cx) * out_px_size;
            float nx = R[0] * gx + R[1] * gy + R[2] * z;
            float ny = R[3] * gx + R[4] * gy + R[5] * z;
            float nz = R[6] * gx + R[7] * gy + R[8] * z;
            size_t i = static_cast<size_t>(yi) * out_w + xi;
            src_x[i] = (nx / nz * neg_in_sdd) / in_px_size + in_cx;
            src_y[i] = -(ny / nz * neg_in_sdd) / in_px_size + in_cy;
        }
    }

    for (int p = 0; p < para.num_projs; ++p) {
        const float* img = projs + p * in_px;
        float* dst = out + p * out_px;
        for (size_t i = 0; i < out_px; ++i)
            dst[i] = bilinearCPU(img, in_w, in_h, src_x[i], src_y[i]);
    }
    std::cerr << "Resampled " << para.num_projs << " projections\n";
}
