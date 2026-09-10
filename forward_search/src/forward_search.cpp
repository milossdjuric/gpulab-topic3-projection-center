#define CL_HPP_ENABLE_EXCEPTIONS
#define CL_HPP_MINIMUM_OPENCL_VERSION 120
#define CL_HPP_TARGET_OPENCL_VERSION 300
#include <CL/opencl.hpp>
#include "forward_search.hpp"
#include "opencl_utils.hpp"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <iostream>
#include <vector>

// Stage timing: prints elapsed ms since computeCOR() started at each
// existing stderr checkpoint. Added during the investigation below and kept
// on since it's cheap and directly useful for the report's performance
// discussion -- run with `--mode buffer` and read the deltas between lines.
static std::chrono::steady_clock::time_point g_t0;
static void logStage(const char* label) {
    auto now = std::chrono::steady_clock::now();
    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(now - g_t0).count();
    std::cerr << "  [+" << ms << "ms] " << label << "\n";
}

// Matches the Python reference: sum all projections into one sinogram, then
// normalize to [0,1] so the MSE metric is scale-invariant across datasets.
//
// CORRECTION (2026-08-25): this was previously commented "one-time CPU cost,
// not a bottleneck vs. the 35k-combo search below" -- that was wrong, and
// the wrongness had real consequences. Stage timing (added while
// investigating why pyopencl_backend's search was ~5x faster than this
// binary's, same GPU, same algorithm) showed the actual search kernel takes
// ~48ms, while this loop alone took ~11.4s -- ~240x more expensive than the
// GPU work it was assumed not to bottleneck. Root cause: meson.build never
// set a buildtype, so this whole binary was being compiled at -O0 the entire
// time (fixed: buildtype=release). At -O0 a naive double-accumulation loop
// over 180 x 1024x1024 elements doesn't get vectorized; at -O3 it does, and
// this drops to ~0.3-0.8s. See docs/ARCHITECTURE.md for the full writeup.
static std::vector<float> buildSinogram(const std::vector<float>& projs,
                                        int num_projs, int H, int W) {
    // Accumulate in double: naive float32 summation over 180 projections
    // drifts enough from numpy's pairwise-summed .sum(axis=0) to tip the
    // argmin in flat regions of the MSE search landscape.
    std::vector<double> sino_d(H * W, 0.0);
    for (int p = 0; p < num_projs; ++p)
        for (int i = 0; i < H * W; ++i)
            sino_d[i] += projs[p * H * W + i];
    double mn = *std::min_element(sino_d.begin(), sino_d.end());
    double mx = *std::max_element(sino_d.begin(), sino_d.end());
    std::vector<float> sino(H * W);
    for (int i = 0; i < H * W; ++i)
        sino[i] = static_cast<float>((sino_d[i] - mn) / (mx - mn));
    return sino;
}

static std::vector<float> makeRange(double lo, double hi, double step) {
    std::vector<float> v;
    for (double x = lo; x <= hi + step * 0.5; x += step)
        v.push_back(static_cast<float>(x));
    return v;
}

CbPose computeCOR(const CbPara& para,
                  const std::vector<float>& projs,
                  const SearchArgs& args,
                  const std::string& kernel_path,
                  const std::string& mode) {
    g_t0 = std::chrono::steady_clock::now();
    bool use_buffer = (mode == "buffer");
    if (!use_buffer && mode != "image")
        throw std::runtime_error("Unknown --mode '" + mode + "' (expected image|buffer)");

    // --- OpenCL setup ---
    cl::Device dev = pickGPU();
    std::cerr << "Device: " << dev.getInfo<CL_DEVICE_NAME>() << "\n";
    logStage("device picked");
    cl::Context ctx({dev});
    // CL_QUEUE_PROFILING_ENABLE: lets us read device-side kernel start/end
    // timestamps (see the enqueueNDRangeKernel(k_search, ...) call below),
    // instead of only the host-side wall-clock stage timing logStage()
    // already provides.
    cl::CommandQueue queue(ctx, dev, CL_QUEUE_PROFILING_ENABLE);
    logStage("context+queue created");

    // --- Sinogram: Image2D (hardware sampler) or plain Buffer (manual bilinear) ---
    int H = para.detector_height, W = para.detector_width;
    std::vector<float> sino = buildSinogram(projs, para.num_projs, H, W);
    logStage("sinogram built (CPU)");

    cl::Image2D sino_img;
    cl::Buffer  sino_buf;
    if (use_buffer) {
        sino_buf = cl::Buffer(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                              W * H * sizeof(float),
                              const_cast<float*>(sino.data()));
        std::cerr << "Sinogram: " << W << "x" << H << " Buffer uploaded\n";
        logStage("sinogram uploaded");
    } else {
        cl::ImageFormat fmt(CL_R, CL_FLOAT);
        sino_img = cl::Image2D(ctx,
                               CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                               fmt, W, H, 0,
                               const_cast<float*>(sino.data()));
        std::cerr << "Sinogram: " << W << "x" << H << " Image2D uploaded\n";
    }

    // --- Parameter grid ---
    const int    N_THETA   = 1000;
    const double RANGE_DEG = 30.0;

    std::vector<float> xshift_arr = makeRange(-args.xshift, args.xshift, args.xshift_step);
    std::vector<float> alpha_arr  = makeRange(-args.alpha,  args.alpha,  args.alpha_step);
    std::vector<float> beta_arr   = makeRange(-args.beta,   args.beta,   args.beta_step);

    int nx = (int)xshift_arr.size();
    int na = (int)alpha_arr.size();
    int nb = (int)beta_arr.size();
    int P  = nx * na * nb;

    // OPTIMIZATION (2026-09-06): the kernel never actually needs the raw
    // dtheta angle -- it only ever computes tan(theta_0 +/- dtheta), and
    // theta_0 itself is only ever used the same way (see forward_search.cl).
    // So instead of shipping angles and calling tan() ~71M times inside the
    // per-combo hot loop (2 * 1000 samples * 35,721 combos), precompute
    // tan(dtheta) once here on the host (1000 calls total, off the hot path)
    // and have the kernel derive tan(theta_0 +/- dtheta) via the tangent
    // addition/subtraction formula from tan(theta_0) alone -- which itself
    // is just x_p/z_p (tan(atan(x)) == x), eliminating the atan() call too.
    // See docs/ALGORITHMS.md section 1.1 step 4 and docs/ARCHITECTURE.md
    // section 4.3 for the analysis this implements.
    std::vector<float> tan_dtheta(N_THETA);
    for (int i = 0; i < N_THETA; ++i) {
        double dtheta = i * RANGE_DEG / N_THETA / 180.0 * PI;
        tan_dtheta[i] = static_cast<float>(std::tan(dtheta));
    }

    std::cerr << "Grid: " << nx << "x" << na << "x" << nb << " = " << P << " combos\n";

    // --- Upload grid buffers ---
    cl::Buffer buf_xshift(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                          nx * sizeof(float), xshift_arr.data());
    cl::Buffer buf_alpha(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                         na * sizeof(float), alpha_arr.data());
    cl::Buffer buf_beta(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                        nb * sizeof(float), beta_arr.data());
    cl::Buffer buf_tan_dtheta(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                             N_THETA * sizeof(float), tan_dtheta.data());

    cl::Buffer buf_sin_alpha(ctx, CL_MEM_READ_WRITE, na * sizeof(float));
    cl::Buffer buf_cos_alpha(ctx, CL_MEM_READ_WRITE, na * sizeof(float));
    cl::Buffer buf_sin_beta(ctx,  CL_MEM_READ_WRITE, nb * sizeof(float));
    cl::Buffer buf_cos_beta(ctx,  CL_MEM_READ_WRITE, nb * sizeof(float));

    cl::Buffer buf_mse(ctx, CL_MEM_WRITE_ONLY, P * sizeof(float));
    cl::Buffer buf_x0( ctx, CL_MEM_WRITE_ONLY, P * sizeof(float));
    cl::Buffer buf_y0( ctx, CL_MEM_WRITE_ONLY, P * sizeof(float));

    // --- Compile kernels ---
    cl::Program prog = buildProgram(ctx, dev, kernel_path);
    logStage("program built (JIT compile)");

    // --- Kernel 1: trig precomputation ---
    cl::Kernel k_trig(prog, "precompute_trig");
    k_trig.setArg(0, buf_alpha);
    k_trig.setArg(1, buf_sin_alpha);
    k_trig.setArg(2, buf_cos_alpha);
    k_trig.setArg(3, na);
    k_trig.setArg(4, buf_beta);
    k_trig.setArg(5, buf_sin_beta);
    k_trig.setArg(6, buf_cos_beta);
    k_trig.setArg(7, nb);

    queue.enqueueNDRangeKernel(k_trig, cl::NullRange,
                               cl::NDRange(std::max(na, nb)), cl::NullRange);
    queue.finish();
    std::cerr << "Trig precomputation done\n";
    logStage("trig kernel done");

    // --- Kernel 2: forward search MSE ---
    cl::Kernel k_search(prog, use_buffer ? "forward_search_mse_buffer" : "forward_search_mse");

    // Size the work-group off the device's actual max work-group size (like
    // pyopencl_backend's OpenCLBackend.search() does: min(256, max_wg),
    // rounded down to a power of two for the kernel's binary-tree reduction),
    // not off CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE -- that value is a
    // SIMD-width alignment hint (often 8/16/32 on this hardware), not a
    // recommended work-group *size*. Treating it as the size (the previous
    // logic here: floor the multiple at 64) left 4x GPU parallelism per
    // combo unused on hardware whose real max work-group size is 256 --
    // confirmed via profiling: pyopencl_backend's own kernel, running the
    // same algorithm at local size 256 on this same GPU, was ~5x faster than
    // this binary at local size 64.
    size_t max_wg = dev.getInfo<CL_DEVICE_MAX_WORK_GROUP_SIZE>();
    size_t WG_SIZE = std::min<size_t>(256, max_wg);
    if (WG_SIZE & (WG_SIZE - 1)) {  // round down to the nearest power of two
        size_t p = 1;
        while (p * 2 <= WG_SIZE) p *= 2;
        WG_SIZE = p;
    }
    if (WG_SIZE < 64) WG_SIZE = 64;  // floor: still keep some occupancy on very constrained devices
    std::cerr << "WG_SIZE: " << WG_SIZE << "\n";

    if (use_buffer) k_search.setArg(0, sino_buf);
    else            k_search.setArg(0, sino_img);
    k_search.setArg(1,  buf_xshift);
    k_search.setArg(2,  buf_sin_alpha);
    k_search.setArg(3,  buf_cos_alpha);
    k_search.setArg(4,  buf_sin_beta);
    k_search.setArg(5,  buf_cos_beta);
    k_search.setArg(6,  buf_tan_dtheta);
    k_search.setArg(7,  buf_mse);
    k_search.setArg(8,  buf_x0);
    k_search.setArg(9,  buf_y0);
    k_search.setArg(10, na);
    k_search.setArg(11, nb);
    k_search.setArg(12, N_THETA);
    k_search.setArg(13, (float)para.SDD);
    k_search.setArg(14, (float)para.SOD);
    k_search.setArg(15, (float)para.pixel_size);
    k_search.setArg(16, para.detector_width);
    k_search.setArg(17, para.detector_height);
    k_search.setArg(18, cl::Local(WG_SIZE * sizeof(float)));
    k_search.setArg(19, cl::Local(WG_SIZE * sizeof(int)));

    cl::Event search_event;
    queue.enqueueNDRangeKernel(k_search,
                               cl::NullRange,
                               cl::NDRange(P, WG_SIZE),
                               cl::NDRange(1, WG_SIZE),
                               nullptr, &search_event);
    queue.finish();
    std::cerr << "Forward search done\n";
    logStage("search kernel done");

    // Device-side kernel time (see CbPose::kernel_ms) -- deliberately not a
    // host-side elapsed-since-enqueue measurement, since that would include
    // driver dispatch/queueing latency on top of actual device execution.
    cl_ulong t_start = search_event.getProfilingInfo<CL_PROFILING_COMMAND_START>();
    cl_ulong t_end   = search_event.getProfilingInfo<CL_PROFILING_COMMAND_END>();
    double kernel_ms = (t_end - t_start) / 1.0e6;
    std::cerr << "Search kernel device time: " << kernel_ms << " ms\n";

    // --- Read back results ---
    std::vector<float> h_mse(P), h_x0(P), h_y0(P);
    queue.enqueueReadBuffer(buf_mse, CL_TRUE, 0, P * sizeof(float), h_mse.data());
    queue.enqueueReadBuffer(buf_x0,  CL_TRUE, 0, P * sizeof(float), h_x0.data());
    queue.enqueueReadBuffer(buf_y0,  CL_TRUE, 0, P * sizeof(float), h_y0.data());
    logStage("results read back");

    // --- Find best combo ---
    // combo_id was encoded row-major as (i_xshift * na + i_alpha) * nb + i_beta
    // in the kernel's ND-range decode; invert that here to recover indices.
    int best     = (int)(std::min_element(h_mse.begin(), h_mse.end()) - h_mse.begin());
    int i_xshift = best / (na * nb);
    int i_alpha  = (best % (na * nb)) / nb;
    int i_beta   =  best % nb;

    CbPose pose;
    pose.mse      = h_mse[best];
    pose.xshift   = xshift_arr[i_xshift];
    pose.alpha    = alpha_arr[i_alpha];
    pose.beta     = beta_arr[i_beta];
    pose.center_x = h_x0[best];
    pose.center_y = h_y0[best];
    pose.kernel_ms = kernel_ms;

    return pose;
}
