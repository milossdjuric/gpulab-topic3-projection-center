#define CL_HPP_ENABLE_EXCEPTIONS
#define CL_HPP_MINIMUM_OPENCL_VERSION 120
#define CL_HPP_TARGET_OPENCL_VERSION 300
#include <CL/opencl.hpp>
#include "forward_search.hpp"
#include "opencl_utils.hpp"
#include <algorithm>
#include <cmath>
#include <iostream>
#include <vector>

// Matches the Python reference: sum all projections into one sinogram, then
// normalize to [0,1] so the MSE metric is scale-invariant across datasets.
// One-time CPU cost, not a bottleneck vs. the 35k-combo search below.
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
    bool use_buffer = (mode == "buffer");
    if (!use_buffer && mode != "image")
        throw std::runtime_error("Unknown --mode '" + mode + "' (expected image|buffer)");

    // --- OpenCL setup ---
    cl::Device dev = pickGPU();
    std::cerr << "Device: " << dev.getInfo<CL_DEVICE_NAME>() << "\n";
    cl::Context ctx({dev});
    cl::CommandQueue queue(ctx, dev);

    // --- Sinogram: Image2D (hardware sampler) or plain Buffer (manual bilinear) ---
    int H = para.detector_height, W = para.detector_width;
    std::vector<float> sino = buildSinogram(projs, para.num_projs, H, W);

    cl::Image2D sino_img;
    cl::Buffer  sino_buf;
    if (use_buffer) {
        sino_buf = cl::Buffer(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                              W * H * sizeof(float),
                              const_cast<float*>(sino.data()));
        std::cerr << "Sinogram: " << W << "x" << H << " Buffer uploaded\n";
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

    std::vector<float> nearest_theta(N_THETA);
    for (int i = 0; i < N_THETA; ++i)
        nearest_theta[i] = static_cast<float>(i * RANGE_DEG / N_THETA / 180.0 * M_PI);

    std::cerr << "Grid: " << nx << "x" << na << "x" << nb << " = " << P << " combos\n";

    // --- Upload grid buffers ---
    cl::Buffer buf_xshift(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                          nx * sizeof(float), xshift_arr.data());
    cl::Buffer buf_alpha(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                         na * sizeof(float), alpha_arr.data());
    cl::Buffer buf_beta(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                        nb * sizeof(float), beta_arr.data());
    cl::Buffer buf_theta(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                         N_THETA * sizeof(float), nearest_theta.data());

    cl::Buffer buf_sin_alpha(ctx, CL_MEM_READ_WRITE, na * sizeof(float));
    cl::Buffer buf_cos_alpha(ctx, CL_MEM_READ_WRITE, na * sizeof(float));
    cl::Buffer buf_sin_beta(ctx,  CL_MEM_READ_WRITE, nb * sizeof(float));
    cl::Buffer buf_cos_beta(ctx,  CL_MEM_READ_WRITE, nb * sizeof(float));

    cl::Buffer buf_mse(ctx, CL_MEM_WRITE_ONLY, P * sizeof(float));
    cl::Buffer buf_x0( ctx, CL_MEM_WRITE_ONLY, P * sizeof(float));
    cl::Buffer buf_y0( ctx, CL_MEM_WRITE_ONLY, P * sizeof(float));

    // --- Compile kernels ---
    cl::Program prog = buildProgram(ctx, dev, kernel_path);

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

    // --- Kernel 2: forward search MSE ---
    cl::Kernel k_search(prog, use_buffer ? "forward_search_mse_buffer" : "forward_search_mse");

    // Query hardware-preferred multiple for alignment, but floor it at 64 —
    // with only 1000 theta steps per combo, a smaller WG under-occupies the
    // reduction tree and adds barrier overhead relative to useful work.
    size_t wg_mult = k_search.getWorkGroupInfo<CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE>(dev);
    size_t WG_SIZE = (wg_mult >= 64) ? wg_mult : 64;
    std::cerr << "WG_SIZE: " << WG_SIZE << "\n";

    if (use_buffer) k_search.setArg(0, sino_buf);
    else            k_search.setArg(0, sino_img);
    k_search.setArg(1,  buf_xshift);
    k_search.setArg(2,  buf_sin_alpha);
    k_search.setArg(3,  buf_cos_alpha);
    k_search.setArg(4,  buf_sin_beta);
    k_search.setArg(5,  buf_cos_beta);
    k_search.setArg(6,  buf_theta);
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

    queue.enqueueNDRangeKernel(k_search,
                               cl::NullRange,
                               cl::NDRange(P, WG_SIZE),
                               cl::NDRange(1, WG_SIZE));
    queue.finish();
    std::cerr << "Forward search done\n";

    // --- Read back results ---
    std::vector<float> h_mse(P), h_x0(P), h_y0(P);
    queue.enqueueReadBuffer(buf_mse, CL_TRUE, 0, P * sizeof(float), h_mse.data());
    queue.enqueueReadBuffer(buf_x0,  CL_TRUE, 0, P * sizeof(float), h_x0.data());
    queue.enqueueReadBuffer(buf_y0,  CL_TRUE, 0, P * sizeof(float), h_y0.data());

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

    return pose;
}
