// The actual search code. Builds a combined image from every projection,
// works out the trig values every candidate needs ahead of time, then
// scores every candidate pose on the GPU in one go and reads back the
// winner. computeCOR(), at the bottom, is the function every CLI binary
// calls; everything above it is a helper only this file uses.
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

// Prints how many milliseconds have passed since the search started, so
// you can see where the time actually goes (loading, building the
// sinogram, compiling the kernel, running it).
static std::chrono::steady_clock::time_point g_t0;
static void logStage(const char* label) {
    auto now = std::chrono::steady_clock::now();
    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(now - g_t0).count();
    std::cerr << "  [+" << ms << "ms] " << label << "\n";
}

// Adds every projection together into one image, then scales it to 0 to 1
// so the error score behaves the same no matter how bright or dark a
// dataset's raw pixels are. A plain CPU loop, same as the Python
// reference.
//
// This step used to be wrongly assumed to be too small to matter next to
// the search kernel. It wasn't: the search kernel only takes about 48ms,
// but this loop alone was taking about 11.4s, over 200x slower than it
// needed to be. The cause was a missing compiler flag (the build was
// compiling without optimizations turned on); once fixed, this drops to
// well under a second.
std::vector<float> buildSinogram(const float* projs, int num_projs, int H, int W) {
    // Adds in double precision, not float, since plain float addition
    // over 180 images drifts just enough to occasionally pick a different
    // winner in a close call.
    std::vector<double> sino_d(H * W, 0.0);
    for (int p = 0; p < num_projs; ++p)
        for (int i = 0; i < H * W; ++i)
            sino_d[i] += projs[static_cast<size_t>(p) * H * W + i];
    double mn = *std::min_element(sino_d.begin(), sino_d.end());
    double mx = *std::max_element(sino_d.begin(), sino_d.end());
    std::vector<float> sino(H * W);
    for (int i = 0; i < H * W; ++i)
        sino[i] = static_cast<float>((sino_d[i] - mn) / (mx - mn));
    return sino;
}

// Builds a list of values from lo to hi, step apart. Used once each for
// xshift, alpha, and beta, to build the grid of candidates to try.
static std::vector<float> makeRange(double lo, double hi, double step) {
    std::vector<float> v;
    for (double x = lo; x <= hi + step * 0.5; x += step)
        v.push_back(static_cast<float>(x));
    return v;
}

// Runs search, called by every CLI binary that does it. Picks a GPU,
// builds the combined sinogram image, builds the grid of candidates from
// args, runs two kernels (one to precompute trig values, one to score
// every candidate), and returns whichever candidate had the lowest error.
CbPose computeCOR(const CbPara& para,
                  const std::vector<float>& projs,
                  const SearchArgs& args,
                  const std::string& kernel_path,
                  const std::string& mode) {
    return computeCOR(para, projs.data(), args, kernel_path, mode);
}

CbPose computeCOR(const CbPara& para,
                  const float* projs,
                  const SearchArgs& args,
                  const std::string& kernel_path,
                  const std::string& mode) {
    g_t0 = std::chrono::steady_clock::now();
    bool use_buffer = (mode == "buffer");
    if (!use_buffer && mode != "image")
        throw std::runtime_error("Unknown --mode '" + mode + "' (expected image|buffer)");

    // Pick a GPU and set up a place to send it work.
    cl::Device dev = pickGPU();
    std::cerr << "Device: " << dev.getInfo<CL_DEVICE_NAME>() << "\n";
    logStage("device picked");
    cl::Context ctx({dev});
    // Profiling turned on so we can read exactly how long the GPU itself
    // spends on the search kernel later, not just wall clock time.
    cl::CommandQueue queue(ctx, dev, CL_QUEUE_PROFILING_ENABLE);
    logStage("context+queue created");

    // Build and upload the sinogram. "image" mode lets the GPU's own
    // hardware do the bilinear interpolation; "buffer" mode does the same
    // interpolation by hand in the kernel instead. buffer is the default,
    // since image mode crashes on this dev machine's Intel GPU driver (a
    // driver bug, not a problem with the kernel).
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

    // N_THETA is how many angle pairs get checked per candidate;
    // RANGE_DEG is how wide a slice of angles gets checked. Both are
    // fixed values, not command line flags, to match the reference.
    const int    N_THETA   = 1000;
    const double RANGE_DEG = 30.0;

    std::vector<float> xshift_arr = makeRange(-args.xshift, args.xshift, args.xshift_step);
    std::vector<float> alpha_arr  = makeRange(-args.alpha,  args.alpha,  args.alpha_step);
    std::vector<float> beta_arr   = makeRange(-args.beta,   args.beta,   args.beta_step);

    int nx = (int)xshift_arr.size();  // number of xshift candidates
    int na = (int)alpha_arr.size();   // number of alpha candidates
    int nb = (int)beta_arr.size();    // number of beta candidates
    int P  = nx * na * nb;            // total number of candidate poses (nx * na * nb)

    // This precomputes one number (tan of the sampling angle) that would
    // otherwise get recalculated about 71 million times inside the
    // kernel, once per candidate instead. See docs/ALGORITHMS.md section
    // 1.1 step 4 for the full math behind why this is safe to do.
    std::vector<float> tan_dtheta(N_THETA);
    for (int i = 0; i < N_THETA; ++i) {
        double dtheta = i * RANGE_DEG / N_THETA / 180.0 * PI;
        tan_dtheta[i] = static_cast<float>(std::tan(dtheta));
    }

    std::cerr << "Grid: " << nx << "x" << na << "x" << nb << " = " << P << " combos\n";

    // Upload the candidates, and make room for what the kernels will
    // write back: one score plus one center point per candidate.
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

    // Compile the kernels once, not once per candidate.
    cl::Program prog = buildProgram(ctx, dev, kernel_path);
    logStage("program built (JIT compile)");

    // Kernel 1: work out sin/cos for every unique alpha and beta value up
    // front, once each, instead of every candidate redoing it.
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

    // Kernel 2: the actual search. Scores every candidate at once, one
    // GPU work-group per candidate.
    cl::Kernel k_search(prog, use_buffer ? "forward_search_mse_buffer" : "forward_search_mse");

    // How many GPU threads work together on one candidate's score. Uses
    // the GPU's real maximum (up to 256), rounded to a power of two,
    // never below 64.
    size_t max_wg = dev.getInfo<CL_DEVICE_MAX_WORK_GROUP_SIZE>();
    size_t WG_SIZE = std::min<size_t>(256, max_wg);
    if (WG_SIZE & (WG_SIZE - 1)) {  // round down to the nearest power of two
        size_t p = 1;
        while (p * 2 <= WG_SIZE) p *= 2;
        WG_SIZE = p;
    }
    if (WG_SIZE < 64) WG_SIZE = 64;  // floor: still keep some occupancy on very constrained devices
    std::cerr << "WG_SIZE: " << WG_SIZE << "\n";

    // Give the kernel everything it needs: the sinogram, the candidates,
    // the precomputed trig values, where to write results, the dataset's
    // own geometry, and two scratch buffers it uses internally.
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

    // How long the GPU itself took, straight from OpenCL, not just how
    // long we waited for it (which would also include driver overhead).
    cl_ulong t_start = search_event.getProfilingInfo<CL_PROFILING_COMMAND_START>();
    cl_ulong t_end   = search_event.getProfilingInfo<CL_PROFILING_COMMAND_END>();
    double kernel_ms = (t_end - t_start) / 1.0e6;
    std::cerr << "Search kernel device time: " << kernel_ms << " ms\n";

    // Copy each candidate's score and center point back from the GPU.
    std::vector<float> h_mse(P), h_x0(P), h_y0(P);
    queue.enqueueReadBuffer(buf_mse, CL_TRUE, 0, P * sizeof(float), h_mse.data());
    queue.enqueueReadBuffer(buf_x0,  CL_TRUE, 0, P * sizeof(float), h_x0.data());
    queue.enqueueReadBuffer(buf_y0,  CL_TRUE, 0, P * sizeof(float), h_y0.data());
    logStage("results read back");

    // Find whichever candidate had the lowest error, then work out which
    // actual xshift/alpha/beta values that candidate was.
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
