// CORRECTION (2026-08-26): a sample pair whose BOTH rays land in background
// (near-zero on the [0,1]-normalized sinogram) is trivially "symmetric"
// (0-0=0) regardless of true alignment -- unlike a signal-vs-background
// pair, which already produces a large, informative diff. Excluding
// background-only pairs (in both kernels below) prevents a candidate pose
// from faking a low MSE by pushing rays into empty space around the object
// instead of genuinely aligning. Must match pyopencl_backend's
// _MIN_SIGNAL/MIN_SIGNAL constant. See docs/ARCHITECTURE.md for the
// investigation this came from.
#define MIN_SIGNAL 0.01f

// A third fix (relative rather than absolute error, to counter dim-pixel
// bias) was implemented and then reverted (2026-08-26): it changed the
// already-reference-matching result on projs_change.hdf5 (beta 4deg -> 5deg,
// no longer matching Topic_3_forwardsearching.py). Fixes #1/#2 above only
// exclude comparisons that are meaningless by the reference's own logic;
// a relative-error metric optimizes a genuinely different objective and
// isn't safe to apply by default. Staying faithful to the Python reference
// takes priority -- see docs/ARCHITECTURE.md §12.4.

// Kernel 1: precompute sin/cos for all unique alpha and beta values.
// Launch 1D with global size = max(na, nb).
__kernel void precompute_trig(
    __global const float* alpha_arr, __global float* sin_alpha, __global float* cos_alpha, int na,
    __global const float* beta_arr,  __global float* sin_beta,  __global float* cos_beta,  int nb
) {
    int id = get_global_id(0);
    if (id < na) {
        sin_alpha[id] = sin(alpha_arr[id]);
        cos_alpha[id] = cos(alpha_arr[id]);
    }
    if (id < nb) {
        sin_beta[id] = sin(beta_arr[id]);
        cos_beta[id] = cos(beta_arr[id]);
    }
}

// Manual bilinear interpolation over a flat row-major buffer, used by the
// opencl-buffer kernel variant below. Mirrors the Image2D+CLK_FILTER_LINEAR
// sampler's semantics (clamp-to-zero outside [0, W-1) x [0, H-1)) so that
// the buffer and image kernels agree with each other bit-for-bit-ish.
inline float bilinear_buffer(__global const float* sino, int W, int H, float x, float y) {
    if (x < 0.0f || x >= (float)(W - 1) || y < 0.0f || y >= (float)(H - 1))
        return 0.0f;
    int ix = (int)x;
    int iy = (int)y;
    float dx = x - ix;
    float dy = y - iy;
    float v00 = sino[iy       * W + ix];
    float v10 = sino[iy       * W + ix + 1];
    float v01 = sino[(iy + 1) * W + ix];
    float v11 = sino[(iy + 1) * W + ix + 1];
    return (1.0f - dx) * ((1.0f - dy) * v00 + dy * v01)
         +         dx  * ((1.0f - dy) * v10 + dy * v11);
}

// Kernel 2: forward search MSE.
// ND-range: global=(P, WG_SIZE), local=(1, WG_SIZE).
// One work group per (xshift, alpha, beta) combo.
// Work items split N theta steps, accumulate partial MSE, tree-reduce in local mem.
__kernel void forward_search_mse(
    __read_only image2d_t   sinogram,
    __global const float*   xshift_arr,
    __global const float*   sin_alpha,
    __global const float*   cos_alpha,
    __global const float*   sin_beta,
    __global const float*   cos_beta,
    __global const float*   nearest_theta,
    __global       float*   mse_out,
    __global       float*   x0_out,
    __global       float*   y0_out,
    int na, int nb, int N,
    float SDD, float SOD, float pixel_size,
    int detector_width, int detector_height,
    __local float* partial,
    __local int*   partial_count_local
) {
    int combo_id = get_global_id(0);
    int local_id = get_local_id(1);
    int wg_size  = get_local_size(1);

    // Decode combo_id -> (i_xshift, i_alpha, i_beta)
    int i_xshift = combo_id / (na * nb);
    int i_alpha  = (combo_id % (na * nb)) / nb;
    int i_beta   =  combo_id % nb;

    float xshift = xshift_arr[i_xshift];
    float sin_a  = sin_alpha[i_alpha];
    float cos_a  = cos_alpha[i_alpha];
    float sin_b  = sin_beta[i_beta];
    float cos_b  = cos_beta[i_beta];

    // Compute x_0, y_0 from the matrix formula in find_conebeam_COR_line_forward
    float b0    = SDD * cos_a * sin_b;
    float b1    = SDD * cos_a * xshift * cos_b;
    float det_A = -sin_a * (-xshift * cos_a * sin_b + SOD * sin_a)
                  - cos_a * cos_b * SOD * cos_a * cos_b;

    float x_0 = 0.0f, y_0 = 0.0f;
    if (det_A != 0.0f) {
        float inv00 = (-xshift * cos_a * sin_b + SOD * sin_a);
        float inv01 = -cos_a * cos_b;
        float inv10 = -SOD * cos_a * cos_b;
        float inv11 = -sin_a;
        x_0 = (inv00 * b0 + inv01 * b1) / det_A;
        y_0 = (inv10 * b0 + inv11 * b1) / det_A;
    }

    // Compute theta_0 = atan(x_p / z_p)
    float x_p    = cos_a * x_0 + sin_a * cos_b * y_0 - sin_a * sin_b * SDD;
    float z_p    = -sin_b * y_0 - cos_b * SDD;
    float theta_0 = atan(x_p / z_p);

    float hw = (detector_width  - 1) * 0.5f;
    float hh = (detector_height - 1) * 0.5f;

    // Unnormalized coords, clamp-to-zero out-of-bounds (matches Python default of 0),
    // hardware bilinear interpolation via texture units
    const sampler_t smp = CLK_NORMALIZED_COORDS_FALSE |
                          CLK_ADDRESS_CLAMP |
                          CLK_FILTER_LINEAR;

    // Each work item handles ceil(N / wg_size) thetas
    float partial_sum = 0.0f;
    int partial_count = 0;
    for (int i = local_id; i < N; i += wg_size) {
        float nt     = nearest_theta[i];
        float theta  =  nt + theta_0;
        float rtheta = -nt + theta_0;

        float tan_t  = tan(theta);
        float tan_rt = tan(rtheta);

        float denom  = tan_t  * sin_a * sin_b + cos_b;
        float rdenom = tan_rt * sin_a * sin_b + cos_b;
        float temp   = (denom  != 0.0f) ? SDD / denom  : 0.0f;
        float rtemp  = (rdenom != 0.0f) ? SDD / rdenom : 0.0f;

        float px  = (-(temp  * tan_t  * cos_a)) / pixel_size + hw;
        float py  =  hh - (temp  * (-tan_t  * sin_a * cos_b + sin_b)) / pixel_size;
        float prx = (-(rtemp * tan_rt * cos_a)) / pixel_size + hw;
        float pry =  hh - (rtemp * (-tan_rt * sin_a * cos_b + sin_b)) / pixel_size;

        // CORRECTION (2026-08-26): same fix as forward_search_mse_buffer --
        // only average over pairs where both rays are on-detector, bounds
        // checked explicitly here rather than relying on the hardware
        // sampler's CLK_ADDRESS_CLAMP behavior (clamping, not zero-outside,
        // so it wouldn't even reproduce the same bug/fix as the buffer path).
        // Untested on this dev machine (image mode crashes here, see
        // docs/ARCHITECTURE.md §7) but kept consistent with the buffer kernel.
        bool valid_t  = (px  >= 0.0f && px  < (float)(detector_width  - 1) && py  >= 0.0f && py  < (float)(detector_height - 1));
        bool valid_rt = (prx >= 0.0f && prx < (float)(detector_width  - 1) && pry >= 0.0f && pry < (float)(detector_height - 1));
        if (valid_t && valid_rt) {
            // +0.5 shifts from pixel index to pixel-center in OpenCL's coordinate system
            float f_t  = read_imagef(sinogram, smp, (float2)(px  + 0.5f, py  + 0.5f)).x;
            float f_rt = read_imagef(sinogram, smp, (float2)(prx + 0.5f, pry + 0.5f)).x;
            if (f_t >= MIN_SIGNAL || f_rt >= MIN_SIGNAL) {
                float diff = f_t - f_rt;
                partial_sum += diff * diff;
                partial_count += 1;
            }
        }
    }

    // Parallel tree reduction in local memory
    partial[local_id] = partial_sum;
    partial_count_local[local_id] = partial_count;
    barrier(CLK_LOCAL_MEM_FENCE);

    for (int stride = wg_size >> 1; stride > 0; stride >>= 1) {
        if (local_id < stride) {
            partial[local_id] += partial[local_id + stride];
            partial_count_local[local_id] += partial_count_local[local_id + stride];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (local_id == 0) {
        mse_out[combo_id] = (partial_count_local[0] > 0) ? (partial[0] / (float)partial_count_local[0]) : FLT_MAX;
        x0_out[combo_id]  = x_0;
        y0_out[combo_id]  = y_0;
    }
}

// opencl-buffer variant of forward_search_mse: identical math, but the
// sinogram is a plain __global buffer with manual bilinear interpolation
// (bilinear_buffer above) instead of an Image2D + hardware sampler. Kept as
// a separate kernel (rather than an #ifdef) so both data-format versions can
// be built and benchmarked from the same program in one run.
__kernel void forward_search_mse_buffer(
    __global const float*   sinogram,
    __global const float*   xshift_arr,
    __global const float*   sin_alpha,
    __global const float*   cos_alpha,
    __global const float*   sin_beta,
    __global const float*   cos_beta,
    __global const float*   nearest_theta,
    __global       float*   mse_out,
    __global       float*   x0_out,
    __global       float*   y0_out,
    int na, int nb, int N,
    float SDD, float SOD, float pixel_size,
    int detector_width, int detector_height,
    __local float* partial,
    __local int*   partial_count_local
) {
    int combo_id = get_global_id(0);
    int local_id = get_local_id(1);
    int wg_size  = get_local_size(1);

    int i_xshift = combo_id / (na * nb);
    int i_alpha  = (combo_id % (na * nb)) / nb;
    int i_beta   =  combo_id % nb;

    float xshift = xshift_arr[i_xshift];
    float sin_a  = sin_alpha[i_alpha];
    float cos_a  = cos_alpha[i_alpha];
    float sin_b  = sin_beta[i_beta];
    float cos_b  = cos_beta[i_beta];

    float b0    = SDD * cos_a * sin_b;
    float b1    = SDD * cos_a * xshift * cos_b;
    float det_A = -sin_a * (-xshift * cos_a * sin_b + SOD * sin_a)
                  - cos_a * cos_b * SOD * cos_a * cos_b;

    float x_0 = 0.0f, y_0 = 0.0f;
    if (det_A != 0.0f) {
        float inv00 = (-xshift * cos_a * sin_b + SOD * sin_a);
        float inv01 = -cos_a * cos_b;
        float inv10 = -SOD * cos_a * cos_b;
        float inv11 = -sin_a;
        x_0 = (inv00 * b0 + inv01 * b1) / det_A;
        y_0 = (inv10 * b0 + inv11 * b1) / det_A;
    }

    float x_p    = cos_a * x_0 + sin_a * cos_b * y_0 - sin_a * sin_b * SDD;
    float z_p    = -sin_b * y_0 - cos_b * SDD;
    float theta_0 = atan(x_p / z_p);

    float hw = (detector_width  - 1) * 0.5f;
    float hh = (detector_height - 1) * 0.5f;

    float partial_sum = 0.0f;
    int partial_count = 0;
    for (int i = local_id; i < N; i += wg_size) {
        float nt     = nearest_theta[i];
        float theta  =  nt + theta_0;
        float rtheta = -nt + theta_0;

        float tan_t  = tan(theta);
        float tan_rt = tan(rtheta);

        float denom  = tan_t  * sin_a * sin_b + cos_b;
        float rdenom = tan_rt * sin_a * sin_b + cos_b;
        float temp   = (denom  != 0.0f) ? SDD / denom  : 0.0f;
        float rtemp  = (rdenom != 0.0f) ? SDD / rdenom : 0.0f;

        // No +0.5 offset here: bilinear_buffer indexes the array directly at
        // pixel-index coordinates, matching the Python reference's indexing.
        float px  = (-(temp  * tan_t  * cos_a)) / pixel_size + hw;
        float py  =  hh - (temp  * (-tan_t  * sin_a * cos_b + sin_b)) / pixel_size;
        float prx = (-(rtemp * tan_rt * cos_a)) / pixel_size + hw;
        float pry =  hh - (rtemp * (-tan_rt * sin_a * cos_b + sin_b)) / pixel_size;

        // CORRECTION (2026-08-26): only average over sample pairs where BOTH
        // rays land on-detector, instead of always dividing by the fixed N.
        // Previously an off-detector ray silently contributed (0-0)^2=0 via
        // bilinear_buffer's out-of-bounds return, diluting a candidate's MSE
        // toward zero the more of its rays miss the detector -- a free way
        // to fake a "good" (low) MSE with no real symmetry match behind it.
        // Confirmed via docs/ARCHITECTURE.md's investigation: on a
        // small-detector dataset this made the search converge on its range
        // boundary with MSE=0.0 instead of the true alignment. A candidate
        // with zero valid pairs gets FLT_MAX (worst possible), not 0.
        bool valid_t  = (px  >= 0.0f && px  < (float)(detector_width  - 1) && py  >= 0.0f && py  < (float)(detector_height - 1));
        bool valid_rt = (prx >= 0.0f && prx < (float)(detector_width  - 1) && pry >= 0.0f && pry < (float)(detector_height - 1));
        if (valid_t && valid_rt) {
            float f_t  = bilinear_buffer(sinogram, detector_width, detector_height, px,  py);
            float f_rt = bilinear_buffer(sinogram, detector_width, detector_height, prx, pry);
            if (f_t >= MIN_SIGNAL || f_rt >= MIN_SIGNAL) {
                float diff = f_t - f_rt;
                partial_sum += diff * diff;
                partial_count += 1;
            }
        }
    }

    partial[local_id] = partial_sum;
    partial_count_local[local_id] = partial_count;
    barrier(CLK_LOCAL_MEM_FENCE);

    for (int stride = wg_size >> 1; stride > 0; stride >>= 1) {
        if (local_id < stride) {
            partial[local_id] += partial[local_id + stride];
            partial_count_local[local_id] += partial_count_local[local_id + stride];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (local_id == 0) {
        mse_out[combo_id] = (partial_count_local[0] > 0) ? (partial[0] / (float)partial_count_local[0]) : FLT_MAX;
        x0_out[combo_id]  = x_0;
        y0_out[combo_id]  = y_0;
    }
}
