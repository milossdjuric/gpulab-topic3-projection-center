// Kernel 1: precompute sin/cos for all unique alpha and beta values.
// Launch 1D with global size = max(na, nb).
__kernel void precompute_trig(
    __global const float* alpha_arr, __global float* sin_alpha, __global float* cos_alpha, int na,
    __global const float* beta_arr,  __global float* sin_beta,  __global float* cos_beta,  int nb
) {
    int id = get_global_id(0);
    if (id < na) {
        sin_alpha[id] = native_sin(alpha_arr[id]);
        cos_alpha[id] = native_cos(alpha_arr[id]);
    }
    if (id < nb) {
        sin_beta[id] = native_sin(beta_arr[id]);
        cos_beta[id] = native_cos(beta_arr[id]);
    }
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
    __local float* partial
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
    for (int i = local_id; i < N; i += wg_size) {
        float nt     = nearest_theta[i];
        float theta  =  nt + theta_0;
        float rtheta = -nt + theta_0;

        float tan_t  = native_tan(theta);
        float tan_rt = native_tan(rtheta);

        float denom  = tan_t  * sin_a * sin_b + cos_b;
        float rdenom = tan_rt * sin_a * sin_b + cos_b;
        float temp   = (denom  != 0.0f) ? SDD / denom  : 0.0f;
        float rtemp  = (rdenom != 0.0f) ? SDD / rdenom : 0.0f;

        float px  = (-(temp  * tan_t  * cos_a)) / pixel_size + hw;
        float py  =  hh - (temp  * (-tan_t  * sin_a * cos_b + sin_b)) / pixel_size;
        float prx = (-(rtemp * tan_rt * cos_a)) / pixel_size + hw;
        float pry =  hh - (rtemp * (-tan_rt * sin_a * cos_b + sin_b)) / pixel_size;

        // +0.5 shifts from pixel index to pixel-center in OpenCL's coordinate system
        float f_t  = read_imagef(sinogram, smp, (float2)(px  + 0.5f, py  + 0.5f)).x;
        float f_rt = read_imagef(sinogram, smp, (float2)(prx + 0.5f, pry + 0.5f)).x;

        float diff = f_t - f_rt;
        partial_sum += diff * diff;
    }

    // Parallel tree reduction in local memory
    partial[local_id] = partial_sum;
    barrier(CLK_LOCAL_MEM_FENCE);

    for (int stride = wg_size >> 1; stride > 0; stride >>= 1) {
        if (local_id < stride)
            partial[local_id] += partial[local_id + stride];
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (local_id == 0) {
        mse_out[combo_id] = partial[0] / (float)N;
        x0_out[combo_id]  = x_0;
        y0_out[combo_id]  = y_0;
    }
}
