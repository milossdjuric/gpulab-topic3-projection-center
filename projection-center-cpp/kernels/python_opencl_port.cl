
#define MIN_SIGNAL 0.01f

float bilinear_sample(__global const float *image, int width, int height, float x, float y)
{
    if (x < 0.0f || x >= (float)(width - 1) || y < 0.0f || y >= (float)(height - 1)) {
        return 0.0f;
    }

    int ix = (int)floor(x);
    int iy = (int)floor(y);
    float dx = x - (float)ix;
    float dy = y - (float)iy;

    int row0 = iy * width;
    int row1 = (iy + 1) * width;
    float v00 = image[row0 + ix];
    float v10 = image[row0 + ix + 1];
    float v01 = image[row1 + ix];
    float v11 = image[row1 + ix + 1];

    float top = mad(dx, v10 - v00, v00);
    float bottom = mad(dx, v11 - v01, v01);
    return mad(dy, bottom - top, top);
}

__kernel void center_search_reduce(
    __global const float *sinogram,
    __global const float *alpha_values,
    __global const float *beta_values,
    __global const float *tan_theta0_values,
    __global const float *tan_dtheta_values,
    __global float *mse_values,
    const float sdd,
    const int detector_width,
    const int detector_height,
    const float pixel_size,
    const int sample_count)
{
    int lane = get_local_id(0);
    int workgroup_size = get_local_size(0);
    int parameter_index = get_group_id(1);

    float alpha = alpha_values[parameter_index];
    float beta = beta_values[parameter_index];
    float tan_theta0 = tan_theta0_values[parameter_index];

    float sin_a = sin(alpha);
    float cos_a = cos(alpha);
    float sin_b = sin(beta);
    float cos_b = cos(beta);

    float local_sum = 0.0f;
    int local_count = 0;

    // tan_dtheta_values is identical for every candidate pose (the sampled
    // angles don't depend on alpha/beta/tan_theta0), precomputed once on
    // the host instead of recomputed here per candidate. tan(theta0 +/-
    // dtheta) then comes from the tangent addition/subtraction formula,
    // pure multiply/add/divide, no trig call in this hot loop at all.
    // Ported from forward_search.cl's identical optimization.
    for (int sample_index = lane; sample_index < sample_count; sample_index += workgroup_size) {
        float tan_dtheta = tan_dtheta_values[sample_index];

        float tan_t  = (tan_theta0 + tan_dtheta) / (1.0f - tan_theta0 * tan_dtheta);
        float tan_rt = (tan_theta0 - tan_dtheta) / (1.0f + tan_theta0 * tan_dtheta);

        float temp = sdd / (tan_t * sin_a * sin_b + cos_b);
        float rtemp = sdd / (tan_rt * sin_a * sin_b + cos_b);

        float y = temp * (-tan_t * sin_a * cos_b + sin_b);
        float ry = rtemp * (-tan_rt * sin_a * cos_b + sin_b);

        float x = -temp * tan_t * cos_a;
        float rx = -rtemp * tan_rt * cos_a;

        y = ((float)(detector_height - 1) * 0.5f) - y / pixel_size;
        ry = ((float)(detector_height - 1) * 0.5f) - ry / pixel_size;
        x = x / pixel_size + ((float)(detector_width - 1) * 0.5f);
        rx = rx / pixel_size + ((float)(detector_width - 1) * 0.5f);

        // CORRECTION (2026-08-26): only count sample pairs where BOTH rays
        // land on-detector. Previously every sample contributed to local_sum
        // (via bilinear_sample's 0.0f-outside-bounds behavior) and the final
        // division was always by the fixed sample_count -- so a candidate
        // pose that pushes most/all rays off-detector got most/all of its
        // squared-diff terms silently replaced by (0-0)^2=0, diluting its
        // MSE toward zero regardless of true alignment. That makes "push
        // everything off-detector" a trivial way to fake a low MSE, which is
        // exactly what happened on a small-detector dataset (128x128, small
        // SDD/SOD) where the default search range pushes many candidates
        // off-detector -- the found "optimum" sat at the search-range
        // boundary with MSE=0.0, not at the true alignment. Normalizing by
        // the count of genuinely valid (in-bounds on both sides) pairs
        // instead removes that incentive; a candidate with zero valid pairs
        // gets no information and must not be treated as a good match.
        bool valid_t  = (x  >= 0.0f && x  < (float)(detector_width  - 1) && y  >= 0.0f && y  < (float)(detector_height - 1));
        bool valid_rt = (rx >= 0.0f && rx < (float)(detector_width  - 1) && ry >= 0.0f && ry < (float)(detector_height - 1));
        if (valid_t && valid_rt) {
            float f_theta = bilinear_sample(sinogram, detector_width, detector_height, x, y);
            float f_rtheta = bilinear_sample(sinogram, detector_width, detector_height, rx, ry);
            // CORRECTION (2026-08-26): a pair where BOTH sides are
            // background (near-zero on the [0,1]-normalized sinogram) is
            // trivially "symmetric" without reflecting real alignment --
            // unlike a signal-vs-background pair, which already produces a
            // large, informative diff. MIN_SIGNAL must match the Python
            // CpuBackend's _MIN_SIGNAL constant above.
            if (f_theta >= MIN_SIGNAL || f_rtheta >= MIN_SIGNAL) {
                float diff = f_theta - f_rtheta;
                local_sum += diff * diff;
                local_count += 1;
            }
        }
    }

    __local float scratch[256];
    __local int count_scratch[256];
    scratch[lane] = local_sum;
    count_scratch[lane] = local_count;
    barrier(CLK_LOCAL_MEM_FENCE);

    for (int stride = workgroup_size / 2; stride > 0; stride >>= 1) {
        if (lane < stride) {
            scratch[lane] += scratch[lane + stride];
            count_scratch[lane] += count_scratch[lane + stride];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (lane == 0) {
        mse_values[parameter_index] = (count_scratch[0] > 0) ? (scratch[0] / (float)count_scratch[0]) : FLT_MAX;
    }
}

__kernel void resample_projections(
    __global const float *input_projections,
    __global float *output_projections,
    __global const float *rotation_matrix,
    const int batch_size,
    const int input_width,
    const int input_height,
    const float input_pixel_size,
    const float input_sdd,
    const int output_width,
    const int output_height,
    const float output_pixel_size,
    const float output_sdd)
{
    int x_idx = get_global_id(0);
    int y_idx = get_global_id(1);
    int proj_idx = get_global_id(2);

    if (x_idx >= output_width || y_idx >= output_height || proj_idx >= batch_size) {
        return;
    }

    float x = ((float)x_idx - ((float)(output_width - 1) * 0.5f)) * output_pixel_size;
    float y = (((float)(output_height - 1) * 0.5f) - (float)y_idx) * output_pixel_size;
    float z = -output_sdd;

    float nx = rotation_matrix[0] * x + rotation_matrix[1] * y + rotation_matrix[2] * z;
    float ny = rotation_matrix[3] * x + rotation_matrix[4] * y + rotation_matrix[5] * z;
    float nz = rotation_matrix[6] * x + rotation_matrix[7] * y + rotation_matrix[8] * z;

    float det_x = nx / nz * (-input_sdd);
    float det_y = ny / nz * (-input_sdd);
    float source_x = det_x / input_pixel_size + ((float)(input_width - 1) * 0.5f);
    float source_y = -det_y / input_pixel_size + ((float)(input_height - 1) * 0.5f);

    int input_proj_offset = proj_idx * input_width * input_height;
    int output_proj_offset = proj_idx * output_width * output_height;
    float value = bilinear_sample(input_projections + input_proj_offset, input_width, input_height, source_x, source_y);
    output_projections[output_proj_offset + y_idx * output_width + x_idx] = value;
}
