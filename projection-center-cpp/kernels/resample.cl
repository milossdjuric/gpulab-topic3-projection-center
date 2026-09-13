// Resampling kernel: one work-item per output detector pixel per projection.
// Mirrors Topic_3_resampling.py's interpolate() and pyopencl_backend's own
// resample_projections kernel (src/projection_center_searching/backends.py) --
// same math, so a work-item here does exactly what one Python-loop iteration
// of interpolate()'s inner double loop did, just for one (pixel, projection)
// pair instead of scanning the whole output image serially.

inline float bilinear_sample(__global const float* image, int width, int height, float x, float y) {
    if (x < 0.0f || x >= (float)(width - 1) || y < 0.0f || y >= (float)(height - 1))
        return 0.0f;
    int ix = (int)x;
    int iy = (int)y;
    float dx = x - ix;
    float dy = y - iy;
    float v00 = image[iy       * width + ix];
    float v10 = image[iy       * width + ix + 1];
    float v01 = image[(iy + 1) * width + ix];
    float v11 = image[(iy + 1) * width + ix + 1];
    return (1.0f - dx) * ((1.0f - dy) * v00 + dy * v01)
         +         dx  * ((1.0f - dy) * v10 + dy * v11);
}

__kernel void resample_projections(
    __global const float* input_projections,
    __global       float* output_projections,
    __global const float* rotation_matrix,   // 9 floats, row-major 3x3
    int   batch_size,
    int   input_width,  int input_height,  float input_pixel_size,  float input_sdd,
    int   output_width, int output_height, float output_pixel_size, float output_sdd
) {
    int x_idx    = get_global_id(0);
    int y_idx    = get_global_id(1);
    int proj_idx = get_global_id(2);

    if (x_idx >= output_width || y_idx >= output_height || proj_idx >= batch_size)
        return;

    // Corrected-detector pixel -> 3D ray direction (same convention as
    // Topic_3_resampling.py's interpolate(): x right, y up, z toward source).
    float x = ((float)x_idx - (float)(output_width  - 1) * 0.5f) * output_pixel_size;
    float y = ((float)(output_height - 1) * 0.5f - (float)y_idx) * output_pixel_size;
    float z = -output_sdd;

    // Rotate into the original (uncorrected) detector's frame.
    float nx = rotation_matrix[0] * x + rotation_matrix[1] * y + rotation_matrix[2] * z;
    float ny = rotation_matrix[3] * x + rotation_matrix[4] * y + rotation_matrix[5] * z;
    float nz = rotation_matrix[6] * x + rotation_matrix[7] * y + rotation_matrix[8] * z;

    // Project onto the original detector plane, convert to pixel coordinates.
    float det_x = nx / nz * (-input_sdd);
    float det_y = ny / nz * (-input_sdd);
    float source_x =  det_x / input_pixel_size + (float)(input_width  - 1) * 0.5f;
    float source_y = -det_y / input_pixel_size + (float)(input_height - 1) * 0.5f;

    int in_offset  = proj_idx * input_width  * input_height;
    int out_offset = proj_idx * output_width * output_height;
    float value = bilinear_sample(input_projections + in_offset, input_width, input_height, source_x, source_y);
    output_projections[out_offset + y_idx * output_width + x_idx] = value;
}
