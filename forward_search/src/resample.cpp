// The actual resample code. Given a dataset and a pose, works out where
// the corrected geometry ends up, builds a rotation matrix for it, and
// runs the resample kernel on the GPU in batches. computeResample(), at
// the bottom, is the function every CLI binary calls; everything above it
// is a helper only this file uses.
#define CL_HPP_ENABLE_EXCEPTIONS
#define CL_HPP_MINIMUM_OPENCL_VERSION 120
#define CL_HPP_TARGET_OPENCL_VERSION 300
#include <CL/opencl.hpp>
#include "resample.hpp"
#include "opencl_utils.hpp"
#include <algorithm>
#include <cmath>
#include <iostream>

namespace {

// A plain 3D point/direction, used for the geometry math below.
struct Vec3 { double x, y, z; };

Vec3 cross(const Vec3& a, const Vec3& b) {
    return { a.y * b.z - a.z * b.y,
             a.z * b.x - a.x * b.z,
             a.x * b.y - a.y * b.x };
}
double norm(const Vec3& v) { return std::sqrt(v.x * v.x + v.y * v.y + v.z * v.z); }
Vec3 normalized(const Vec3& v) { double n = norm(v); return { v.x / n, v.y / n, v.z / n }; }

// Works out the corrected geometry for the output image: the new
// effective distances (SDD/SOD) once the pose is accounted for, and the
// new detector size after downsample is applied. Matches
// Topic_3_resampling.py's get_cb_para().
CbPara computeCorrectedGeometry(const CbPara& para, const ResamplePose& pose, int downsample) {
    CbPara out = para;
    out.pixel_size      = para.pixel_size * downsample;
    out.detector_width  = para.detector_width  / downsample;
    out.detector_height = para.detector_height / downsample;

    Vec3 center{ pose.center_x, pose.center_y, para.SDD };
    out.SDD = norm(center);

    double sin_a = std::sin(pose.alpha), cos_a = std::cos(pose.alpha);
    double sin_b = std::sin(pose.beta),  cos_b = std::cos(pose.beta);
    Vec3 A{ pose.xshift, 0.0, -para.SOD };
    Vec3 v_real{ -sin_a, cos_a * cos_b, cos_a * sin_b };
    out.SOD = norm(cross(A, v_real)) / norm(v_real);

    return out;
}

// Builds the 3x3 rotation matrix for the found pose: three axes (x, y, z)
// worked out from alpha/beta and the found center point, stored flat so
// the resample.cl kernel can read it directly. Matches
// Topic_3_resampling.py's get_rotation_matrix().
void computeRotationMatrix(const CbPara& para, const ResamplePose& pose, float out[9]) {
    double sin_a = std::sin(pose.alpha), cos_a = std::cos(pose.alpha);
    double sin_b = std::sin(pose.beta),  cos_b = std::cos(pose.beta);

    Vec3 y_axis{ -sin_a, cos_a * cos_b, cos_a * sin_b };
    Vec3 z_axis = normalized(Vec3{ -pose.center_x, -pose.center_y, para.SDD });
    Vec3 x_axis = normalized(cross(y_axis, z_axis));
    // The reference does a pointless extra step here (computes a cross
    // product it never uses, then just re-normalizes z_axis again). It
    // does nothing, but it's kept here on purpose so this code matches
    // Topic_3_resampling.py's own math exactly, not just its result.
    z_axis = normalized(z_axis);

    out[0] = (float)x_axis.x; out[1] = (float)y_axis.x; out[2] = (float)z_axis.x;
    out[3] = (float)x_axis.y; out[4] = (float)y_axis.y; out[5] = (float)z_axis.y;
    out[6] = (float)x_axis.z; out[7] = (float)y_axis.z; out[8] = (float)z_axis.z;
}

} // namespace

// Runs resample, called by every CLI binary that does it. Works out the
// corrected geometry once, picks a GPU and compiles the kernel once, then
// sends the images through it a batch at a time (so the whole dataset
// never has to fit on the GPU at once), and returns everything corrected.
ResampleOutput computeResample(const CbPara& para,
                               const std::vector<float>& projs,
                               const ResamplePose& pose,
                               int downsample_factor,
                               const std::string& kernel_path,
                               int batch_size) {
    CbPara out_para = computeCorrectedGeometry(para, pose, downsample_factor);
    float rotation[9];
    computeRotationMatrix(para, pose, rotation);

    cl::Device dev = pickGPU();
    std::cerr << "Device: " << dev.getInfo<CL_DEVICE_NAME>() << "\n";
    cl::Context ctx({dev});
    cl::CommandQueue queue(ctx, dev);
    cl::Program prog = buildProgram(ctx, dev, kernel_path);
    cl::Kernel kernel(prog, "resample_projections");

    // Uploaded once, read by every batch: the rotation matrix does not
    // change from one batch to the next.
    cl::Buffer rot_buf(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, 9 * sizeof(float), rotation);

    int in_w = para.detector_width,      in_h = para.detector_height;   // input detector size
    int out_w = out_para.detector_width, out_h = out_para.detector_height; // output (corrected) detector size
    int num_projs = para.num_projs;

    std::vector<float> result(static_cast<size_t>(num_projs) * out_w * out_h);

    // One pair of GPU buffers, reused for every batch instead of
    // allocating a new pair each time. The last batch may be smaller and
    // just uses part of them.
    cl::Buffer in_buf(ctx, CL_MEM_READ_ONLY, static_cast<size_t>(batch_size) * in_w * in_h * sizeof(float));
    cl::Buffer out_buf(ctx, CL_MEM_WRITE_ONLY, static_cast<size_t>(batch_size) * out_w * out_h * sizeof(float));

    // One iteration per batch of projections: upload this batch's pixels,
    // dispatch the kernel on them, read the corrected pixels back, repeat
    // until every projection has been processed.
    for (int start = 0; start < num_projs; start += batch_size) {
        int count = std::min(batch_size, num_projs - start);  // last batch may be smaller

        queue.enqueueWriteBuffer(in_buf, CL_TRUE, 0,
                                 static_cast<size_t>(count) * in_w * in_h * sizeof(float),
                                 projs.data() + static_cast<size_t>(start) * in_w * in_h);

        kernel.setArg(0, in_buf);
        kernel.setArg(1, out_buf);
        kernel.setArg(2, rot_buf);
        kernel.setArg(3, count);
        kernel.setArg(4, in_w);
        kernel.setArg(5, in_h);
        kernel.setArg(6, (float)para.pixel_size);
        kernel.setArg(7, (float)para.SDD);
        kernel.setArg(8, out_w);
        kernel.setArg(9, out_h);
        kernel.setArg(10, (float)out_para.pixel_size);
        kernel.setArg(11, (float)out_para.SDD);

        queue.enqueueNDRangeKernel(kernel, cl::NullRange, cl::NDRange(out_w, out_h, count), cl::NullRange);
        queue.enqueueReadBuffer(out_buf, CL_TRUE, 0,
                                static_cast<size_t>(count) * out_w * out_h * sizeof(float),
                                result.data() + static_cast<size_t>(start) * out_w * out_h);
    }

    std::cerr << "Resampled " << num_projs << " projections\n";
    return { out_para, std::move(result) };
}
