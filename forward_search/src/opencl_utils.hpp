// Two small helpers every binary in this project needs: finding a GPU,
// and compiling a .cl kernel file for it. Kept here once instead of
// copy-pasted into every main.cpp.
#pragma once
#define CL_HPP_ENABLE_EXCEPTIONS
#define CL_HPP_MINIMUM_OPENCL_VERSION 120
#define CL_HPP_TARGET_OPENCL_VERSION 300
#include <CL/opencl.hpp>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

// Finds the first GPU it can on the machine. Checks every installed
// OpenCL driver in turn (there can be more than one, e.g. Intel and
// NVIDIA both installed), skipping any that error out, and fails only if
// none of them has a GPU at all.
inline cl::Device pickGPU() {
    std::vector<cl::Platform> platforms;
    cl::Platform::get(&platforms);
    for (auto& p : platforms) {
        std::vector<cl::Device> devs;
        try { p.getDevices(CL_DEVICE_TYPE_GPU, &devs); } catch (...) { continue; }
        if (!devs.empty()) return devs[0];
    }
    throw std::runtime_error("No GPU device found");
}

// Reads the kernel file at path and compiles it for the given device. If
// it fails to compile, the real compiler error gets attached to the
// message instead of just saying "build failed" with no details.
inline cl::Program buildProgram(const cl::Context& ctx,
                                const cl::Device& dev,
                                const std::string& path,
                                const std::string& opts = "") {
    std::ifstream f(path);
    if (!f) throw std::runtime_error("Cannot open kernel: " + path);
    std::string src((std::istreambuf_iterator<char>(f)), {});
    cl::Program prog(ctx, src);
    try {
        prog.build({dev}, opts.c_str());
    } catch (const cl::Error&) {
        std::string log = prog.getBuildInfo<CL_PROGRAM_BUILD_LOG>(dev);
        throw std::runtime_error("Kernel build failed:\n" + log);
    }
    return prog;
}
