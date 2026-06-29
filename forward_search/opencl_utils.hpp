#pragma once
#define CL_HPP_ENABLE_EXCEPTIONS
#define CL_HPP_MINIMUM_OPENCL_VERSION 120
#define CL_HPP_TARGET_OPENCL_VERSION 300
#include <CL/opencl.hpp>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

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
