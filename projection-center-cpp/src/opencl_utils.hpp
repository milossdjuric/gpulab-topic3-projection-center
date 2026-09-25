// Two small helpers every binary in this project needs: finding a GPU,
// and compiling a .cl kernel file for it. Kept here once instead of
// copy-pasted into every main.cpp.
#pragma once
#define CL_HPP_ENABLE_EXCEPTIONS
// Minimum 2.0 (not 1.2): with a minimum below 2.0, opencl.hpp adds a
// run-time platform-version check whose helper some header versions
// (2022.09-2023.02, e.g. Debian 12's) forget to define, so the build fails.
// Every current GPU driver (NVIDIA, Intel, AMD) provides OpenCL 2.0+/3.0.
// The same pair of defines is repeated at the top of each .cpp that
// includes opencl.hpp directly.
#define CL_HPP_MINIMUM_OPENCL_VERSION 200
#define CL_HPP_TARGET_OPENCL_VERSION 300
#include <CL/opencl.hpp>
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

// The name of an OpenCL error code (CL_OUT_OF_RESOURCES for -5, ...), so
// an error message says what went wrong, not just which call failed.
inline const char* clErrorName(cl_int err) {
    switch (err) {
        case CL_DEVICE_NOT_FOUND:                 return "CL_DEVICE_NOT_FOUND";
        case CL_DEVICE_NOT_AVAILABLE:             return "CL_DEVICE_NOT_AVAILABLE";
        case CL_COMPILER_NOT_AVAILABLE:           return "CL_COMPILER_NOT_AVAILABLE";
        case CL_MEM_OBJECT_ALLOCATION_FAILURE:    return "CL_MEM_OBJECT_ALLOCATION_FAILURE";
        case CL_OUT_OF_RESOURCES:                 return "CL_OUT_OF_RESOURCES";
        case CL_OUT_OF_HOST_MEMORY:               return "CL_OUT_OF_HOST_MEMORY";
        case CL_PROFILING_INFO_NOT_AVAILABLE:     return "CL_PROFILING_INFO_NOT_AVAILABLE";
        case CL_IMAGE_FORMAT_NOT_SUPPORTED:       return "CL_IMAGE_FORMAT_NOT_SUPPORTED";
        case CL_BUILD_PROGRAM_FAILURE:            return "CL_BUILD_PROGRAM_FAILURE";
        case CL_INVALID_VALUE:                    return "CL_INVALID_VALUE";
        case CL_INVALID_DEVICE_TYPE:              return "CL_INVALID_DEVICE_TYPE";
        case CL_INVALID_PLATFORM:                 return "CL_INVALID_PLATFORM";
        case CL_INVALID_DEVICE:                   return "CL_INVALID_DEVICE";
        case CL_INVALID_CONTEXT:                  return "CL_INVALID_CONTEXT";
        case CL_INVALID_QUEUE_PROPERTIES:         return "CL_INVALID_QUEUE_PROPERTIES";
        case CL_INVALID_COMMAND_QUEUE:            return "CL_INVALID_COMMAND_QUEUE";
        case CL_INVALID_MEM_OBJECT:               return "CL_INVALID_MEM_OBJECT";
        case CL_INVALID_IMAGE_SIZE:               return "CL_INVALID_IMAGE_SIZE";
        case CL_INVALID_BINARY:                   return "CL_INVALID_BINARY";
        case CL_INVALID_BUILD_OPTIONS:            return "CL_INVALID_BUILD_OPTIONS";
        case CL_INVALID_PROGRAM:                  return "CL_INVALID_PROGRAM";
        case CL_INVALID_PROGRAM_EXECUTABLE:       return "CL_INVALID_PROGRAM_EXECUTABLE";
        case CL_INVALID_KERNEL_NAME:              return "CL_INVALID_KERNEL_NAME";
        case CL_INVALID_KERNEL:                   return "CL_INVALID_KERNEL";
        case CL_INVALID_ARG_INDEX:                return "CL_INVALID_ARG_INDEX";
        case CL_INVALID_ARG_VALUE:                return "CL_INVALID_ARG_VALUE";
        case CL_INVALID_ARG_SIZE:                 return "CL_INVALID_ARG_SIZE";
        case CL_INVALID_KERNEL_ARGS:              return "CL_INVALID_KERNEL_ARGS";
        case CL_INVALID_WORK_DIMENSION:           return "CL_INVALID_WORK_DIMENSION";
        case CL_INVALID_WORK_GROUP_SIZE:          return "CL_INVALID_WORK_GROUP_SIZE";
        case CL_INVALID_WORK_ITEM_SIZE:           return "CL_INVALID_WORK_ITEM_SIZE";
        case CL_INVALID_GLOBAL_OFFSET:            return "CL_INVALID_GLOBAL_OFFSET";
        case CL_INVALID_EVENT_WAIT_LIST:          return "CL_INVALID_EVENT_WAIT_LIST";
        case CL_INVALID_EVENT:                    return "CL_INVALID_EVENT";
        case CL_INVALID_OPERATION:                return "CL_INVALID_OPERATION";
        case CL_INVALID_BUFFER_SIZE:              return "CL_INVALID_BUFFER_SIZE";
        case CL_INVALID_GLOBAL_WORK_SIZE:         return "CL_INVALID_GLOBAL_WORK_SIZE";
        case -1001:                               return "CL_PLATFORM_NOT_FOUND_KHR (no OpenCL driver/ICD installed)";
        default:                                  return "unknown OpenCL error";
    }
}

// Every OpenCL device on the machine, one line each, with the
// "platform:device" index OPENCL_DEVICE (below) takes. Used in error
// messages so a failure shows what the machine actually has.
inline std::string describeDevices(const std::vector<cl::Platform>& platforms) {
    std::string s;
    for (size_t p = 0; p < platforms.size(); ++p) {
        std::vector<cl::Device> devs;
        try { platforms[p].getDevices(CL_DEVICE_TYPE_ALL, &devs); } catch (...) {}
        std::string pname;
        try { pname = platforms[p].getInfo<CL_PLATFORM_NAME>(); } catch (...) { pname = "?"; }
        if (devs.empty()) s += "  platform " + std::to_string(p) + " (" + pname + "): no devices\n";
        for (size_t d = 0; d < devs.size(); ++d) {
            // Guarded: this only runs to build an error message, which a
            // failing query must not replace with an error of its own.
            std::string dname = "?", dtype = "?";
            try {
                dname = devs[d].getInfo<CL_DEVICE_NAME>();
                cl_device_type t = devs[d].getInfo<CL_DEVICE_TYPE>();
                dtype = (t & CL_DEVICE_TYPE_GPU) ? "GPU" : (t & CL_DEVICE_TYPE_CPU) ? "CPU" : "other";
            } catch (...) {}
            s += "  " + std::to_string(p) + ":" + std::to_string(d) + "  " + dname + " (" + pname + ", " + dtype + ")\n";
        }
    }
    return s;
}

// Finds the device to run on. By default, the first GPU it can find:
// checks every installed OpenCL driver in turn (there can be more than
// one, e.g. Intel and NVIDIA both installed), skipping any that error
// out. Setting OPENCL_DEVICE=<platform>:<device> (e.g. OPENCL_DEVICE=1:0)
// picks one explicitly instead, of any type, for machines with several
// GPUs or none; the indexes are listed in the error message below.
inline cl::Device pickGPU() {
    std::vector<cl::Platform> platforms;
    // With no OpenCL driver (ICD) installed at all, the loader reports an
    // error here instead of an empty list; turn that into a message that
    // says what is actually missing.
    try { cl::Platform::get(&platforms); } catch (...) { platforms.clear(); }
    if (platforms.empty())
        throw std::runtime_error("No OpenCL platform found: no GPU OpenCL driver (ICD) is installed. "
                                 "Check with `clinfo -l`.");

    if (const char* sel = std::getenv("OPENCL_DEVICE"); sel && *sel) {
        unsigned p = 0, d = 0;
        char extra = 0;
        if (std::sscanf(sel, "%u:%u%c", &p, &d, &extra) != 2)
            throw std::runtime_error(std::string("OPENCL_DEVICE='") + sel + "' is not <platform>:<device>. Devices:\n"
                                     + describeDevices(platforms));
        std::vector<cl::Device> devs;
        if (p < platforms.size())
            try { platforms[p].getDevices(CL_DEVICE_TYPE_ALL, &devs); } catch (...) {}
        if (d >= devs.size())
            throw std::runtime_error(std::string("OPENCL_DEVICE='") + sel + "' does not exist. Devices:\n"
                                     + describeDevices(platforms));
        return devs[d];
    }

    for (auto& p : platforms) {
        std::vector<cl::Device> devs;
        try { p.getDevices(CL_DEVICE_TYPE_GPU, &devs); } catch (...) { continue; }
        if (!devs.empty()) return devs[0];
    }
    throw std::runtime_error("No GPU device found. OpenCL devices on this machine:\n" + describeDevices(platforms)
                             + "Pick one with OPENCL_DEVICE=<platform>:<device>, or use backend=\"cpu\".");
}

// The largest power-of-two work-group size (at most `cap`) this kernel
// can actually be launched with on dev. The device's own maximum isn't
// enough: a kernel that needs many registers can have a lower limit
// (CL_KERNEL_WORK_GROUP_SIZE), and launching above it fails with
// CL_INVALID_WORK_GROUP_SIZE. Power of two because the kernels' tree
// reductions halve the group each step.
inline size_t kernelWorkGroupSize(const cl::Kernel& kernel, const cl::Device& dev, size_t cap = 256) {
    size_t wg = std::min(cap, dev.getInfo<CL_DEVICE_MAX_WORK_GROUP_SIZE>());
    wg = std::min(wg, kernel.getWorkGroupInfo<CL_KERNEL_WORK_GROUP_SIZE>(dev));
    size_t p = 1;
    while (p * 2 <= wg) p *= 2;
    return p;
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
