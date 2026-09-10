#include <chrono>
#include <iostream>
#include <stdexcept>
#include <vector>
#include <cmath>
#include <filesystem>
#include <H5Cpp.h>
#include <boost/program_options.hpp>
#include "forward_search.hpp"

namespace po = boost::program_options;

static CbPara loadHDF5(const std::string& path, std::vector<float>& projs) {
    H5::H5File file(path, H5F_ACC_RDONLY);

    auto readDouble = [&](const std::string& name) {
        double v;
        file.openDataSet(name).read(&v, H5::PredType::NATIVE_DOUBLE);
        return v;
    };
    // Source HDF5 stores scalar ints (num_projs, detector_width, ...) as
    // float64 too, matching how the Python reference's h5py writer saved them.
    auto readInt = [&](const std::string& name) {
        double v;
        file.openDataSet(name).read(&v, H5::PredType::NATIVE_DOUBLE);
        return static_cast<int>(v);
    };

    H5::DataSet p_ds = file.openDataSet("Projection");
    hsize_t dims[3];
    p_ds.getSpace().getSimpleExtentDims(dims);

    CbPara p;
    p.SDD             = readDouble("SDD");
    p.SOD             = readDouble("SOD");
    p.pixel_size      = readDouble("pixelSize");
    p.voxel_size      = readDouble("voxelSize");
    p.num_projs       = readInt("num_projs");
    // Derived from Projection's real shape (num_projs, height, width), not
    // trusted from the file's detector_width/detector_height scalars:
    // some dataset files store those two swapped relative to the actual
    // array, which used to silently read past the real data.
    p.detector_height = static_cast<int>(dims[1]);
    p.detector_width  = static_cast<int>(dims[2]);
    p.volumen_num_xz  = readInt("Volumen_num_xz");
    p.volumen_num_y   = readInt("Volumen_num_y");

    H5::DataSet a_ds = file.openDataSet("Angle");
    hsize_t a_dim[1];
    a_ds.getSpace().getSimpleExtentDims(a_dim);
    p.angles.resize(a_dim[0]);
    a_ds.read(p.angles.data(), H5::PredType::NATIVE_DOUBLE);

    projs.resize(dims[0] * dims[1] * dims[2]);
    p_ds.read(projs.data(), H5::PredType::NATIVE_FLOAT);

    return p;
}

static void writeScalar(H5::H5File& file, const std::string& name, double v) {
    H5::DataSpace space(H5S_SCALAR);
    file.createDataSet(name, H5::PredType::NATIVE_DOUBLE, space).write(&v, H5::PredType::NATIVE_DOUBLE);
}

static void writeHDF5(const std::string& path, const CbPose& pose) {
    std::filesystem::path fpath(path);
    if (fpath.has_parent_path())
        std::filesystem::create_directories(fpath.parent_path());

    H5::H5File file(path, H5F_ACC_TRUNC);

    writeScalar(file, "xshift", pose.xshift);
    writeScalar(file, "alpha",  pose.alpha);
    writeScalar(file, "beta",   pose.beta);
    writeScalar(file, "MSE",    pose.mse);

    double center[2] = {pose.center_x, pose.center_y};
    hsize_t dim[1] = {2};
    H5::DataSpace space(1, dim);
    file.createDataSet("center_point", H5::PredType::NATIVE_DOUBLE, space)
        .write(center, H5::PredType::NATIVE_DOUBLE);
}

int main(int argc, char* argv[]) {
    po::options_description desc("Options");
    desc.add_options()
        ("data",        po::value<std::string>()->default_value(
                            "/lgrp/edu-2026-1-gpulab/projs_change.hdf5"), "HDF5 input")
        ("xshift",      po::value<double>()->default_value(40.0), "Range of xshift (mm)")
        ("alpha",       po::value<double>()->default_value(10.0), "Range of alpha (deg)")
        ("beta",        po::value<double>()->default_value(10.0), "Range of beta (deg)")
        ("xshift-step", po::value<double>()->default_value(1.0),  "Step of xshift (mm)")
        ("alpha-step",  po::value<double>()->default_value(1.0),  "Step of alpha (deg)")
        ("beta-step",   po::value<double>()->default_value(1.0),  "Step of beta (deg)")
        ("kernel",      po::value<std::string>()->default_value(
                            "kernels/forward_search.cl"), "Kernel file path")
        ("mode",        po::value<std::string>()->default_value("image"),
                            "Sinogram data format: image (Image2D+sampler) or buffer (manual bilinear)")
        ("output",      po::value<std::string>()->default_value(
                            "runs/real_cb_pose.h5"), "Output HDF5 path");

    po::variables_map vm;
    po::store(po::parse_command_line(argc, argv, desc), vm);
    po::notify(vm);

    try {
        auto t_load0 = std::chrono::steady_clock::now();
        std::vector<float> projs;
        CbPara para = loadHDF5(vm["data"].as<std::string>(), projs);
        auto t_load1 = std::chrono::steady_clock::now();
        auto load_ms = std::chrono::duration_cast<std::chrono::milliseconds>(t_load1 - t_load0).count();
        std::cerr << "Loaded " << para.num_projs << " projections, "
                  << para.detector_width << "x" << para.detector_height
                  << " (" << load_ms << "ms)\n";

        SearchArgs args;
        args.xshift      = vm["xshift"].as<double>()      / 1000.0;
        args.alpha       = vm["alpha"].as<double>()       / 180.0 * M_PI;
        args.beta        = vm["beta"].as<double>()        / 180.0 * M_PI;
        args.xshift_step = vm["xshift-step"].as<double>() / 1000.0;
        args.alpha_step  = vm["alpha-step"].as<double>()  / 180.0 * M_PI;
        args.beta_step   = vm["beta-step"].as<double>()   / 180.0 * M_PI;

        CbPose pose = computeCOR(para, projs, args,
                                 vm["kernel"].as<std::string>(),
                                 vm["mode"].as<std::string>());

        writeHDF5(vm["output"].as<std::string>(), pose);
        std::cerr << "MSE:    " << pose.mse << "\n";
        std::cerr << "xshift: " << pose.xshift * 1000.0 << " mm\n";
        std::cerr << "alpha:  " << pose.alpha / M_PI * 180.0 << " deg\n";
        std::cerr << "beta:   " << pose.beta  / M_PI * 180.0 << " deg\n";

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }
    return 0;
}
