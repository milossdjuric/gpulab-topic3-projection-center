// The resample-only command line tool. Reads a dataset plus a pose file
// from an earlier search, runs the GPU resample once (computeResample(),
// in resample.cpp), and writes the corrected images to a file. Does not
// run search itself, it expects the pose to already exist. See main.cpp
// for search, or pipeline_main.cpp for both together.
#include <chrono>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <vector>
#include <cmath>
#include <filesystem>
#include <H5Cpp.h>
#include <boost/program_options.hpp>
#include <boost/json.hpp>
#include "forward_search.hpp"
#include "resample.hpp"

namespace po = boost::program_options;
namespace json = boost::json;

// Reads one dataset out of its HDF5 file, same as main.cpp's loadHDF5().
// Copied here instead of shared, since each CLI binary in this project
// stands on its own.
static CbPara loadHDF5(const std::string& path, std::vector<float>& projs) {
    H5::H5File file(path, H5F_ACC_RDONLY);

    auto readDouble = [&](const std::string& name) {
        double v;
        file.openDataSet(name).read(&v, H5::PredType::NATIVE_DOUBLE);
        return v;
    };
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
    // Derived from Projection's real shape (num_projs, height, width),
    // not trusted from the file's detector_width/detector_height scalars.
    // Some dataset files store those two swapped relative to the actual
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

// Reads a pose file, whichever of the two implementations wrote it. Both
// save the same shape, so one parser can read either.
static ResamplePose readPoseJSON(const std::string& path) {
    std::ifstream f(path);
    if (!f) throw std::runtime_error("Cannot open pose file: " + path);
    std::stringstream ss;
    ss << f.rdbuf();

    json::value v = json::parse(ss.str());
    const json::object& obj = v.as_object();
    const json::array& center = obj.at("center_point").as_array();

    ResamplePose pose;
    pose.center_x = center.at(0).to_number<double>();
    pose.center_y = center.at(1).to_number<double>();
    pose.xshift   = obj.at("xshift").to_number<double>();
    pose.alpha    = obj.at("alpha").to_number<double>();
    pose.beta     = obj.at("beta").to_number<double>();
    return pose;
}

// Writes the corrected images and geometry to an HDF5 file, same field
// names every other implementation uses, so any of them can read any
// other's output.
static void writeResampledHDF5(const std::string& path, const CbPara& para, const std::vector<float>& projs) {
    std::filesystem::path fpath(path);
    if (fpath.has_parent_path())
        std::filesystem::create_directories(fpath.parent_path());

    H5::H5File file(path, H5F_ACC_TRUNC);
    H5::DataSpace scalar(H5S_SCALAR);

    auto writeScalar = [&](const std::string& name, double v) {
        file.createDataSet(name, H5::PredType::NATIVE_DOUBLE, scalar).write(&v, H5::PredType::NATIVE_DOUBLE);
    };
    writeScalar("pixelSize", para.pixel_size);
    writeScalar("SDD", para.SDD);
    writeScalar("SOD", para.SOD);
    writeScalar("voxelSize", para.voxel_size);
    writeScalar("Volumen_num_xz", para.volumen_num_xz);
    writeScalar("Volumen_num_y", para.volumen_num_y);
    writeScalar("num_projs", para.num_projs);
    writeScalar("detector_width", para.detector_width);
    writeScalar("detector_height", para.detector_height);

    hsize_t a_dim[1] = { para.angles.size() };
    H5::DataSpace a_space(1, a_dim);
    file.createDataSet("Angle", H5::PredType::NATIVE_DOUBLE, a_space)
        .write(para.angles.data(), H5::PredType::NATIVE_DOUBLE);

    hsize_t p_dim[3] = { static_cast<hsize_t>(para.num_projs),
                         static_cast<hsize_t>(para.detector_height),
                         static_cast<hsize_t>(para.detector_width) };
    H5::DataSpace p_space(3, p_dim);
    file.createDataSet("Projection", H5::PredType::NATIVE_FLOAT, p_space)
        .write(projs.data(), H5::PredType::NATIVE_FLOAT);
}

// Reads the command line flags, loads the dataset and the pose, runs
// resample, and writes the corrected images. All the real GPU work
// happens inside computeResample(), this is just the setup around it.
int main(int argc, char* argv[]) {
    po::options_description desc("Options");
    desc.add_options()
        ("data",       po::value<std::string>()->required(), "Original HDF5 input (same file forward search ran on)")
        ("pose",       po::value<std::string>()->default_value("real_cb_pose.json"), "Pose JSON from either forward-search implementation")
        ("output",     po::value<std::string>()->default_value("runs/projs_resample.h5"), "Output HDF5 path")
        ("downsample", po::value<int>()->default_value(1), "Detector downsample factor")
        ("batch-size", po::value<int>()->default_value(16), "Projections per GPU batch")
        ("kernel",     po::value<std::string>()->default_value("kernels/resample.cl"), "Kernel file path");

    po::variables_map vm;
    po::store(po::parse_command_line(argc, argv, desc), vm);
    po::notify(vm);

    try {
        auto t0 = std::chrono::steady_clock::now();
        std::vector<float> projs;
        CbPara para = loadHDF5(vm["data"].as<std::string>(), projs);
        auto t1 = std::chrono::steady_clock::now();
        std::cerr << "Loaded " << para.num_projs << " projections, "
                  << para.detector_width << "x" << para.detector_height
                  << " (" << std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count() << "ms)\n";

        ResamplePose pose = readPoseJSON(vm["pose"].as<std::string>());

        ResampleOutput result = computeResample(para, projs, pose,
                                                vm["downsample"].as<int>(),
                                                vm["kernel"].as<std::string>(),
                                                vm["batch-size"].as<int>());

        writeResampledHDF5(vm["output"].as<std::string>(), result.para, result.projs);
        auto t2 = std::chrono::steady_clock::now();
        std::cerr << "Wrote " << vm["output"].as<std::string>()
                  << " (" << std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count() << "ms total)\n";
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }
    return 0;
}
