// The full pipeline command line tool, projection_center_pipeline_cpp.
// Does both stages in one program: loads the file once, runs search then
// resample in the same process, and hands the found pose straight over in
// memory instead of writing it to a file and reading it back.
//
// Running `forward_search` then `forward_search_resample` separately does
// not get you this: each one is its own program, so each pays its own
// startup cost (loading the file, picking the GPU, compiling the kernel)
// all over again. This binary still compiles two separate kernels
// internally, one for search and one for resample, so that part is not
// saved, only the doubled startup cost is.
#include <chrono>
#include <iostream>
#include <stdexcept>
#include <vector>
#include <cmath>
#include <filesystem>
#include <H5Cpp.h>
#include <boost/program_options.hpp>
#include "forward_search.hpp"
#include "resample.hpp"
#include "stage_timing.hpp"

namespace po = boost::program_options;

// Reads one dataset out of its HDF5 file, same as the other two binaries'
// loadHDF5(). Copied here instead of shared, since each CLI binary in
// this project stands on its own.
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

// Writes one plain number into an already-open HDF5 file. Saves
// writePoseHDF5() below from repeating the same lines for every field.
static void writeScalar(H5::H5File& file, const std::string& name, double v) {
    H5::DataSpace space(H5S_SCALAR);
    file.createDataSet(name, H5::PredType::NATIVE_DOUBLE, space).write(&v, H5::PredType::NATIVE_DOUBLE);
}

// Writes the pose search just found to a file.
static void writePoseHDF5(const std::string& path, const CbPose& pose) {
    std::filesystem::path fpath(path);
    if (fpath.has_parent_path())
        std::filesystem::create_directories(fpath.parent_path());

    H5::H5File file(path, H5F_ACC_TRUNC);
    writeScalar(file, "xshift", pose.xshift);
    writeScalar(file, "alpha",  pose.alpha);
    writeScalar(file, "beta",   pose.beta);
    writeScalar(file, "MSE",    pose.mse);
    writeScalar(file, "kernel_ms", pose.kernel_ms);

    double center[2] = {pose.center_x, pose.center_y};
    hsize_t dim[1] = {2};
    H5::DataSpace space(1, dim);
    file.createDataSet("center_point", H5::PredType::NATIVE_DOUBLE, space)
        .write(center, H5::PredType::NATIVE_DOUBLE);
}

// Writes the corrected images and geometry that resample just produced to
// a file, same field names every implementation in this project uses.
static void writeResampledHDF5(const std::string& path, const CbPara& para, const std::vector<float>& projs) {
    std::filesystem::path fpath(path);
    if (fpath.has_parent_path())
        std::filesystem::create_directories(fpath.parent_path());

    H5::H5File file(path, H5F_ACC_TRUNC);
    H5::DataSpace scalar(H5S_SCALAR);

    auto ws = [&](const std::string& name, double v) {
        file.createDataSet(name, H5::PredType::NATIVE_DOUBLE, scalar).write(&v, H5::PredType::NATIVE_DOUBLE);
    };
    ws("pixelSize", para.pixel_size);
    ws("SDD", para.SDD);
    ws("SOD", para.SOD);
    ws("voxelSize", para.voxel_size);
    ws("Volumen_num_xz", para.volumen_num_xz);
    ws("Volumen_num_y", para.volumen_num_y);
    ws("num_projs", para.num_projs);
    ws("detector_width", para.detector_width);
    ws("detector_height", para.detector_height);

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

// Reads the command line flags, loads the dataset once, runs search, then
// runs resample right on the pose search just found, no file in between.
// Writes both a pose file and a corrected-images file, and prints three
// timings: search alone, resample alone, and the grand total.
int main(int argc, char* argv[]) {
    po::options_description desc("Options");
    desc.add_options()
        ("data",           po::value<std::string>()->required(), "HDF5 input")
        ("xshift",         po::value<double>()->default_value(40.0), "Range of xshift (mm)")
        ("alpha",          po::value<double>()->default_value(10.0), "Range of alpha (deg)")
        ("beta",           po::value<double>()->default_value(10.0), "Range of beta (deg)")
        ("xshift-step",    po::value<double>()->default_value(1.0),  "Step of xshift (mm)")
        ("alpha-step",     po::value<double>()->default_value(1.0),  "Step of alpha (deg)")
        ("beta-step",      po::value<double>()->default_value(1.0),  "Step of beta (deg)")
        ("mode",           po::value<std::string>()->default_value("buffer"),
                               "Search sinogram data format: image (Image2D+sampler) or buffer (manual bilinear)")
        ("search-kernel",  po::value<std::string>()->default_value("kernels/forward_search.cl"), "Search kernel file path")
        ("resample-kernel",po::value<std::string>()->default_value("kernels/resample.cl"), "Resample kernel file path")
        ("downsample",     po::value<int>()->default_value(1), "Detector downsample factor")
        ("batch-size",     po::value<int>()->default_value(16), "Projections per GPU batch")
        ("output-pose",    po::value<std::string>()->default_value("runs/pipeline_pose.h5"), "Output pose HDF5 path")
        ("output-data",    po::value<std::string>()->default_value("runs/pipeline_resampled.hdf5"), "Output resampled HDF5 path");

    po::variables_map vm;
    po::store(po::parse_command_line(argc, argv, desc), vm);
    po::notify(vm);

    try {
        auto t_load0 = std::chrono::steady_clock::now();
        std::vector<float> projs;
        CbPara para = loadHDF5(vm["data"].as<std::string>(), projs);
        auto t_load1 = std::chrono::steady_clock::now();
        std::cerr << "Loaded " << para.num_projs << " projections, "
                  << para.detector_width << "x" << para.detector_height
                  << " (" << std::chrono::duration_cast<std::chrono::milliseconds>(t_load1 - t_load0).count() << "ms)\n";

        SearchArgs args;
        args.xshift      = vm["xshift"].as<double>()      / 1000.0;
        args.alpha       = vm["alpha"].as<double>()       / 180.0 * PI;
        args.beta        = vm["beta"].as<double>()        / 180.0 * PI;
        args.xshift_step = vm["xshift-step"].as<double>() / 1000.0;
        args.alpha_step  = vm["alpha-step"].as<double>()  / 180.0 * PI;
        args.beta_step   = vm["beta-step"].as<double>()   / 180.0 * PI;

        CbPose pose = computeCOR(para, projs, args,
                                 vm["search-kernel"].as<std::string>(),
                                 vm["mode"].as<std::string>());
        auto t_search_done = std::chrono::steady_clock::now();

        auto t_write0 = std::chrono::steady_clock::now();
        writePoseHDF5(vm["output-pose"].as<std::string>(), pose);
        double write_ms = msSince(t_write0);
        std::cerr << "MSE:    " << pose.mse << "\n";
        std::cerr << "xshift: " << pose.xshift * 1000.0 << " mm\n";
        std::cerr << "alpha:  " << pose.alpha / PI * 180.0 << " deg\n";
        std::cerr << "beta:   " << pose.beta  / PI * 180.0 << " deg\n";
        std::cerr << "search kernel: " << pose.kernel_ms << " ms\n";
        std::cerr << "search total: "
                  << std::chrono::duration_cast<std::chrono::milliseconds>(t_search_done - t_load1).count()
                  << " ms (device pick, context, sinogram build/upload, kernel compile, and the kernel itself)\n";

        // Hand the pose straight to resample, no file needed here since
        // both stages run in the same process.
        ResamplePose rpose;
        rpose.center_x = pose.center_x;
        rpose.center_y = pose.center_y;
        rpose.xshift    = pose.xshift;
        rpose.alpha     = pose.alpha;
        rpose.beta      = pose.beta;

        auto t_resample0 = std::chrono::steady_clock::now();
        ResampleOutput result = computeResample(para, projs, rpose,
                                                vm["downsample"].as<int>(),
                                                vm["resample-kernel"].as<std::string>(),
                                                vm["batch-size"].as<int>());
        auto t_resample_done = std::chrono::steady_clock::now();
        double compute_ms = std::chrono::duration<double, std::milli>(t_search_done - t_load1).count()
                          + std::chrono::duration<double, std::milli>(t_resample_done - t_resample0).count();
        std::cerr << "resample total: "
                  << std::chrono::duration_cast<std::chrono::milliseconds>(t_resample_done - t_search_done).count()
                  << " ms (device pick, context, kernel compile, and every batch)\n";

        auto t_write1 = std::chrono::steady_clock::now();
        writeResampledHDF5(vm["output-data"].as<std::string>(), result.para, result.projs);
        write_ms += msSince(t_write1);
        auto t_done = std::chrono::steady_clock::now();
        std::cerr << "Wrote " << vm["output-data"].as<std::string>() << "\n";
        std::cerr << "total (search + resample): "
                  << std::chrono::duration_cast<std::chrono::milliseconds>(t_done - t_load1).count()
                  << " ms\n";
        printTimingLine(std::chrono::duration<double, std::milli>(t_load1 - t_load0).count(),
                        compute_ms, write_ms);

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }
    return 0;
}
