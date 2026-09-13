"""Regression test: --backend hybrid must search exactly like --backend cpp
(HybridBackend.search_from_file() delegates to its own CppBackend) and
resample exactly like --backend opencl (HybridBackend.resample() delegates
to its own OpenCLBackend), not some third, independent implementation."""
import sys

sys.path.insert(0, "projection-center/src")
from projection_center_searching.pipeline import run_pipeline
from projection_center_searching.models import SearchConfig, ResampleConfig

import h5py
import numpy as np

DATA = "data/proj_shepplogan128.hdf5"


def test_hybrid_search_matches_cpp_search():
    cpp_result, _ = run_pipeline(
        DATA, "/tmp/hybrid_test_cpp_pose.json", "/tmp/hybrid_test_cpp_resampled.hdf5",
        SearchConfig(), ResampleConfig(), backend_name="cpp",
    )
    hybrid_result, _ = run_pipeline(
        DATA, "/tmp/hybrid_test_hybrid_pose.json", "/tmp/hybrid_test_hybrid_resampled.hdf5",
        SearchConfig(), ResampleConfig(), backend_name="hybrid",
    )
    assert cpp_result.xshift == hybrid_result.xshift
    assert cpp_result.alpha == hybrid_result.alpha
    assert cpp_result.beta == hybrid_result.beta
    assert cpp_result.mse == hybrid_result.mse
    print(f"test_hybrid_search_matches_cpp_search: PASS  {hybrid_result}")


def test_hybrid_resample_matches_opencl_resample():
    # Same pose (from the hybrid run above) fed into a pure opencl resample
    # must produce byte-identical output to hybrid's own resample step,
    # since hybrid's resample() is literally OpenCLBackend.resample().
    from projection_center_searching.pipeline import run_resample
    run_resample(
        DATA, "/tmp/hybrid_test_hybrid_pose.json", "/tmp/hybrid_test_opencl_only_resampled.hdf5",
        ResampleConfig(), backend_name="opencl",
    )
    with h5py.File("/tmp/hybrid_test_hybrid_resampled.hdf5") as f:
        hybrid_projs = f["Projection"][()]
    with h5py.File("/tmp/hybrid_test_opencl_only_resampled.hdf5") as f:
        opencl_projs = f["Projection"][()]
    assert np.array_equal(hybrid_projs, opencl_projs)
    print("test_hybrid_resample_matches_opencl_resample: PASS")


if __name__ == "__main__":
    test_hybrid_search_matches_cpp_search()
    test_hybrid_resample_matches_opencl_resample()
