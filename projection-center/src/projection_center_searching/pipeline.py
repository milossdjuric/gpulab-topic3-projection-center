from __future__ import annotations

from pathlib import Path

import numpy as np

from .backends import get_backend
from .geometry import (
    build_parameter_grid,
    build_sinogram,
    compute_forward_geometry,
    compute_resample_geometry,
)
from .hdf5_io import (
    build_sinogram_from_hdf5,
    initialize_resampled_dataset,
    load_metadata,
    read_pose_json,
    stream_projection_batches,
    write_pose_json,
    write_projection_batch,
)
from .models import ResampleConfig, SearchConfig, SearchResult


def run_search(
    data_path: str | Path,
    output_pose_path: str | Path,
    search_config: SearchConfig,
    backend_name: str = "opencl",
    platform_index: int | None = None,
    device_index: int | None = None,
    cpp_mode: str = "buffer",
) -> SearchResult:
    backend = get_backend(backend_name, platform_index=platform_index, device_index=device_index, cpp_mode=cpp_mode)

    if backend_name in ("cpp", "hybrid"):
        # forward_search/ does its own HDF5 loading, sinogram build, and grid
        # search internally -- it isn't a drop-in kernel swap for the shared
        # sinogram/parameter_grid interface below, so it gets the raw file
        # path instead. hybrid's search is cpp's search (HybridBackend
        # delegates search_from_file() to its own CppBackend); only
        # hybrid's resample differs from plain cpp, handled in run_resample().
        result = backend.search_from_file(data_path, search_config)
        write_pose_json(output_pose_path, result)
        return result

    cb_params, sinogram_sum = build_sinogram_from_hdf5(data_path)
    sinogram = build_sinogram(sinogram_sum)
    parameter_grid = build_parameter_grid(search_config)
    x0, y0, tan_theta0 = compute_forward_geometry(cb_params, parameter_grid)

    artifacts = backend.search(
        sinogram=sinogram,
        alpha=parameter_grid[:, 1],
        beta=parameter_grid[:, 2],
        tan_theta0=tan_theta0,
        x0=x0,
        y0=y0,
        cb_params=cb_params,
        config=search_config,
    )
    best_index = int(np.argmin(artifacts.mse_values))
    result = SearchResult(
        center_point=(float(artifacts.x0[best_index]), float(artifacts.y0[best_index])),
        xshift=float(parameter_grid[best_index, 0]),
        alpha=float(parameter_grid[best_index, 1]),
        beta=float(parameter_grid[best_index, 2]),
        mse=float(artifacts.mse_values[best_index]),
        kernel_ms=artifacts.kernel_ms,
    )
    write_pose_json(output_pose_path, result)
    return result


def run_resample(
    data_path: str | Path,
    pose_path: str | Path,
    output_data_path: str | Path,
    resample_config: ResampleConfig,
    backend_name: str = "opencl",
    platform_index: int | None = None,
    device_index: int | None = None,
    cpp_mode: str = "buffer",
) -> tuple[SearchResult, Path]:
    if backend_name == "cpp":
        # forward_search_resample does its own HDF5 loading, pose parsing,
        # and geometry/resample computation internally -- like CppBackend's
        # search_from_file(), it gets the raw file paths instead of the
        # shared rotation_matrix/output_params path the other two backends use.
        backend = get_backend(backend_name, cpp_mode=cpp_mode)
        backend.resample_from_file(data_path, pose_path, output_data_path, resample_config)
        pose = read_pose_json(pose_path)
        return pose, Path(output_data_path)

    cb_params = load_metadata(data_path)
    pose = read_pose_json(pose_path)
    output_params, rotation_matrix = compute_resample_geometry(
        cb_params,
        pose,
        resample_config.downsample_factor,
    )
    backend = get_backend(backend_name, platform_index=platform_index, device_index=device_index, cpp_mode=cpp_mode)
    initialize_resampled_dataset(output_data_path, output_params)
    for start, batch in stream_projection_batches(data_path, resample_config.batch_size):
        resampled = backend.resample(
            projections=batch,
            rotation_matrix=rotation_matrix,
            input_params=cb_params,
            output_params=output_params,
            batch_size=resample_config.batch_size,
        )
        write_projection_batch(output_data_path, start, resampled)
    return pose, Path(output_data_path)


def run_pipeline(
    data_path: str | Path,
    output_pose_path: str | Path,
    output_data_path: str | Path,
    search_config: SearchConfig,
    resample_config: ResampleConfig,
    backend_name: str = "opencl",
    platform_index: int | None = None,
    device_index: int | None = None,
    cpp_mode: str = "buffer",
) -> tuple[SearchResult, Path]:
    if backend_name == "cpp":
        # forward_search_pipeline runs search+resample in one process,
        # instead of run_search()+run_resample()'s two separate cpp
        # subprocess calls round-tripping the pose through a JSON file --
        # see CppBackend.pipeline_from_file().
        backend = get_backend(backend_name, cpp_mode=cpp_mode)
        result = backend.pipeline_from_file(data_path, output_data_path, search_config, resample_config)
        write_pose_json(output_pose_path, result)
        return result, Path(output_data_path)

    result = run_search(
        data_path=data_path,
        output_pose_path=output_pose_path,
        search_config=search_config,
        backend_name=backend_name,
        platform_index=platform_index,
        device_index=device_index,
        cpp_mode=cpp_mode,
    )
    _, output_path = run_resample(
        data_path=data_path,
        pose_path=output_pose_path,
        output_data_path=output_data_path,
        resample_config=resample_config,
        backend_name=backend_name,
        platform_index=platform_index,
        device_index=device_index,
        cpp_mode=cpp_mode,
    )
    return result, output_path
