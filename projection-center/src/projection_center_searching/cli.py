from __future__ import annotations

import argparse
from pathlib import Path

from .backends import list_opencl_devices
from .pipeline import run_pipeline, run_resample, run_search
from .models import ResampleConfig, SearchConfig


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="projection-center",
        description="OpenCL-accelerated projection center search and resampling for HDF5 cone-beam projection datasets.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    devices_parser = subparsers.add_parser("devices", help="List available OpenCL devices.")
    devices_parser.set_defaults(handler=_handle_devices)

    search_parser = subparsers.add_parser("search", help="Search the projection center.")
    _add_common_runtime_args(search_parser)
    _add_search_args(search_parser)
    search_parser.add_argument("--data", required=True, type=Path, help="Input HDF5 file.")
    search_parser.add_argument("--output-pose", default=Path("real_cb_pose.json"), type=Path, help="Output pose JSON file.")
    search_parser.set_defaults(handler=_handle_search)

    resample_parser = subparsers.add_parser("resample", help="Resample the projection stack with a center pose.")
    _add_common_runtime_args(resample_parser)
    resample_parser.add_argument("--data", required=True, type=Path, help="Input HDF5 file.")
    resample_parser.add_argument("--pose", default=Path("real_cb_pose.json"), type=Path, help="Input pose JSON file.")
    resample_parser.add_argument("--output-data", default=Path("projs_resample.hdf5"), type=Path, help="Output HDF5 file.")
    resample_parser.add_argument("--downsample", default=1, type=int, help="Detector downsample factor.")
    resample_parser.add_argument("--batch-size", default=16, type=int, help="Number of projections to process per GPU batch.")
    resample_parser.set_defaults(handler=_handle_resample)

    pipeline_parser = subparsers.add_parser("pipeline", help="Run search and resampling end-to-end.")
    _add_common_runtime_args(pipeline_parser)
    _add_search_args(pipeline_parser)
    pipeline_parser.add_argument("--data", required=True, type=Path, help="Input HDF5 file.")
    pipeline_parser.add_argument("--output-pose", default=Path("real_cb_pose.json"), type=Path, help="Output pose JSON file.")
    pipeline_parser.add_argument("--output-data", default=Path("projs_resample.hdf5"), type=Path, help="Output HDF5 file.")
    pipeline_parser.add_argument("--downsample", default=1, type=int, help="Detector downsample factor.")
    pipeline_parser.add_argument("--batch-size", default=16, type=int, help="Number of projections to process per GPU batch.")
    pipeline_parser.set_defaults(handler=_handle_pipeline)

    return parser


def _add_common_runtime_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--backend", choices=("opencl", "cpu", "cpp"), default="opencl",
        help="Execution backend. 'cpp' delegates both search and resample to this repo's "
             "C++/OpenCL forward_search/ implementation (forward_search / "
             "forward_search_resample binaries) instead of this package's own kernels.",
    )
    parser.add_argument("--platform-index", type=int, default=None, help="OpenCL platform index (opencl backend only).")
    parser.add_argument("--device-index", type=int, default=None, help="OpenCL device index (opencl backend only).")
    parser.add_argument(
        "--cpp-mode", choices=("image", "buffer"), default="buffer",
        help="cpp backend only: OpenCL sinogram format (forward_search/'s own --mode). "
             "Default 'buffer' avoids this project's known Image2D driver bug -- see "
             "docs/ARCHITECTURE.md §7.",
    )


def _add_search_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--xshift", default=40.0, type=float, help="Search half-range for xshift in mm.")
    parser.add_argument("--alpha", default=10.0, type=float, help="Search half-range for alpha in degrees.")
    parser.add_argument("--beta", default=10.0, type=float, help="Search half-range for beta in degrees.")
    parser.add_argument("--xshift-step", default=1.0, type=float, help="Step for xshift in mm.")
    parser.add_argument("--alpha-step", default=1.0, type=float, help="Step for alpha in degrees.")
    parser.add_argument("--beta-step", default=1.0, type=float, help="Step for beta in degrees.")
    parser.add_argument("--sample-count", default=1000, type=int, help="Number of forward-search angle samples per parameter.")
    parser.add_argument("--sample-angle-range", default=30.0, type=float, help="Angle span in degrees for the forward search samples.")


def _build_search_config(args: argparse.Namespace) -> SearchConfig:
    return SearchConfig(
        xshift_range_mm=args.xshift,
        alpha_range_deg=args.alpha,
        beta_range_deg=args.beta,
        xshift_step_mm=args.xshift_step,
        alpha_step_deg=args.alpha_step,
        beta_step_deg=args.beta_step,
        sample_count=args.sample_count,
        sample_angle_range_deg=args.sample_angle_range,
    )


def _build_resample_config(args: argparse.Namespace) -> ResampleConfig:
    return ResampleConfig(
        downsample_factor=args.downsample,
        batch_size=args.batch_size,
    )


def _handle_devices(args: argparse.Namespace) -> int:
    devices = list_opencl_devices()
    if not devices:
        print("No OpenCL devices found. Install an OpenCL runtime or use --backend cpu.")
        return 1
    for line in devices:
        print(line)
    return 0


def _handle_search(args: argparse.Namespace) -> int:
    result = run_search(
        data_path=args.data,
        output_pose_path=args.output_pose,
        search_config=_build_search_config(args),
        backend_name=args.backend,
        platform_index=args.platform_index,
        device_index=args.device_index,
        cpp_mode=args.cpp_mode,
    )
    print(f"Best MSE: {result.mse:.8f}")
    print(f"xshift (mm): {result.xshift * 1000.0:.6f}")
    print(f"alpha (deg): {result.alpha * 180.0 / 3.141592653589793:.6f}")
    print(f"beta (deg): {result.beta * 180.0 / 3.141592653589793:.6f}")
    print(f"center_point: ({result.center_point[0]:.6f}, {result.center_point[1]:.6f})")
    print(f"pose json: {args.output_pose}")
    return 0


def _handle_resample(args: argparse.Namespace) -> int:
    pose, output_path = run_resample(
        data_path=args.data,
        pose_path=args.pose,
        output_data_path=args.output_data,
        resample_config=_build_resample_config(args),
        backend_name=args.backend,
        platform_index=args.platform_index,
        device_index=args.device_index,
        cpp_mode=args.cpp_mode,
    )
    print(f"Resampled using pose MSE: {pose.mse:.8f}")
    print(f"output hdf5: {output_path}")
    return 0


def _handle_pipeline(args: argparse.Namespace) -> int:
    result, output_path = run_pipeline(
        data_path=args.data,
        output_pose_path=args.output_pose,
        output_data_path=args.output_data,
        search_config=_build_search_config(args),
        resample_config=_build_resample_config(args),
        backend_name=args.backend,
        platform_index=args.platform_index,
        device_index=args.device_index,
        cpp_mode=args.cpp_mode,
    )
    print(f"Best MSE: {result.mse:.8f}")
    print(f"output pose: {args.output_pose}")
    print(f"output hdf5: {output_path}")
    return 0


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    raise SystemExit(args.handler(args))
