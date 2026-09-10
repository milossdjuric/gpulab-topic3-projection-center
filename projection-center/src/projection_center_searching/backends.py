from __future__ import annotations

import os
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .models import ConeBeamParameters, ResampleConfig, SearchConfig, SearchResult

try:
    import pyopencl as cl
except ImportError:  # pragma: no cover
    cl = None


# CORRECTION (2026-08-26): a sample pair whose BOTH rays land in background
# (near-zero on the sinogram, normalized to [0,1] by geometry.build_sinogram)
# is trivially "symmetric" (0-0=0) regardless of true alignment -- unlike a
# signal-vs-background pair, which already produces a large, informative
# diff and needs no special handling. Excluding background-only pairs from
# both CpuBackend's search() and the center_search_reduce kernel below (and
# from forward_search.cl's kernels) prevents a candidate pose from faking a
# low MSE by pushing rays into empty space around the phantom instead of
# genuinely aligning. See docs/ARCHITECTURE.md for the investigation this
# came from. Kept as a plain module constant (not per-dataset-tuned) since
# it only needs to separate "definitely background" from "definitely some
# signal" on an already-normalized [0,1] scale, not draw a precise edge.
_MIN_SIGNAL = 0.01

# A third fix (relative rather than absolute error, to counter dim-pixel
# bias) was implemented and then reverted (2026-08-26): it changed the
# already-reference-matching result on projs_change.hdf5 (beta 4deg -> 5deg,
# no longer matching Topic_3_forwardsearching.py). Fixes #1/#2 above only
# exclude comparisons that are meaningless by the reference's own logic; a
# relative-error metric optimizes a genuinely different objective and isn't
# safe to apply by default. Staying faithful to the Python reference takes
# priority -- see docs/ARCHITECTURE.md §12.4.

KERNEL_SOURCE = r"""
#define MIN_SIGNAL 0.01f

float bilinear_sample(__global const float *image, int width, int height, float x, float y)
{
    if (x < 0.0f || x >= (float)(width - 1) || y < 0.0f || y >= (float)(height - 1)) {
        return 0.0f;
    }

    int ix = (int)floor(x);
    int iy = (int)floor(y);
    float dx = x - (float)ix;
    float dy = y - (float)iy;

    int row0 = iy * width;
    int row1 = (iy + 1) * width;
    float v00 = image[row0 + ix];
    float v10 = image[row0 + ix + 1];
    float v01 = image[row1 + ix];
    float v11 = image[row1 + ix + 1];

    float top = mad(dx, v10 - v00, v00);
    float bottom = mad(dx, v11 - v01, v01);
    return mad(dy, bottom - top, top);
}

__kernel void center_search_reduce(
    __global const float *sinogram,
    __global const float *alpha_values,
    __global const float *beta_values,
    __global const float *theta0_values,
    __global float *mse_values,
    const float sdd,
    const int detector_width,
    const int detector_height,
    const float pixel_size,
    const int sample_count,
    const float angle_step)
{
    int lane = get_local_id(0);
    int workgroup_size = get_local_size(0);
    int parameter_index = get_group_id(1);

    float alpha = alpha_values[parameter_index];
    float beta = beta_values[parameter_index];
    float theta0 = theta0_values[parameter_index];

    float sin_a = sin(alpha);
    float cos_a = cos(alpha);
    float sin_b = sin(beta);
    float cos_b = cos(beta);

    float local_sum = 0.0f;
    int local_count = 0;

    for (int sample_index = lane; sample_index < sample_count; sample_index += workgroup_size) {
        float nearest_theta = angle_step * (float)sample_index;
        float theta = nearest_theta + theta0;
        float reflect_theta = -nearest_theta + theta0;

        float tan_t = tan(theta);
        float tan_rt = tan(reflect_theta);

        float temp = sdd / (tan_t * sin_a * sin_b + cos_b);
        float rtemp = sdd / (tan_rt * sin_a * sin_b + cos_b);

        float y = temp * (-tan_t * sin_a * cos_b + sin_b);
        float ry = rtemp * (-tan_rt * sin_a * cos_b + sin_b);

        float x = -temp * tan_t * cos_a;
        float rx = -rtemp * tan_rt * cos_a;

        y = ((float)(detector_height - 1) * 0.5f) - y / pixel_size;
        ry = ((float)(detector_height - 1) * 0.5f) - ry / pixel_size;
        x = x / pixel_size + ((float)(detector_width - 1) * 0.5f);
        rx = rx / pixel_size + ((float)(detector_width - 1) * 0.5f);

        // CORRECTION (2026-08-26): only count sample pairs where BOTH rays
        // land on-detector. Previously every sample contributed to local_sum
        // (via bilinear_sample's 0.0f-outside-bounds behavior) and the final
        // division was always by the fixed sample_count -- so a candidate
        // pose that pushes most/all rays off-detector got most/all of its
        // squared-diff terms silently replaced by (0-0)^2=0, diluting its
        // MSE toward zero regardless of true alignment. That makes "push
        // everything off-detector" a trivial way to fake a low MSE, which is
        // exactly what happened on a small-detector dataset (128x128, small
        // SDD/SOD) where the default search range pushes many candidates
        // off-detector -- the found "optimum" sat at the search-range
        // boundary with MSE=0.0, not at the true alignment. Normalizing by
        // the count of genuinely valid (in-bounds on both sides) pairs
        // instead removes that incentive; a candidate with zero valid pairs
        // gets no information and must not be treated as a good match.
        bool valid_t  = (x  >= 0.0f && x  < (float)(detector_width  - 1) && y  >= 0.0f && y  < (float)(detector_height - 1));
        bool valid_rt = (rx >= 0.0f && rx < (float)(detector_width  - 1) && ry >= 0.0f && ry < (float)(detector_height - 1));
        if (valid_t && valid_rt) {
            float f_theta = bilinear_sample(sinogram, detector_width, detector_height, x, y);
            float f_rtheta = bilinear_sample(sinogram, detector_width, detector_height, rx, ry);
            // CORRECTION (2026-08-26): a pair where BOTH sides are
            // background (near-zero on the [0,1]-normalized sinogram) is
            // trivially "symmetric" without reflecting real alignment --
            // unlike a signal-vs-background pair, which already produces a
            // large, informative diff. MIN_SIGNAL must match the Python
            // CpuBackend's _MIN_SIGNAL constant above.
            if (f_theta >= MIN_SIGNAL || f_rtheta >= MIN_SIGNAL) {
                float diff = f_theta - f_rtheta;
                local_sum += diff * diff;
                local_count += 1;
            }
        }
    }

    __local float scratch[256];
    __local int count_scratch[256];
    scratch[lane] = local_sum;
    count_scratch[lane] = local_count;
    barrier(CLK_LOCAL_MEM_FENCE);

    for (int stride = workgroup_size / 2; stride > 0; stride >>= 1) {
        if (lane < stride) {
            scratch[lane] += scratch[lane + stride];
            count_scratch[lane] += count_scratch[lane + stride];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (lane == 0) {
        mse_values[parameter_index] = (count_scratch[0] > 0) ? (scratch[0] / (float)count_scratch[0]) : FLT_MAX;
    }
}

__kernel void resample_projections(
    __global const float *input_projections,
    __global float *output_projections,
    __global const float *rotation_matrix,
    const int batch_size,
    const int input_width,
    const int input_height,
    const float input_pixel_size,
    const float input_sdd,
    const int output_width,
    const int output_height,
    const float output_pixel_size,
    const float output_sdd)
{
    int x_idx = get_global_id(0);
    int y_idx = get_global_id(1);
    int proj_idx = get_global_id(2);

    if (x_idx >= output_width || y_idx >= output_height || proj_idx >= batch_size) {
        return;
    }

    float x = ((float)x_idx - ((float)(output_width - 1) * 0.5f)) * output_pixel_size;
    float y = (((float)(output_height - 1) * 0.5f) - (float)y_idx) * output_pixel_size;
    float z = -output_sdd;

    float nx = rotation_matrix[0] * x + rotation_matrix[1] * y + rotation_matrix[2] * z;
    float ny = rotation_matrix[3] * x + rotation_matrix[4] * y + rotation_matrix[5] * z;
    float nz = rotation_matrix[6] * x + rotation_matrix[7] * y + rotation_matrix[8] * z;

    float det_x = nx / nz * (-input_sdd);
    float det_y = ny / nz * (-input_sdd);
    float source_x = det_x / input_pixel_size + ((float)(input_width - 1) * 0.5f);
    float source_y = -det_y / input_pixel_size + ((float)(input_height - 1) * 0.5f);

    int input_proj_offset = proj_idx * input_width * input_height;
    int output_proj_offset = proj_idx * output_width * output_height;
    float value = bilinear_sample(input_projections + input_proj_offset, input_width, input_height, source_x, source_y);
    output_projections[output_proj_offset + y_idx * output_width + x_idx] = value;
}
"""


def list_opencl_devices() -> list[str]:
    if cl is None:
        return []
    devices: list[str] = []
    try:
        platforms = cl.get_platforms()
    except Exception:
        return []
    for platform_index, platform in enumerate(platforms):
        for device_index, device in enumerate(platform.get_devices()):
            devices.append(
                f"platform={platform_index} device={device_index} name={device.name.strip()} type={cl.device_type.to_string(device.type)}"
            )
    return devices


@dataclass(slots=True)
class SearchArtifacts:
    mse_values: np.ndarray
    x0: np.ndarray
    y0: np.ndarray


class CpuBackend:
    name = "cpu"

    def search(
        self,
        sinogram: np.ndarray,
        alpha: np.ndarray,
        beta: np.ndarray,
        theta0: np.ndarray,
        x0: np.ndarray,
        y0: np.ndarray,
        cb_params: ConeBeamParameters,
        config: SearchConfig,
    ) -> SearchArtifacts:
        sample_angles = np.arange(config.sample_count, dtype=np.float64) * (
            np.deg2rad(config.sample_angle_range_deg) / config.sample_count
        )
        detector_width = cb_params.detector_width
        detector_height = cb_params.detector_height
        pixel_size = cb_params.pixel_size
        sdd = cb_params.sdd

        mse_values = np.empty(alpha.shape[0], dtype=np.float32)
        for idx in range(alpha.shape[0]):
            sin_a = np.sin(alpha[idx])
            cos_a = np.cos(alpha[idx])
            sin_b = np.sin(beta[idx])
            cos_b = np.cos(beta[idx])

            theta = sample_angles + theta0[idx]
            reflect_theta = -sample_angles + theta0[idx]
            tan_t = np.tan(theta)
            tan_rt = np.tan(reflect_theta)

            temp = sdd / (tan_t * sin_a * sin_b + cos_b)
            rtemp = sdd / (tan_rt * sin_a * sin_b + cos_b)

            y = temp * (-tan_t * sin_a * cos_b + sin_b)
            ry = rtemp * (-tan_rt * sin_a * cos_b + sin_b)
            x = -temp * tan_t * cos_a
            rx = -rtemp * tan_rt * cos_a

            x = x / pixel_size + (detector_width - 1) / 2.0
            rx = rx / pixel_size + (detector_width - 1) / 2.0
            y = (detector_height - 1) / 2.0 - y / pixel_size
            ry = (detector_height - 1) / 2.0 - ry / pixel_size

            # Normalize by the count of *valid* (both rays on-detector, at
            # least one side with real signal) sample pairs, not the fixed
            # sample_count -- see the CORRECTION notes in the OpenCL kernels
            # below (same two fixes, same reasoning: this backend has its own
            # independent implementation of the same math).
            f_t = _bilinear_sample_numpy(sinogram, x, y)
            f_rt = _bilinear_sample_numpy(sinogram, rx, ry)
            in_bounds = _valid_mask_numpy(sinogram, x, y) & _valid_mask_numpy(sinogram, rx, ry)
            has_signal = (f_t >= _MIN_SIGNAL) | (f_rt >= _MIN_SIGNAL)
            valid = in_bounds & has_signal
            valid_count = int(np.count_nonzero(valid))
            if valid_count > 0:
                mse_values[idx] = np.sum(np.square(f_t[valid] - f_rt[valid]), dtype=np.float64) / valid_count
            else:
                mse_values[idx] = np.float32(np.inf)

        return SearchArtifacts(mse_values=mse_values, x0=x0, y0=y0)

    def resample(
        self,
        projections: np.ndarray,
        rotation_matrix: np.ndarray,
        input_params: ConeBeamParameters,
        output_params: ConeBeamParameters,
        batch_size: int = 0,
    ) -> np.ndarray:
        out = np.zeros(
            (projections.shape[0], output_params.detector_height, output_params.detector_width),
            dtype=np.float32,
        )
        x_coords = (
            np.arange(output_params.detector_width, dtype=np.float32)
            - (output_params.detector_width - 1) / 2.0
        ) * output_params.pixel_size
        y_coords = (
            (output_params.detector_height - 1) / 2.0
            - np.arange(output_params.detector_height, dtype=np.float32)
        ) * output_params.pixel_size
        grid_x, grid_y = np.meshgrid(x_coords, y_coords)
        z = np.full_like(grid_x, -output_params.sdd, dtype=np.float32)

        nx = rotation_matrix[0, 0] * grid_x + rotation_matrix[0, 1] * grid_y + rotation_matrix[0, 2] * z
        ny = rotation_matrix[1, 0] * grid_x + rotation_matrix[1, 1] * grid_y + rotation_matrix[1, 2] * z
        nz = rotation_matrix[2, 0] * grid_x + rotation_matrix[2, 1] * grid_y + rotation_matrix[2, 2] * z
        source_x = (nx / nz * (-input_params.sdd)) / input_params.pixel_size + (input_params.detector_width - 1) / 2.0
        source_y = -(ny / nz * (-input_params.sdd)) / input_params.pixel_size + (input_params.detector_height - 1) / 2.0

        flat_x = source_x.ravel()
        flat_y = source_y.ravel()
        for proj_idx in range(projections.shape[0]):
            sampled = _bilinear_sample_numpy(projections[proj_idx], flat_x, flat_y)
            out[proj_idx] = sampled.reshape(output_params.detector_height, output_params.detector_width)
        return out


class OpenCLBackend:
    name = "opencl"

    def __init__(self, platform_index: int | None = None, device_index: int | None = None) -> None:
        if cl is None:
            raise RuntimeError("pyopencl is not installed. Install the project dependencies to use the OpenCL backend.")

        platforms = cl.get_platforms()
        if not platforms:
            raise RuntimeError("No OpenCL platforms were found.")

        if platform_index is None:
            platform_index = 0
        platform = platforms[platform_index]
        devices = platform.get_devices()
        if not devices:
            raise RuntimeError(f"No OpenCL devices were found on platform {platform_index}.")

        if device_index is None:
            device_index = 0
        device = devices[device_index]
        self.device = device
        self.context = cl.Context(devices=[device])
        self.queue = cl.CommandQueue(self.context)
        self.program = cl.Program(self.context, KERNEL_SOURCE).build()

        # resample() is called once per streamed batch from pipeline.run_resample()
        # (typically ~12 times for a 180-projection dataset at the default batch
        # size). These caches let repeated calls reuse the kernel object, the
        # rotation-matrix buffer (identical across all batches of one resample
        # run), and the input/output device buffers, instead of paying full
        # allocate/build/free overhead on every batch -- see the
        # RepeatedKernelRetrieval warning this used to trigger and
        # docs/ARCHITECTURE.md's note on resample() being overhead-bound, not
        # compute-bound, at the per-batch granularity this ran at before.
        self._resample_kernel: cl.Kernel | None = None
        self._rotation_buffer: cl.Buffer | None = None
        self._rotation_matrix_id: int | None = None
        self._resample_input_buffer: cl.Buffer | None = None
        self._resample_output_buffer: cl.Buffer | None = None
        self._resample_buffer_capacity: int = 0

    def search(
        self,
        sinogram: np.ndarray,
        alpha: np.ndarray,
        beta: np.ndarray,
        theta0: np.ndarray,
        x0: np.ndarray,
        y0: np.ndarray,
        cb_params: ConeBeamParameters,
        config: SearchConfig,
    ) -> SearchArtifacts:
        mf = cl.mem_flags
        sinogram_buffer = cl.Buffer(
            self.context,
            mf.READ_ONLY | mf.COPY_HOST_PTR,
            hostbuf=np.asarray(sinogram, dtype=np.float32, order="C"),
        )
        alpha_buffer = cl.Buffer(self.context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=np.asarray(alpha, dtype=np.float32))
        beta_buffer = cl.Buffer(self.context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=np.asarray(beta, dtype=np.float32))
        theta0_buffer = cl.Buffer(self.context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=np.asarray(theta0, dtype=np.float32))
        mse_values = np.empty(alpha.shape[0], dtype=np.float32)
        mse_buffer = cl.Buffer(self.context, mf.WRITE_ONLY, mse_values.nbytes)

        local_size = min(256, int(self.device.max_work_group_size))
        local_size = max(1, local_size)
        if local_size & (local_size - 1):
            local_size = 1 << (local_size.bit_length() - 1)
        angle_step = np.float32(np.deg2rad(config.sample_angle_range_deg) / config.sample_count)

        self.program.center_search_reduce(
            self.queue,
            (local_size, alpha.shape[0]),
            (local_size, 1),
            sinogram_buffer,
            alpha_buffer,
            beta_buffer,
            theta0_buffer,
            mse_buffer,
            np.float32(cb_params.sdd),
            np.int32(cb_params.detector_width),
            np.int32(cb_params.detector_height),
            np.float32(cb_params.pixel_size),
            np.int32(config.sample_count),
            angle_step,
        )
        cl.enqueue_copy(self.queue, mse_values, mse_buffer).wait()
        return SearchArtifacts(mse_values=mse_values, x0=x0, y0=y0)

    def resample(
        self,
        projections: np.ndarray,
        rotation_matrix: np.ndarray,
        input_params: ConeBeamParameters,
        output_params: ConeBeamParameters,
        batch_size: int = 16,
    ) -> np.ndarray:
        mf = cl.mem_flags
        output = np.zeros(
            (projections.shape[0], output_params.detector_height, output_params.detector_width),
            dtype=np.float32,
        )

        # Reuse the rotation buffer across calls -- pipeline.run_resample()
        # passes the exact same rotation_matrix array object on every one of
        # its streamed-batch calls within a single resample run, so an
        # identity check is enough to know it's safe to skip re-uploading.
        if self._rotation_buffer is None or self._rotation_matrix_id != id(rotation_matrix):
            self._rotation_buffer = cl.Buffer(
                self.context,
                mf.READ_ONLY | mf.COPY_HOST_PTR,
                hostbuf=np.asarray(rotation_matrix, dtype=np.float32).reshape(-1),
            )
            self._rotation_matrix_id = id(rotation_matrix)
        rotation_buffer = self._rotation_buffer

        if self._resample_kernel is None:
            self._resample_kernel = cl.Kernel(self.program, "resample_projections")
        kernel = self._resample_kernel

        input_pixels = input_params.detector_width * input_params.detector_height
        output_pixels = output_params.detector_width * output_params.detector_height

        for start in range(0, projections.shape[0], batch_size):
            stop = min(start + batch_size, projections.shape[0])
            count = stop - start
            batch = np.asarray(projections[start:stop], dtype=np.float32, order="C")

            # Grow the reusable input/output buffers to fit the largest batch
            # seen so far, but never shrink/reallocate them for a smaller
            # (e.g. final, partial) batch -- enqueue_copy transfers exactly
            # `count` elements either way, sized off the host array, not off
            # buffer capacity.
            if self._resample_buffer_capacity < count:
                self._resample_input_buffer = cl.Buffer(self.context, mf.READ_ONLY, batch_size * input_pixels * 4)
                self._resample_output_buffer = cl.Buffer(self.context, mf.WRITE_ONLY, batch_size * output_pixels * 4)
                self._resample_buffer_capacity = batch_size
            input_buffer = self._resample_input_buffer
            output_buffer = self._resample_output_buffer

            cl.enqueue_copy(self.queue, input_buffer, batch)

            kernel.set_args(
                input_buffer,
                output_buffer,
                rotation_buffer,
                np.int32(count),
                np.int32(input_params.detector_width),
                np.int32(input_params.detector_height),
                np.float32(input_params.pixel_size),
                np.float32(input_params.sdd),
                np.int32(output_params.detector_width),
                np.int32(output_params.detector_height),
                np.float32(output_params.pixel_size),
                np.float32(output_params.sdd),
            )
            cl.enqueue_nd_range_kernel(
                self.queue,
                kernel,
                (output_params.detector_width, output_params.detector_height, count),
                None,
            )
            batch_output = np.empty(
                (count, output_params.detector_height, output_params.detector_width),
                dtype=np.float32,
            )
            cl.enqueue_copy(self.queue, batch_output, output_buffer).wait()
            output[start:stop] = batch_output

        return output


def _repo_root() -> Path:
    # .../projection-center/src/projection_center_searching/backends.py -> repo root
    return Path(__file__).resolve().parents[3]


def _run_cpp_quietly(cmd: list[str]) -> None:
    # forward_search/'s binaries print their own stage diagnostics
    # (device pick, sinogram build, kernel timings) to stderr -- captured
    # here rather than let through, so --backend cpp's output matches the
    # quiet, 3-line summary opencl/cpu already print (see cli.py's own
    # print statements, unmodified from the source repo). On failure, the
    # captured output is surfaced in the raised error instead of being
    # lost, so nothing is harder to debug than before.
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"cpp backend command failed (exit {result.returncode}): {' '.join(cmd)}\n"
            f"--- stdout ---\n{result.stdout}"
            f"--- stderr ---\n{result.stderr}"
        )


def _exe_name(name: str) -> str:
    # meson names build output <name>.exe on Windows, plain <name> everywhere
    # else -- Path(...).exists() checking the Linux name would always be
    # False on Windows even after a successful build.
    return f"{name}.exe" if sys.platform.startswith("win") else name


def _cpp_binary_path() -> Path:
    override = os.environ.get("FORWARD_SEARCH_CPP_BINARY")
    if override:
        return Path(override)
    return _repo_root() / "forward_search" / "builddir" / _exe_name("forward_search")


def _cpp_kernel_path() -> Path:
    override = os.environ.get("FORWARD_SEARCH_CPP_KERNEL")
    if override:
        return Path(override)
    return _repo_root() / "forward_search" / "kernels" / "forward_search.cl"


def _cpp_resample_binary_path() -> Path:
    override = os.environ.get("FORWARD_SEARCH_CPP_RESAMPLE_BINARY")
    if override:
        return Path(override)
    return _repo_root() / "forward_search" / "builddir" / _exe_name("forward_search_resample")


def _cpp_resample_kernel_path() -> Path:
    override = os.environ.get("FORWARD_SEARCH_CPP_RESAMPLE_KERNEL")
    if override:
        return Path(override)
    return _repo_root() / "forward_search" / "kernels" / "resample.cl"


class CppBackend:
    """Delegates both forward search and resampling to this repo's own
    C++/OpenCL implementation (forward_search/builddir/forward_search and
    forward_search_resample) by invoking them as subprocesses, instead of
    reimplementing their kernels in Python. This is the only backend that
    isn't a Python/PyOpenCL implementation of its own -- it's this repo's
    contribution connected into the shared CLI/pipeline as just another
    --backend choice.
    """

    name = "cpp"

    def __init__(self, mode: str = "buffer") -> None:
        self.mode = mode
        binary = _cpp_binary_path()
        if not binary.exists():
            raise RuntimeError(
                f"cpp backend binary not found at {binary}. Build it first: "
                "cd forward_search && meson setup builddir && meson compile -C builddir "
                "(or set FORWARD_SEARCH_CPP_BINARY to point elsewhere)."
            )

    def search_from_file(self, data_path: str | Path, config: SearchConfig) -> SearchResult:
        # forward_search.cpp hardcodes N_THETA=1000 and RANGE_DEG=30.0 (see
        # forward_search.cpp's computeCOR()) and doesn't expose them via its
        # CLI, unlike the opencl/cpu backends' sample_count/sample_angle_range_deg.
        if config.sample_count != 1000 or config.sample_angle_range_deg != 30.0:
            raise ValueError(
                "cpp backend hardcodes sample_count=1000 and sample_angle_range_deg=30.0 "
                "(forward_search.cpp's N_THETA/RANGE_DEG); it doesn't expose these via its "
                "CLI. Use --backend opencl or --backend cpu to vary them."
            )

        import h5py  # local import: only this backend needs it, others are pure numpy/pyopencl

        with tempfile.TemporaryDirectory() as tmp:
            output_h5 = Path(tmp) / "cpp_search_output.h5"
            cmd = [
                str(_cpp_binary_path()),
                "--data", str(data_path),
                "--mode", self.mode,
                "--kernel", str(_cpp_kernel_path()),
                "--xshift", str(config.xshift_range_mm),
                "--alpha", str(config.alpha_range_deg),
                "--beta", str(config.beta_range_deg),
                "--xshift-step", str(config.xshift_step_mm),
                "--alpha-step", str(config.alpha_step_deg),
                "--beta-step", str(config.beta_step_deg),
                "--output", str(output_h5),
            ]
            _run_cpp_quietly(cmd)

            with h5py.File(output_h5, "r") as handle:
                return SearchResult(
                    center_point=(float(handle["center_point"][0]), float(handle["center_point"][1])),
                    xshift=float(handle["xshift"][()]),
                    alpha=float(handle["alpha"][()]),
                    beta=float(handle["beta"][()]),
                    mse=float(handle["MSE"][()]),
                )

    def resample_from_file(
        self,
        data_path: str | Path,
        pose_path: str | Path,
        output_data_path: str | Path,
        resample_config: ResampleConfig,
    ) -> None:
        binary = _cpp_resample_binary_path()
        if not binary.exists():
            raise RuntimeError(
                f"cpp resample binary not found at {binary}. Build it first: "
                "cd forward_search && meson setup builddir && meson compile -C builddir "
                "(or set FORWARD_SEARCH_CPP_RESAMPLE_BINARY to point elsewhere)."
            )
        cmd = [
            str(binary),
            "--data", str(data_path),
            "--pose", str(pose_path),
            "--kernel", str(_cpp_resample_kernel_path()),
            "--downsample", str(resample_config.downsample_factor),
            "--batch-size", str(resample_config.batch_size),
            "--output", str(output_data_path),
        ]
        _run_cpp_quietly(cmd)


def get_backend(
    backend_name: str,
    platform_index: int | None = None,
    device_index: int | None = None,
    cpp_mode: str = "buffer",
):
    if backend_name == "cpu":
        return CpuBackend()
    if backend_name == "opencl":
        return OpenCLBackend(platform_index=platform_index, device_index=device_index)
    if backend_name == "cpp":
        return CppBackend(mode=cpp_mode)
    raise ValueError(f"Unsupported backend: {backend_name}")


def _valid_mask_numpy(image: np.ndarray, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    width = image.shape[1]
    height = image.shape[0]
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    return (x >= 0.0) & (x < width - 1) & (y >= 0.0) & (y < height - 1)


def _bilinear_sample_numpy(image: np.ndarray, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    width = image.shape[1]
    height = image.shape[0]
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    valid = _valid_mask_numpy(image, x, y)
    values = np.zeros(x.shape, dtype=np.float64)
    if not np.any(valid):
        return values.astype(np.float32)

    xv = x[valid]
    yv = y[valid]
    ix = np.floor(xv).astype(np.int64)
    iy = np.floor(yv).astype(np.int64)
    dx = xv - ix
    dy = yv - iy

    v00 = image[iy, ix]
    v10 = image[iy, ix + 1]
    v01 = image[iy + 1, ix]
    v11 = image[iy + 1, ix + 1]
    values[valid] = (1.0 - dx) * ((1.0 - dy) * v00 + dy * v01) + dx * ((1.0 - dy) * v10 + dy * v11)
    return values.astype(np.float32)
