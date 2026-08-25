from __future__ import annotations

import math

import numpy as np

from .models import ConeBeamParameters, SearchConfig, SearchResult


def build_sinogram(projections: np.ndarray) -> np.ndarray:
    if projections.ndim == 2:
        sinogram = np.asarray(projections, dtype=np.float64)
    else:
        sinogram = projections.sum(axis=0, dtype=np.float64)
    min_value = float(sinogram.min())
    max_value = float(sinogram.max())
    if max_value == min_value:
        return np.zeros_like(sinogram, dtype=np.float32)
    normalized = (sinogram - min_value) / (max_value - min_value)
    return normalized.astype(np.float32, copy=False)


def build_parameter_grid(config: SearchConfig) -> np.ndarray:
    xshift = np.arange(
        -config.xshift_range_mm / 1000.0,
        config.xshift_range_mm / 1000.0 + config.xshift_step_mm / 2000.0,
        config.xshift_step_mm / 1000.0,
        dtype=np.float64,
    )
    alpha = np.deg2rad(
        np.arange(
            -config.alpha_range_deg,
            config.alpha_range_deg + config.alpha_step_deg / 2.0,
            config.alpha_step_deg,
            dtype=np.float64,
        )
    )
    beta = np.deg2rad(
        np.arange(
            -config.beta_range_deg,
            config.beta_range_deg + config.beta_step_deg / 2.0,
            config.beta_step_deg,
            dtype=np.float64,
        )
    )
    x_grid, a_grid, b_grid = np.meshgrid(xshift, alpha, beta, indexing="ij")
    return np.column_stack(
        [
            x_grid.ravel(),
            a_grid.ravel(),
            b_grid.ravel(),
        ]
    )


def compute_forward_geometry(
    cb_params: ConeBeamParameters,
    parameter_grid: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    xshift = parameter_grid[:, 0]
    alpha = parameter_grid[:, 1]
    beta = parameter_grid[:, 2]
    sod = cb_params.sod
    sdd = cb_params.sdd

    sin_a = np.sin(alpha)
    cos_a = np.cos(alpha)
    sin_b = np.sin(beta)
    cos_b = np.cos(beta)

    b0 = sdd * cos_a * sin_b
    b1 = sdd * cos_a * xshift * cos_b

    det_a = -sin_a * (-xshift * cos_a * sin_b + sod * sin_a) - cos_a * cos_b * sod * cos_a * cos_b
    safe = np.abs(det_a) > 1.0e-12
    inv_a00 = np.zeros_like(det_a)
    inv_a01 = np.zeros_like(det_a)
    inv_a10 = np.zeros_like(det_a)
    inv_a11 = np.zeros_like(det_a)
    inv_a00[safe] = (-xshift[safe] * cos_a[safe] * sin_b[safe] + sod * sin_a[safe]) / det_a[safe]
    inv_a01[safe] = (-cos_a[safe] * cos_b[safe]) / det_a[safe]
    inv_a10[safe] = (-sod * cos_a[safe] * cos_b[safe]) / det_a[safe]
    inv_a11[safe] = (-sin_a[safe]) / det_a[safe]

    x0 = inv_a00 * b0 + inv_a01 * b1
    y0 = inv_a10 * b0 + inv_a11 * b1

    r00 = cos_a
    r10 = sin_a * cos_b
    r20 = sin_a * sin_b
    r01 = -sin_a
    r11 = cos_a * cos_b
    r21 = cos_a * sin_b
    r02 = np.zeros_like(alpha)
    r12 = -sin_b
    r22 = cos_b

    x_p = r00 * x0 + r10 * y0 - r20 * sdd
    z_p = r02 * x0 + r12 * y0 - r22 * sdd
    theta0 = np.arctan2(x_p, z_p)
    return x0, y0, theta0


def compute_resample_geometry(
    cb_params: ConeBeamParameters,
    pose: SearchResult,
    downsample_factor: int,
) -> tuple[ConeBeamParameters, np.ndarray]:
    if downsample_factor < 1:
        raise ValueError("downsample_factor must be >= 1")
    if cb_params.detector_width % downsample_factor != 0 or cb_params.detector_height % downsample_factor != 0:
        raise ValueError("downsample_factor must evenly divide detector_width and detector_height")

    real_pixel_size = cb_params.pixel_size * downsample_factor
    real_width = cb_params.detector_width // downsample_factor
    real_height = cb_params.detector_height // downsample_factor

    x0, y0 = pose.center_point
    xshift = pose.xshift
    alpha = pose.alpha
    beta = pose.beta

    sin_a = math.sin(alpha)
    cos_a = math.cos(alpha)
    sin_b = math.sin(beta)
    cos_b = math.cos(beta)

    real_sdd = float(np.linalg.norm([x0, y0, cb_params.sdd]))
    axis = np.array([-sin_a, cos_a * cos_b, cos_a * sin_b], dtype=np.float64)
    offset = np.array([xshift, 0.0, -cb_params.sod], dtype=np.float64)
    real_sod = float(np.linalg.norm(np.cross(offset, axis)) / np.linalg.norm(axis))

    y_axis = axis / np.linalg.norm(axis)
    z_axis = -np.array([x0, y0, -cb_params.sdd], dtype=np.float64)
    z_axis = z_axis / np.linalg.norm(z_axis)
    x_axis = np.cross(y_axis, z_axis)
    x_axis = x_axis / np.linalg.norm(x_axis)
    z_axis = np.cross(x_axis, y_axis)
    z_axis = z_axis / np.linalg.norm(z_axis)
    rotation_matrix = np.column_stack([x_axis, y_axis, z_axis]).astype(np.float32)

    real_params = ConeBeamParameters(
        num_projs=cb_params.num_projs,
        sdd=real_sdd,
        sod=real_sod,
        pixel_size=real_pixel_size,
        voxel_size=cb_params.voxel_size,
        volume_num_xz=cb_params.volume_num_xz,
        volume_num_y=cb_params.volume_num_y,
        detector_width=real_width,
        detector_height=real_height,
        angles=cb_params.angles,
    )
    return real_params, rotation_matrix
