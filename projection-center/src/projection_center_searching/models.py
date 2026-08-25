from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(slots=True)
class ConeBeamParameters:
    num_projs: int
    sdd: float
    sod: float
    pixel_size: float
    voxel_size: float
    volume_num_xz: int
    volume_num_y: int
    detector_width: int
    detector_height: int
    angles: np.ndarray


@dataclass(slots=True)
class SearchConfig:
    xshift_range_mm: float = 40.0
    alpha_range_deg: float = 10.0
    beta_range_deg: float = 10.0
    xshift_step_mm: float = 1.0
    alpha_step_deg: float = 1.0
    beta_step_deg: float = 1.0
    sample_count: int = 1000
    sample_angle_range_deg: float = 30.0


@dataclass(slots=True)
class SearchResult:
    center_point: tuple[float, float]
    xshift: float
    alpha: float
    beta: float
    mse: float


@dataclass(slots=True)
class ResampleConfig:
    downsample_factor: int = 1
    batch_size: int = 16
