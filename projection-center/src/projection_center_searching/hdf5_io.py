from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path

import h5py
import numpy as np

from .models import ConeBeamParameters, SearchResult


def load_metadata(path: str | Path) -> ConeBeamParameters:
    dataset_path = Path(path)
    with h5py.File(dataset_path, "r") as handle:
        projection_shape = tuple(int(v) for v in handle["Projection"].shape)
        if len(projection_shape) != 3:
            raise ValueError(f"Expected Projection to have 3 dimensions, got shape {projection_shape}.")

        num_projs, projection_height, projection_width = projection_shape
        params = ConeBeamParameters(
            num_projs=num_projs,
            sdd=float(handle["SDD"][()]),
            sod=float(handle["SOD"][()]),
            pixel_size=float(handle["pixelSize"][()]),
            voxel_size=float(handle["voxelSize"][()]),
            volume_num_xz=int(handle["Volumen_num_xz"][()]),
            volume_num_y=int(handle["Volumen_num_y"][()]),
            detector_width=projection_width,
            detector_height=projection_height,
            angles=np.asarray(handle["Angle"][()], dtype=np.float64),
        )
    return params


def load_dataset(path: str | Path) -> tuple[ConeBeamParameters, np.ndarray]:
    params = load_metadata(path)
    dataset_path = Path(path)
    with h5py.File(dataset_path, "r") as handle:
        projections = np.asarray(handle["Projection"][()], dtype=np.float32)
    return params, projections


def build_sinogram_from_hdf5(path: str | Path, batch_size: int = 16) -> tuple[ConeBeamParameters, np.ndarray]:
    dataset_path = Path(path)
    params = load_metadata(dataset_path)
    sinogram = np.zeros((params.detector_height, params.detector_width), dtype=np.float64)
    with h5py.File(dataset_path, "r") as handle:
        projections = handle["Projection"]
        for start in range(0, params.num_projs, batch_size):
            stop = min(start + batch_size, params.num_projs)
            batch = np.asarray(projections[start:stop], dtype=np.float32)
            sinogram += batch.sum(axis=0, dtype=np.float64)
    return params, sinogram


def stream_projection_batches(
    path: str | Path,
    batch_size: int,
) -> Iterator[tuple[int, np.ndarray]]:
    dataset_path = Path(path)
    params = load_metadata(dataset_path)
    with h5py.File(dataset_path, "r") as handle:
        projections = handle["Projection"]
        for start in range(0, params.num_projs, batch_size):
            stop = min(start + batch_size, params.num_projs)
            yield start, np.asarray(projections[start:stop], dtype=np.float32)


def write_pose_json(path: str | Path, result: SearchResult) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "center_point": [float(result.center_point[0]), float(result.center_point[1])],
        "xshift": float(result.xshift),
        "alpha": float(result.alpha),
        "beta": float(result.beta),
        "MSE": float(result.mse),
    }
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def read_pose_json(path: str | Path) -> SearchResult:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return SearchResult(
        center_point=(float(payload["center_point"][0]), float(payload["center_point"][1])),
        xshift=float(payload["xshift"]),
        alpha=float(payload["alpha"]),
        beta=float(payload["beta"]),
        mse=float(payload["MSE"]),
    )


def write_resampled_dataset(
    path: str | Path,
    params: ConeBeamParameters,
    projections: np.ndarray,
) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(output_path, "w") as handle:
        handle.create_dataset("pixelSize", dtype=np.float64, data=params.pixel_size)
        handle.create_dataset("SDD", dtype=np.float64, data=params.sdd)
        handle.create_dataset("SOD", dtype=np.float64, data=params.sod)
        handle.create_dataset("voxelSize", dtype=np.float64, data=params.voxel_size)
        handle.create_dataset("Volumen_num_xz", dtype=np.float64, data=params.volume_num_xz)
        handle.create_dataset("Volumen_num_y", dtype=np.float64, data=params.volume_num_y)
        handle.create_dataset("num_projs", dtype=np.float64, data=params.num_projs)
        handle.create_dataset("detector_width", dtype=np.float64, data=params.detector_width)
        handle.create_dataset("detector_height", dtype=np.float64, data=params.detector_height)
        handle.create_dataset("Angle", dtype=np.float64, data=params.angles)
        handle.create_dataset(
            "Projection",
            dtype=np.float32,
            data=np.asarray(projections, dtype=np.float32),
        )


def initialize_resampled_dataset(path: str | Path, params: ConeBeamParameters) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(output_path, "w") as handle:
        handle.create_dataset("pixelSize", dtype=np.float64, data=params.pixel_size)
        handle.create_dataset("SDD", dtype=np.float64, data=params.sdd)
        handle.create_dataset("SOD", dtype=np.float64, data=params.sod)
        handle.create_dataset("voxelSize", dtype=np.float64, data=params.voxel_size)
        handle.create_dataset("Volumen_num_xz", dtype=np.float64, data=params.volume_num_xz)
        handle.create_dataset("Volumen_num_y", dtype=np.float64, data=params.volume_num_y)
        handle.create_dataset("num_projs", dtype=np.float64, data=params.num_projs)
        handle.create_dataset("detector_width", dtype=np.float64, data=params.detector_width)
        handle.create_dataset("detector_height", dtype=np.float64, data=params.detector_height)
        handle.create_dataset("Angle", dtype=np.float64, data=params.angles)
        handle.create_dataset(
            "Projection",
            shape=(params.num_projs, params.detector_height, params.detector_width),
            dtype=np.float32,
        )


def write_projection_batch(path: str | Path, start: int, projections: np.ndarray) -> None:
    output_path = Path(path)
    stop = start + projections.shape[0]
    with h5py.File(output_path, "r+") as handle:
        handle["Projection"][start:stop] = np.asarray(projections, dtype=np.float32)
