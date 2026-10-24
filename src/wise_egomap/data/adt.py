"""Typed reader for extracted ADT (Aria Digital Twin) sequence files.

Layout on disk (one directory per sequence)::

    seq/
      main_groundtruth/2d_bounding_box.csv
      main_groundtruth/3d_bounding_box.csv
      main_groundtruth/aria_trajectory.csv
      main_groundtruth/instances.json
      main_groundtruth/metadata.json
      main_groundtruth/scene_objects.csv
      mps_slam_calibration/online_calibration.jsonl

Time fields use seconds unless the field name says otherwise. Lengths use
metres. Poses map device coordinates to world coordinates.
"""

from __future__ import annotations

import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path


def _f(value: str) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"non-finite number in ADT file: {value!r}")
    return number


@dataclass(frozen=True)
class Instance:
    """One annotated instance in a sequence."""

    instance_id: int
    instance_name: str
    prototype_name: str
    category: str
    motion_type: str
    rigidity: str
    up_vector: tuple[float, float, float]
    front_vector: tuple[float, float, float]


@dataclass(frozen=True)
class ObjectExtent:
    """Local bounding box of one object, in metres."""

    object_uid: int
    x_min_m: float
    x_max_m: float
    y_min_m: float
    y_max_m: float
    z_min_m: float
    z_max_m: float

    @property
    def size_m(self) -> tuple[float, float, float]:
        return (
            self.x_max_m - self.x_min_m,
            self.y_max_m - self.y_min_m,
            self.z_max_m - self.z_min_m,
        )


@dataclass(frozen=True)
class ObjectPose:
    """Object pose in the world frame.

    ``timestamp_s`` is -1.0 for the canonical pose of a static object.
    """

    object_uid: int
    timestamp_s: float
    t_world_m: tuple[float, float, float]
    q_world_wxyz: tuple[float, float, float, float]


@dataclass(frozen=True)
class Box2D:
    """One 2D annotation box of one object in one camera stream."""

    stream_id: str
    object_uid: int
    timestamp_s: float
    x_min_px: float
    x_max_px: float
    y_min_px: float
    y_max_px: float
    visibility_ratio: float

    @property
    def area_px(self) -> int:
        width = max(0.0, self.x_max_px - self.x_min_px)
        height = max(0.0, self.y_max_px - self.y_min_px)
        return int(round(width * height))


@dataclass(frozen=True)
class CameraPose:
    """Device pose in the world frame from the GT trajectory."""

    tracking_timestamp_s: float
    t_device_m: tuple[float, float, float]
    q_device_wxyz: tuple[float, float, float, float]
    quality_score: float


@dataclass(frozen=True)
class CameraCalibration:
    """One camera's calibration record.

    ``intrinsics_params`` holds the 15 raw values of the
    FisheyeRadTanThinPrism model. The first four are the pinhole
    approximation: fx, fy, cx, cy.
    """

    label: str
    serial_number: str
    intrinsics_name: str
    intrinsics_params: tuple[float, ...]
    t_device_m: tuple[float, float, float]
    q_device_wxyz: tuple[float, float, float, float]

    @property
    def pinhole(self) -> tuple[float, float, float, float]:
        if len(self.intrinsics_params) < 4:
            raise ValueError(f"{self.label} has fewer than 4 intrinsic params")
        return self.intrinsics_params[:4]


@dataclass(frozen=True)
class AdtSequence:
    """One fully loaded ADT sequence."""

    sequence_id: str
    dataset_name: str
    scene: str
    instances: dict[int, Instance]
    extents: dict[int, ObjectExtent]
    object_poses: list[ObjectPose]
    boxes_2d: list[Box2D]
    camera_poses: list[CameraPose]
    calibrations: dict[str, CameraCalibration]
    start_s: float
    end_s: float

    @property
    def duration_s(self) -> float:
        return self.end_s - self.start_s

    def dynamic_instance_ids(self) -> tuple[int, ...]:
        return tuple(
            instance_id
            for instance_id, item in sorted(self.instances.items())
            if item.motion_type == "dynamic"
        )


def load_adt_sequence(path: Path) -> AdtSequence:
    """Load one extracted ADT sequence directory."""
    root = Path(path)
    gt = root / "main_groundtruth"
    calib = root / "mps_slam_calibration" / "online_calibration.jsonl"
    for required in (
        gt / "2d_bounding_box.csv",
        gt / "3d_bounding_box.csv",
        gt / "aria_trajectory.csv",
        gt / "instances.json",
        gt / "metadata.json",
        gt / "scene_objects.csv",
        calib,
    ):
        if not required.is_file():
            raise FileNotFoundError(f"missing ADT file: {required}")

    metadata = json.loads((gt / "metadata.json").read_text())
    instances = _load_instances(gt / "instances.json")
    extents = _load_extents(gt / "3d_bounding_box.csv")
    object_poses = _load_object_poses(gt / "scene_objects.csv")
    boxes_2d = _load_boxes_2d(gt / "2d_bounding_box.csv")
    camera_poses = _load_camera_poses(gt / "aria_trajectory.csv")
    calibrations = _load_calibrations(calib)

    timestamps = [pose.timestamp_s for pose in object_poses if pose.timestamp_s >= 0.0]
    if not timestamps:
        raise ValueError(f"no timed object poses in {root}")
    return AdtSequence(
        sequence_id=root.name,
        dataset_name=metadata.get("dataset_name", ""),
        scene=metadata.get("scene", ""),
        instances=instances,
        extents=extents,
        object_poses=object_poses,
        boxes_2d=boxes_2d,
        camera_poses=camera_poses,
        calibrations=calibrations,
        start_s=min(timestamps),
        end_s=max(timestamps),
    )


def _load_instances(path: Path) -> dict[int, Instance]:
    raw = json.loads(path.read_text())
    out: dict[int, Instance] = {}
    for _, value in raw.items():
        pose = value["canonical_pose"]
        instance = Instance(
            instance_id=int(value["instance_id"]),
            instance_name=str(value["instance_name"]),
            prototype_name=str(value["prototype_name"]),
            category=str(value["category"]),
            motion_type=str(value["motion_type"]),
            rigidity=str(value["rigidity"]),
            up_vector=tuple(float(v) for v in pose["up_vector"]),
            front_vector=tuple(float(v) for v in pose["front_vector"]),
        )
        out[instance.instance_id] = instance
    return out


def _load_extents(path: Path) -> dict[int, ObjectExtent]:
    out: dict[int, ObjectExtent] = {}
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            extent = ObjectExtent(
                object_uid=int(row["object_uid"]),
                x_min_m=_f(row["p_local_obj_xmin[m]"]),
                x_max_m=_f(row["p_local_obj_xmax[m]"]),
                y_min_m=_f(row["p_local_obj_ymin[m]"]),
                y_max_m=_f(row["p_local_obj_ymax[m]"]),
                z_min_m=_f(row["p_local_obj_zmin[m]"]),
                z_max_m=_f(row["p_local_obj_zmax[m]"]),
            )
            out[extent.object_uid] = extent
    return out


def _load_object_poses(path: Path) -> list[ObjectPose]:
    out: list[ObjectPose] = []
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            out.append(
                ObjectPose(
                    object_uid=int(row["object_uid"]),
                    timestamp_s=_f(row["timestamp[ns]"]) / 1e9,
                    t_world_m=(
                        _f(row["t_wo_x[m]"]),
                        _f(row["t_wo_y[m]"]),
                        _f(row["t_wo_z[m]"]),
                    ),
                    q_world_wxyz=(
                        _f(row["q_wo_w"]),
                        _f(row["q_wo_x"]),
                        _f(row["q_wo_y"]),
                        _f(row["q_wo_z"]),
                    ),
                )
            )
    return out


def _load_boxes_2d(path: Path) -> list[Box2D]:
    out: list[Box2D] = []
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            out.append(
                Box2D(
                    stream_id=str(row["stream_id"]),
                    object_uid=int(row["object_uid"]),
                    timestamp_s=_f(row["timestamp[ns]"]) / 1e9,
                    x_min_px=_f(row["x_min[pixel]"]),
                    x_max_px=_f(row["x_max[pixel]"]),
                    y_min_px=_f(row["y_min[pixel]"]),
                    y_max_px=_f(row["y_max[pixel]"]),
                    visibility_ratio=_f(row["visibility_ratio[%]"]) / 100.0,
                )
            )
    return out


def _load_camera_poses(path: Path) -> list[CameraPose]:
    out: list[CameraPose] = []
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            out.append(
                CameraPose(
                    tracking_timestamp_s=_f(row["tracking_timestamp_us"]) / 1e6,
                    t_device_m=(
                        _f(row["tx_world_device"]),
                        _f(row["ty_world_device"]),
                        _f(row["tz_world_device"]),
                    ),
                    q_device_wxyz=(
                        _f(row["qw_world_device"]),
                        _f(row["qx_world_device"]),
                        _f(row["qy_world_device"]),
                        _f(row["qz_world_device"]),
                    ),
                    quality_score=_f(row["quality_score"]),
                )
            )
    out.sort(key=lambda pose: pose.tracking_timestamp_s)
    return out


def _load_calibrations(path: Path) -> dict[str, CameraCalibration]:
    with path.open() as handle:
        record = json.loads(handle.read().strip().splitlines()[-1])
    out: dict[str, CameraCalibration] = {}
    for camera in record["CameraCalibrations"]:
        label = str(camera["Label"])
        projection = camera["Projection"]
        t_device = camera["T_Device_Camera"]["Translation"]
        wxyz = camera["T_Device_Camera"]["UnitQuaternion"]
        out[label] = CameraCalibration(
            label=label,
            serial_number=str(camera.get("SerialNumber", "")),
            intrinsics_name=str(projection.get("Name", "")),
            intrinsics_params=tuple(float(v) for v in projection["Params"]),
            t_device_m=tuple(float(v) for v in t_device),
            q_device_wxyz=(float(wxyz[0]),) + tuple(float(v) for v in wxyz[1]),
        )
    return out
