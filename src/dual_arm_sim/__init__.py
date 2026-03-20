"""Dual-arm MDH simulation package for sorting robot design."""

from .config import (
    ARM_OFFSET,
    ARM_MODELS,
    CAMERA_CONFIG,
    CONVEYOR_CONFIG,
    DEFAULT_ARM_MODEL,
    OFFICIAL_SOURCES,
    GRIPPER_CONFIG,
    JOINT_LIMITS_EXAMPLE,
    MDH_PARAMS_EXAMPLE,
    MODEL_SPEC,
    SORTING_BINS,
)
from .io_utils import repo_root_from_file, resolve_animation_output_path
from .robot import DualArmSystem, IKResult, RobotArm6DOF, WaypointResult
from .scenario import SortingObject, SortingScenario

__all__ = [
    "ARM_OFFSET",
    "ARM_MODELS",
    "CAMERA_CONFIG",
    "CONVEYOR_CONFIG",
    "DEFAULT_ARM_MODEL",
    "OFFICIAL_SOURCES",
    "GRIPPER_CONFIG",
    "JOINT_LIMITS_EXAMPLE",
    "MDH_PARAMS_EXAMPLE",
    "MODEL_SPEC",
    "SORTING_BINS",
    "repo_root_from_file",
    "resolve_animation_output_path",
    "DualArmSystem",
    "IKResult",
    "WaypointResult",
    "RobotArm6DOF",
    "SortingObject",
    "SortingScenario",
]
