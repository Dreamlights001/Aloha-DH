"""Dual-arm kinematics simulation package for sorting robot design."""

from .config import (
    ARM_OFFSET,
    ARM_MODELS,
    CAMERA_CONFIG,
    CONVEYOR_CONFIG,
    DEFAULT_ARM_MODEL,
    KINEMATICS_DEFAULT,
    OFFICIAL_POE_AXIS_POINTS,
    OFFICIAL_POE_M,
    OFFICIAL_POE_SLIST,
    OFFICIAL_SOURCES,
    GRIPPER_CONFIG,
    JOINT_LIMITS_EXAMPLE,
    MDH_PARAMS_EXAMPLE,
    MODEL_SPEC,
    SORTING_BINS,
)
from .io_utils import repo_root_from_file, resolve_animation_output_path
from .input_controls import GamepadInputManager, GamepadState
from .platform_utils import (
    RuntimeInfo,
    configure_matplotlib_runtime,
    default_mpl_config_dir,
    detect_gui_available,
    detect_wsl,
    get_runtime_info,
)
from .robot import DualArmSystem, IKResult, RobotArm6DOF, WaypointResult
from .scenario import SortingObject, SortingScenario

__all__ = [
    "ARM_OFFSET",
    "ARM_MODELS",
    "CAMERA_CONFIG",
    "CONVEYOR_CONFIG",
    "DEFAULT_ARM_MODEL",
    "KINEMATICS_DEFAULT",
    "OFFICIAL_POE_AXIS_POINTS",
    "OFFICIAL_POE_M",
    "OFFICIAL_POE_SLIST",
    "OFFICIAL_SOURCES",
    "GRIPPER_CONFIG",
    "JOINT_LIMITS_EXAMPLE",
    "MDH_PARAMS_EXAMPLE",
    "MODEL_SPEC",
    "SORTING_BINS",
    "repo_root_from_file",
    "resolve_animation_output_path",
    "GamepadState",
    "GamepadInputManager",
    "RuntimeInfo",
    "configure_matplotlib_runtime",
    "default_mpl_config_dir",
    "detect_gui_available",
    "detect_wsl",
    "get_runtime_info",
    "DualArmSystem",
    "IKResult",
    "WaypointResult",
    "RobotArm6DOF",
    "SortingObject",
    "SortingScenario",
]
