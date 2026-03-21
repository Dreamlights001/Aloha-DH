"""Official configuration for the dual-arm sorting simulation.

All geometric units are meters and angles are radians in runtime values.
Official dimensions and joint ranges are stored in mm/deg and converted.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Tuple

MDHRow = Tuple[float, float, float, float]
Twist = Tuple[float, float, float, float, float, float]
Point3 = Tuple[float, float, float]

OFFICIAL_SOURCES = {
    "aloha_vx300s_6dof": "https://docs.trossenrobotics.com/interbotix_xsarms_docs/specifications/avx300s.html",
    "vx300s_6dof": "https://docs.trossenrobotics.com/interbotix_xsarms_docs/specifications/vx300s.html",
    "xseries_linkage": "https://docs.trossenrobotics.com/interbotix_xsarms_docs/specifications.html",
}

ARM_MODELS: Dict[str, Dict[str, Any]] = {
    "aloha_vx300s_6dof": {
        "display_name": "ALOHA ViperX-300 6DOF",
        # Order follows the official 6DOF PoE Slist specification.
        "joint_order": (
            "waist",
            "shoulder",
            "elbow",
            "forearm_roll",
            "wrist_angle",
            "wrist_rotate",
        ),
        # Source: avx300s official specifications page (re-ordered to match joint_order above).
        "joint_limits_deg": [
            (-180.0, 180.0),
            (-101.0, 101.0),
            (-101.0, 92.0),
            (-180.0, 180.0),
            (-107.0, 130.0),
            (-180.0, 180.0),
        ],
        # Source: avx300s arm sections table.
        "arm_sections_mm": {
            "upper_arm": 306.0,
            "forearm": 300.0,
            "wrist_tilt_to_rotate": 70.0,
            "gripper_to_rail": 69.0,
            "finger_tip": 68.0,
        },
        # Source: X-Series linkage dimensions table.
        "linkage_mm": {
            "A": 300.0,
            "B": 60.0,
            "C_true_upper_arm": 306.0,
            "D": 300.0,
            "E_deg": 11.3,
        },
        # Source: avx300s gripper opening table.
        "gripper_opening_mm": {
            "min": 42.0,
            "max": 116.0,
        },
        # The base-to-shoulder z offset is from official kinematics constants used on VX300S docs.
        "base_shoulder_offset_m": 0.12705,
        # Source: vx300s kinematic properties (PoE M matrix and Slist).
        "poe_home_m": (
            (1.0, 0.0, 0.0, 0.536494),
            (0.0, 1.0, 0.0, 0.0),
            (0.0, 0.0, 1.0, 0.42705),
            (0.0, 0.0, 0.0, 1.0),
        ),
        "poe_slist": (
            (0.0, 0.0, 1.0, 0.0, 0.0, 0.0),
            (0.0, 1.0, 0.0, -0.12705, 0.0, 0.0),
            (0.0, 1.0, 0.0, -0.42705, 0.0, 0.05955),
            (1.0, 0.0, 0.0, 0.0, 0.42705, 0.0),
            (0.0, 1.0, 0.0, -0.42705, 0.0, 0.35955),
            (1.0, 0.0, 0.0, 0.0, 0.42705, 0.0),
        ),
        # Preferred points on each joint axis for visualization continuity.
        # These points do not affect FK math; they only avoid visual fold-back on collinear roll axes.
        "poe_axis_points": (
            (0.0, 0.0, 0.0),         # waist
            (0.0, 0.0, 0.12705),     # shoulder
            (0.05955, 0.0, 0.42705), # elbow
            (0.35955, 0.0, 0.42705), # forearm roll
            (0.35955, 0.0, 0.42705), # wrist angle
            (0.536494, 0.0, 0.42705),# wrist rotate
        ),
    }
}

DEFAULT_ARM_MODEL = "aloha_vx300s_6dof"
KINEMATICS_DEFAULT = "poe"


def joint_limits_deg_to_rad(joint_limits_deg: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
    return [(math.radians(low), math.radians(high)) for low, high in joint_limits_deg]


def build_mdh_params_from_official_specs(
    model_spec: Dict[str, Any],
    include_finger_tip: bool = True,
) -> List[MDHRow]:
    """Build a practical MDH chain from official section dimensions.

    Mapping assumptions:
    - J2/J3 use true upper-arm and forearm lengths from official dimensions.
    - J5/J6 use wrist-to-tool dimensions for the terminal segment.
    - This is a minimal design-stage kinematics model, not a factory-calibrated controller model.
    """
    sections = model_spec["arm_sections_mm"]
    upper_arm_m = sections["upper_arm"] / 1000.0
    forearm_m = sections["forearm"] / 1000.0
    wrist_to_rotate_m = sections["wrist_tilt_to_rotate"] / 1000.0
    tool_m = sections["gripper_to_rail"] / 1000.0
    if include_finger_tip:
        tool_m += sections["finger_tip"] / 1000.0

    base_shoulder_offset_m = float(model_spec.get("base_shoulder_offset_m", 0.12705))

    # [theta_offset, d, a, alpha]
    return [
        (0.0, base_shoulder_offset_m, 0.0, math.pi / 2),
        (0.0, 0.0, upper_arm_m, 0.0),
        (0.0, 0.0, forearm_m, 0.0),
        (0.0, 0.0, 0.0, math.pi / 2),
        (0.0, 0.0, wrist_to_rotate_m, -math.pi / 2),
        (0.0, 0.0, tool_m, 0.0),
    ]


def get_default_model_spec() -> Dict[str, Any]:
    return ARM_MODELS[DEFAULT_ARM_MODEL]


MODEL_SPEC = get_default_model_spec()

# Runtime defaults used by the rest of the simulation package.
MDH_PARAMS_EXAMPLE: List[MDHRow] = build_mdh_params_from_official_specs(MODEL_SPEC)
JOINT_LIMITS_EXAMPLE: List[Tuple[float, float]] = joint_limits_deg_to_rad(MODEL_SPEC["joint_limits_deg"])
OFFICIAL_POE_M: List[List[float]] = [list(row) for row in MODEL_SPEC["poe_home_m"]]
OFFICIAL_POE_SLIST: List[Twist] = [tuple(row) for row in MODEL_SPEC["poe_slist"]]
OFFICIAL_POE_AXIS_POINTS: List[Point3] = [tuple(row) for row in MODEL_SPEC["poe_axis_points"]]

# Adjustable gripper setup for design iteration.
GRIPPER_CONFIG: Dict[str, float | str] = {
    "max_width": MODEL_SPEC["gripper_opening_mm"]["max"] / 1000.0,
    "max_force": 50.0,
    "type": "parallel",
}

# Right arm base offset relative to the left arm base.
ARM_OFFSET = (0.60, 0.0, 0.0)

CONVEYOR_CONFIG: Dict[str, float] = {
    "length": 1.0,
    "width": 0.3,
    "height": 0.1,
    "speed": 0.1,
}

CAMERA_CONFIG = {
    "top_view": {
        "position": (0.5, 0.0, 1.0),
        "orientation": (0.0, -math.pi / 2, 0.0),
    },
    "front_view": {
        "position": (0.8, 0.0, 0.5),
        "orientation": (0.0, 0.0, -math.pi / 4),
    },
}

# Typical sorting bins: adjust to your line layout.
SORTING_BINS = {
    "normal": (0.30, -0.30, 0.12),
    "defect": (0.90, 0.30, 0.12),
}
