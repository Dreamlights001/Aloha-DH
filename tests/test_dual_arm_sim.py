from __future__ import annotations

import math
import os
import sys
from pathlib import Path
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from dual_arm_sim import (  # noqa: E402
    ARM_OFFSET,
    DEFAULT_ARM_MODEL,
    JOINT_LIMITS_EXAMPLE,
    MDH_PARAMS_EXAMPLE,
    MODEL_SPEC,
    OFFICIAL_SOURCES,
    DualArmSystem,
    RobotArm6DOF,
    configure_matplotlib_runtime,
    detect_gui_available,
    detect_wsl,
    resolve_animation_output_path,
)


class TestRobotArm6DOF(unittest.TestCase):
    def setUp(self) -> None:
        self.arm = RobotArm6DOF(
            name="test",
            mdh_params=MDH_PARAMS_EXAMPLE,
            joint_limits=JOINT_LIMITS_EXAMPLE,
        )

    def test_forward_kinematics_output_shape(self) -> None:
        t, positions, transforms = self.arm.forward_kinematics([0.0] * 6)
        self.assertEqual(len(t), 4)
        self.assertEqual(len(t[0]), 4)
        self.assertEqual(len(positions), 7)
        self.assertEqual(len(transforms), 7)

    def test_end_effector_reach_within_expected_range(self) -> None:
        p = self.arm.end_effector_position([0.0] * 6)
        reach_xy = math.sqrt(p[0] * p[0] + p[1] * p[1])
        self.assertGreater(reach_xy, 0.70)
        self.assertLess(reach_xy, 0.90)

    def test_inverse_kinematics_position(self) -> None:
        target = [0.55, 0.05, 0.20]
        ik = self.arm.inverse_kinematics_position(target)
        self.assertTrue(ik.success)
        p = self.arm.end_effector_position(ik.joint_angles)
        err = math.sqrt(sum((target[i] - p[i]) ** 2 for i in range(3)))
        self.assertLess(err, 2e-3)


class TestDualArmSystem(unittest.TestCase):
    def setUp(self) -> None:
        self.dual = DualArmSystem(
            mdh_params=MDH_PARAMS_EXAMPLE,
            joint_limits=JOINT_LIMITS_EXAMPLE,
            arm_offset=ARM_OFFSET,
        )

    def test_base_offset(self) -> None:
        state = self.dual.forward_both([0.0] * 6, [0.0] * 6)
        left_base_x = state["left_joint_positions"][0][0]
        right_base_x = state["right_joint_positions"][0][0]
        self.assertAlmostEqual(right_base_x - left_base_x, ARM_OFFSET[0], places=6)

    def test_synchronized_trajectory_lengths(self) -> None:
        left, right = self.dual.synchronized_sort_trajectories(
            defect_pick=[0.78, 0.05, 0.13],
            defect_place=[0.95, 0.35, 0.12],
            normal_pick=[0.35, -0.03, 0.13],
            normal_place=[0.25, -0.35, 0.12],
        )
        self.assertGreater(len(left), 0)
        self.assertEqual(len(left), len(right))

    def test_reported_waypoint_errors_for_reachable_points(self) -> None:
        left, right, report = self.dual.synchronized_sort_trajectories(
            defect_pick=[0.15, 0.00, 0.13],
            defect_place=[0.95, 0.35, 0.12],
            normal_pick=[0.36, -0.08, 0.13],
            normal_place=[0.25, -0.35, 0.12],
            return_report=True,
        )
        self.assertEqual(len(left), len(right))
        self.assertEqual(len(report["left"]), 6)
        self.assertEqual(len(report["right"]), 6)
        self.assertLess(max(wp.error_norm for wp in report["left"]), 0.03)
        self.assertLess(max(wp.error_norm for wp in report["right"]), 0.03)

    def test_strict_ik_raises_for_unreachable_waypoint(self) -> None:
        with self.assertRaises(ValueError):
            self.dual.plan_pick_and_place(
                self.dual.left_arm,
                pick_pos=[2.0, 2.0, 2.0],
                place_pos=[2.1, 2.1, 2.1],
                strict_ik=True,
            )


class TestOfficialModelConfig(unittest.TestCase):
    def test_official_sources_and_default_model(self) -> None:
        self.assertEqual(DEFAULT_ARM_MODEL, "aloha_vx300s_6dof")
        self.assertIn("aloha_vx300s_6dof", OFFICIAL_SOURCES)
        self.assertIn("display_name", MODEL_SPEC)
        self.assertEqual(MODEL_SPEC["display_name"], "ALOHA ViperX-300 6DOF")
        self.assertEqual(len(MODEL_SPEC["joint_limits_deg"]), 6)


class TestOutputPathHelpers(unittest.TestCase):
    def test_resolve_relative_save_path_to_repo_output(self) -> None:
        repo_root = Path(__file__).resolve().parents[1]
        path = resolve_animation_output_path("demo.gif", repo_root=repo_root)
        self.assertTrue(str(path).endswith("output/demo.gif"))
        self.assertTrue(path.parent.exists())

    def test_resolve_none_uses_default_name(self) -> None:
        repo_root = Path(__file__).resolve().parents[1]
        path = resolve_animation_output_path(None, repo_root=repo_root, default_filename="sorting_demo.gif")
        self.assertTrue(str(path).endswith("output/sorting_demo.gif"))


class TestPlatformUtils(unittest.TestCase):
    def test_detect_wsl_from_env(self) -> None:
        self.assertTrue(detect_wsl(env={"WSL_DISTRO_NAME": "Ubuntu"}))
        self.assertFalse(detect_wsl(env={}, uname_release="6.5.0-generic"))

    def test_detect_gui_available_linux_without_display(self) -> None:
        self.assertFalse(detect_gui_available(platform_name="linux", env={}, is_wsl=False))

    def test_detect_gui_available_linux_with_display(self) -> None:
        self.assertTrue(detect_gui_available(platform_name="linux", env={"DISPLAY": ":0"}, is_wsl=False))

    def test_configure_matplotlib_runtime_creates_dir(self) -> None:
        path = configure_matplotlib_runtime(env={})
        self.assertTrue(path.exists())


if __name__ == "__main__":
    unittest.main()
