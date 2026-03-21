"""Kinematics models for a 6-DOF dual-arm sorting robot."""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence, Tuple

from .math3d import (
    clamp,
    homogeneous_translation,
    identity,
    mat_mul,
    mat_vec_mul,
    solve_linear_system,
    transpose,
    vector_norm,
    vector_sub,
    zeros,
)

MDHRow = Tuple[float, float, float, float]
Twist = Tuple[float, float, float, float, float, float]
JointLimits = List[Tuple[float, float]]


@dataclass
class IKResult:
    joint_angles: List[float]
    success: bool
    iterations: int
    error_norm: float


@dataclass
class WaypointResult:
    target_position: List[float]
    achieved_position: List[float]
    success: bool
    iterations: int
    error_norm: float


class RobotArm6DOF:
    """Single 6-DOF robot arm with PoE/MDH-based FK and numeric IK."""

    def __init__(
        self,
        name: str,
        mdh_params: Sequence[MDHRow],
        joint_limits: Sequence[Tuple[float, float]],
        base_transform: List[List[float]] | None = None,
        gripper_config: Dict[str, float | str] | None = None,
        kinematics_mode: str = "poe",
        poe_home_m: Sequence[Sequence[float]] | None = None,
        poe_slist: Sequence[Sequence[float]] | None = None,
        poe_axis_points: Sequence[Sequence[float]] | None = None,
        joint_names: Sequence[str] | None = None,
    ) -> None:
        if len(mdh_params) != 6:
            raise ValueError("RobotArm6DOF requires exactly 6 MDH rows")
        if len(joint_limits) != 6:
            raise ValueError("RobotArm6DOF requires 6 joint-limit rows")

        self.name = name
        self.mdh_params = list(mdh_params)
        self.joint_limits: JointLimits = list(joint_limits)
        self.base_transform = base_transform if base_transform is not None else identity(4)
        self.gripper_config = gripper_config or {
            "max_width": 0.10,
            "max_force": 50.0,
            "type": "parallel",
        }
        self.home_joint_angles = [0.0] * 6
        self.joint_names = list(joint_names) if joint_names is not None else [f"J{i+1}" for i in range(6)]

        if kinematics_mode not in {"poe", "mdh"}:
            raise ValueError("kinematics_mode must be 'poe' or 'mdh'")
        self.kinematics_mode = kinematics_mode

        if self.kinematics_mode == "poe" and (
            poe_home_m is None or poe_slist is None or poe_axis_points is None
        ):
            # Keep default constructor usable by auto-loading the official ALOHA PoE model.
            try:
                from .config import OFFICIAL_POE_AXIS_POINTS, OFFICIAL_POE_M, OFFICIAL_POE_SLIST
            except Exception:
                OFFICIAL_POE_AXIS_POINTS = None  # type: ignore[assignment]
                OFFICIAL_POE_M = None  # type: ignore[assignment]
                OFFICIAL_POE_SLIST = None  # type: ignore[assignment]
            if poe_home_m is None and OFFICIAL_POE_M is not None:
                poe_home_m = OFFICIAL_POE_M
            if poe_slist is None and OFFICIAL_POE_SLIST is not None:
                poe_slist = OFFICIAL_POE_SLIST
            if poe_axis_points is None and OFFICIAL_POE_AXIS_POINTS is not None:
                poe_axis_points = OFFICIAL_POE_AXIS_POINTS

        self.poe_home_m = [list(row) for row in poe_home_m] if poe_home_m is not None else None
        self.poe_slist: List[Twist] | None = (
            [tuple(float(v) for v in row) for row in poe_slist] if poe_slist is not None else None
        )
        if self.poe_slist is not None and len(self.poe_slist) != 6:
            raise ValueError("poe_slist must contain 6 screw axes")

        if self.kinematics_mode == "poe" and (self.poe_slist is None or self.poe_home_m is None):
            raise ValueError("PoE mode selected but poe_slist/poe_home_m are missing")

        self.poe_axis_points: List[List[float]] | None = None
        if poe_axis_points is not None:
            if len(poe_axis_points) != 6:
                raise ValueError("poe_axis_points must contain 6 points")
            self.poe_axis_points = [[float(v) for v in pt] for pt in poe_axis_points]
        elif self.poe_slist is not None:
            self.poe_axis_points = [self._axis_point_from_twist(s) for s in self.poe_slist]

    @staticmethod
    def _dot3(a: Sequence[float], b: Sequence[float]) -> float:
        return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]

    @staticmethod
    def _cross3(a: Sequence[float], b: Sequence[float]) -> List[float]:
        return [
            a[1] * b[2] - a[2] * b[1],
            a[2] * b[0] - a[0] * b[2],
            a[0] * b[1] - a[1] * b[0],
        ]

    @staticmethod
    def _norm3(v: Sequence[float]) -> float:
        return math.sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2])

    @staticmethod
    def _skew3(omega: Sequence[float]) -> List[List[float]]:
        return [
            [0.0, -omega[2], omega[1]],
            [omega[2], 0.0, -omega[0]],
            [-omega[1], omega[0], 0.0],
        ]

    @staticmethod
    def _mat3_mul(a: List[List[float]], b: List[List[float]]) -> List[List[float]]:
        return [
            [sum(a[r][k] * b[k][c] for k in range(3)) for c in range(3)]
            for r in range(3)
        ]

    @staticmethod
    def _mat3_vec_mul(a: List[List[float]], v: Sequence[float]) -> List[float]:
        return [sum(a[r][k] * v[k] for k in range(3)) for r in range(3)]

    @staticmethod
    def _mat3_add(a: List[List[float]], b: List[List[float]]) -> List[List[float]]:
        return [[a[r][c] + b[r][c] for c in range(3)] for r in range(3)]

    @staticmethod
    def _mat3_scale(a: List[List[float]], s: float) -> List[List[float]]:
        return [[a[r][c] * s for c in range(3)] for r in range(3)]

    @staticmethod
    def _axis_point_from_twist(screw: Twist) -> List[float]:
        w = [screw[0], screw[1], screw[2]]
        v = [screw[3], screw[4], screw[5]]
        if RobotArm6DOF._norm3(w) < 1e-9:
            return [0.0, 0.0, 0.0]
        # For revolute axis: v = -w x q => q = w x v (with w.q=0 choice).
        return RobotArm6DOF._cross3(w, v)

    @staticmethod
    def _twist_exp_revolute(screw: Twist, theta: float) -> List[List[float]]:
        w = [screw[0], screw[1], screw[2]]
        v = [screw[3], screw[4], screw[5]]

        w_norm = RobotArm6DOF._norm3(w)
        t = identity(4)

        if w_norm < 1e-12:
            t[0][3] = v[0] * theta
            t[1][3] = v[1] * theta
            t[2][3] = v[2] * theta
            return t

        w = [x / w_norm for x in w]
        v = [x / w_norm for x in v]

        wx = RobotArm6DOF._skew3(w)
        wx2 = RobotArm6DOF._mat3_mul(wx, wx)
        i3 = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]

        r = RobotArm6DOF._mat3_add(
            RobotArm6DOF._mat3_add(i3, RobotArm6DOF._mat3_scale(wx, math.sin(theta))),
            RobotArm6DOF._mat3_scale(wx2, 1.0 - math.cos(theta)),
        )

        g = RobotArm6DOF._mat3_add(
            RobotArm6DOF._mat3_add(
                RobotArm6DOF._mat3_scale(i3, theta),
                RobotArm6DOF._mat3_scale(wx, 1.0 - math.cos(theta)),
            ),
            RobotArm6DOF._mat3_scale(wx2, theta - math.sin(theta)),
        )
        p = RobotArm6DOF._mat3_vec_mul(g, v)

        for r_i in range(3):
            for c_i in range(3):
                t[r_i][c_i] = r[r_i][c_i]
        t[0][3], t[1][3], t[2][3] = p[0], p[1], p[2]
        return t

    @staticmethod
    def mdh_transform(theta: float, d: float, a: float, alpha: float) -> List[List[float]]:
        """Modified DH transform: Rot_z(theta) * Trans_z(d) * Trans_x(a) * Rot_x(alpha)."""
        ct = math.cos(theta)
        st = math.sin(theta)
        ca = math.cos(alpha)
        sa = math.sin(alpha)

        return [
            [ct, -st * ca, st * sa, a * ct],
            [st, ct * ca, -ct * sa, a * st],
            [0.0, sa, ca, d],
            [0.0, 0.0, 0.0, 1.0],
        ]

    def set_kinematics_mode(self, mode: str) -> None:
        if mode not in {"poe", "mdh"}:
            raise ValueError("kinematics mode must be 'poe' or 'mdh'")
        self.kinematics_mode = mode

    def joint_axis_mode_report(self) -> List[str]:
        if self.poe_slist is None:
            return [
                "J1: rotate(z)",
                "J2: pitch(y)",
                "J3: pitch(y)",
                "J4: rotate(z)",
                "J5: pitch(y)",
                "J6: rotate(z)",
            ]

        report: List[str] = []
        for i, s in enumerate(self.poe_slist):
            w = [abs(s[0]), abs(s[1]), abs(s[2])]
            idx = max(range(3), key=lambda k: w[k])
            axis = "xyz"[idx]
            motion = "pitch" if axis == "y" else ("roll" if axis == "x" else "rotate")
            report.append(f"{self.joint_names[i]}: {motion}({axis})")
        return report

    def clamp_joint_angles(self, joint_angles: Sequence[float]) -> List[float]:
        return [
            clamp(q, self.joint_limits[i][0], self.joint_limits[i][1])
            for i, q in enumerate(joint_angles)
        ]

    def _forward_kinematics_mdh(
        self,
        joint_angles: Sequence[float],
    ) -> Tuple[List[List[float]], List[List[float]], List[List[List[float]]]]:
        q = self.clamp_joint_angles(joint_angles)
        transforms: List[List[List[float]]] = [self.base_transform]
        joint_positions: List[List[float]] = [
            [self.base_transform[0][3], self.base_transform[1][3], self.base_transform[2][3]]
        ]

        t = [row[:] for row in self.base_transform]
        for i in range(6):
            theta_offset, d, a, alpha = self.mdh_params[i]
            ti = self.mdh_transform(theta_offset + q[i], d, a, alpha)
            t = mat_mul(t, ti)
            transforms.append(t)
            joint_positions.append([t[0][3], t[1][3], t[2][3]])

        return t, joint_positions, transforms

    def _forward_kinematics_poe(
        self,
        joint_angles: Sequence[float],
    ) -> Tuple[List[List[float]], List[List[float]], List[List[List[float]]]]:
        if self.poe_slist is None or self.poe_home_m is None or self.poe_axis_points is None:
            raise ValueError("PoE mode selected but poe_slist/poe_home_m/poe_axis_points are missing")

        q = self.clamp_joint_angles(joint_angles)
        rel = identity(4)

        transforms: List[List[List[float]]] = [self.base_transform]
        joint_positions: List[List[float]] = [
            [self.base_transform[0][3], self.base_transform[1][3], self.base_transform[2][3]]
        ]

        for i in range(6):
            axis_pt_h = self.poe_axis_points[i] + [1.0]
            axis_pt_rel = mat_vec_mul(rel, axis_pt_h)
            axis_pt_world = mat_vec_mul(self.base_transform, axis_pt_rel)
            joint_positions.append(axis_pt_world[:3])

            exp_i = self._twist_exp_revolute(self.poe_slist[i], q[i])
            rel = mat_mul(rel, exp_i)
            transforms.append(mat_mul(self.base_transform, rel))

        end_rel = mat_mul(rel, self.poe_home_m)
        end_tf = mat_mul(self.base_transform, end_rel)
        transforms[-1] = end_tf
        return end_tf, joint_positions, transforms

    def forward_kinematics(
        self,
        joint_angles: Sequence[float],
    ) -> Tuple[List[List[float]], List[List[float]], List[List[List[float]]]]:
        """Return (T_0_6, joint_positions, transforms)."""
        if len(joint_angles) != 6:
            raise ValueError("forward_kinematics expects 6 joint angles")

        if self.kinematics_mode == "poe":
            return self._forward_kinematics_poe(joint_angles)
        return self._forward_kinematics_mdh(joint_angles)

    def end_effector_position(self, joint_angles: Sequence[float]) -> List[float]:
        t, _, _ = self.forward_kinematics(joint_angles)
        return [t[0][3], t[1][3], t[2][3]]

    def numeric_jacobian(self, joint_angles: Sequence[float], eps: float = 1e-5) -> List[List[float]]:
        """3x6 positional Jacobian via finite differences."""
        q = list(joint_angles)
        p0 = self.end_effector_position(q)
        j = zeros(3, 6)

        for i in range(6):
            q_eps = q[:]
            q_eps[i] += eps
            p1 = self.end_effector_position(q_eps)
            for row in range(3):
                j[row][i] = (p1[row] - p0[row]) / eps

        return j

    def inverse_kinematics_position(
        self,
        target_position: Sequence[float],
        initial_angles: Sequence[float] | None = None,
        max_iters: int = 200,
        tol: float = 1e-4,
        damping: float = 1e-3,
        step_size: float = 0.8,
    ) -> IKResult:
        """Damped-least-squares IK for end-effector position only."""
        if len(target_position) != 3:
            raise ValueError("target_position must be 3D")

        if initial_angles is None:
            q = self.home_joint_angles[:]
        else:
            q = self.clamp_joint_angles(initial_angles)

        final_error = float("inf")
        for it in range(max_iters):
            p = self.end_effector_position(q)
            err = vector_sub(target_position, p)
            final_error = vector_norm(err)
            if final_error < tol:
                return IKResult(q, True, it + 1, final_error)

            j = self.numeric_jacobian(q)
            jt = transpose(j)

            a = zeros(6, 6)
            rhs = mat_vec_mul(jt, err)
            for r in range(6):
                for c in range(6):
                    a[r][c] = sum(jt[r][k] * j[k][c] for k in range(3))
                a[r][r] += damping * damping

            try:
                dq = solve_linear_system(a, rhs)
            except ValueError:
                return IKResult(q, False, it + 1, final_error)

            for i in range(6):
                q[i] += step_size * dq[i]
            q = self.clamp_joint_angles(q)

        return IKResult(q, False, max_iters, final_error)

    def inverse_kinematics_position_multistart(
        self,
        target_position: Sequence[float],
        seed_angles: Sequence[float] | None = None,
        attempts: int = 8,
        max_iters: int = 220,
        tol: float = 1e-4,
        damping: float = 1e-3,
        step_size: float = 0.8,
        rng_seed: int = 0,
    ) -> IKResult:
        """Multi-start IK to reduce local-minimum failure in 6-DOF planning."""
        if attempts < 1:
            attempts = 1

        rng = random.Random(rng_seed)
        initial_guesses: List[List[float]] = []
        if seed_angles is not None:
            initial_guesses.append(self.clamp_joint_angles(seed_angles))
        else:
            initial_guesses.append(self.home_joint_angles[:])

        while len(initial_guesses) < attempts:
            guess = [
                rng.uniform(self.joint_limits[i][0], self.joint_limits[i][1])
                for i in range(6)
            ]
            initial_guesses.append(guess)

        best: IKResult | None = None
        for guess in initial_guesses:
            res = self.inverse_kinematics_position(
                target_position=target_position,
                initial_angles=guess,
                max_iters=max_iters,
                tol=tol,
                damping=damping,
                step_size=step_size,
            )
            if best is None or res.error_norm < best.error_norm:
                best = res
            if res.success:
                return res

        if best is None:
            raise RuntimeError("Unexpected IK failure: no attempts were executed")
        return best

    @staticmethod
    def interpolate_joint_trajectory(
        start_angles: Sequence[float],
        end_angles: Sequence[float],
        steps: int,
    ) -> List[List[float]]:
        if len(start_angles) != 6 or len(end_angles) != 6:
            raise ValueError("interpolate_joint_trajectory expects 6-DOF vectors")
        if steps < 2:
            return [list(end_angles)]

        out: List[List[float]] = []
        for i in range(steps):
            t = i / (steps - 1)
            out.append([
                (1.0 - t) * start_angles[j] + t * end_angles[j]
                for j in range(6)
            ])
        return out


class DualArmSystem:
    """Dual 6-DOF arm system for pick-and-place sorting simulation."""

    def __init__(
        self,
        mdh_params: Sequence[MDHRow],
        joint_limits: Sequence[Tuple[float, float]],
        arm_offset: Sequence[float],
        gripper_config: Dict[str, float | str] | None = None,
        kinematics_mode: str = "poe",
        poe_home_m: Sequence[Sequence[float]] | None = None,
        poe_slist: Sequence[Sequence[float]] | None = None,
        poe_axis_points: Sequence[Sequence[float]] | None = None,
        joint_names: Sequence[str] | None = None,
    ) -> None:
        if len(arm_offset) != 3:
            raise ValueError("arm_offset must be [x, y, z]")

        if kinematics_mode == "poe" and (
            poe_home_m is None or poe_slist is None or poe_axis_points is None
        ):
            try:
                from .config import OFFICIAL_POE_AXIS_POINTS, OFFICIAL_POE_M, OFFICIAL_POE_SLIST
            except Exception:
                OFFICIAL_POE_AXIS_POINTS = None  # type: ignore[assignment]
                OFFICIAL_POE_M = None  # type: ignore[assignment]
                OFFICIAL_POE_SLIST = None  # type: ignore[assignment]

            if poe_home_m is None and OFFICIAL_POE_M is not None:
                poe_home_m = OFFICIAL_POE_M
            if poe_slist is None and OFFICIAL_POE_SLIST is not None:
                poe_slist = OFFICIAL_POE_SLIST
            if poe_axis_points is None and OFFICIAL_POE_AXIS_POINTS is not None:
                poe_axis_points = OFFICIAL_POE_AXIS_POINTS

        left_base = identity(4)
        right_base = homogeneous_translation(arm_offset[0], arm_offset[1], arm_offset[2])

        self.left_arm = RobotArm6DOF(
            "left_arm",
            mdh_params,
            joint_limits,
            base_transform=left_base,
            gripper_config=gripper_config,
            kinematics_mode=kinematics_mode,
            poe_home_m=poe_home_m,
            poe_slist=poe_slist,
            poe_axis_points=poe_axis_points,
            joint_names=joint_names,
        )
        self.right_arm = RobotArm6DOF(
            "right_arm",
            mdh_params,
            joint_limits,
            base_transform=right_base,
            gripper_config=gripper_config,
            kinematics_mode=kinematics_mode,
            poe_home_m=poe_home_m,
            poe_slist=poe_slist,
            poe_axis_points=poe_axis_points,
            joint_names=joint_names,
        )

    def joint_axis_mode_report(self) -> Dict[str, List[str]]:
        return {
            "left": self.left_arm.joint_axis_mode_report(),
            "right": self.right_arm.joint_axis_mode_report(),
        }

    def forward_both(self, left_q: Sequence[float], right_q: Sequence[float]) -> Dict[str, List[List[float]]]:
        left_t, left_pos, _ = self.left_arm.forward_kinematics(left_q)
        right_t, right_pos, _ = self.right_arm.forward_kinematics(right_q)
        return {
            "left_end_transform": left_t,
            "left_joint_positions": left_pos,
            "right_end_transform": right_t,
            "right_joint_positions": right_pos,
        }

    @staticmethod
    def _stitch_segments(segments: Iterable[List[List[float]]]) -> List[List[float]]:
        full: List[List[float]] = []
        for seg in segments:
            if not seg:
                continue
            if not full:
                full.extend(seg)
            else:
                full.extend(seg[1:])
        return full

    def plan_pick_and_place(
        self,
        arm: RobotArm6DOF,
        pick_pos: Sequence[float],
        place_pos: Sequence[float],
        start_q: Sequence[float] | None = None,
        transit_lift: float = 0.12,
        segment_steps: int = 24,
        ik_attempts: int = 8,
        max_waypoint_error: float = 0.03,
        strict_ik: bool = False,
        return_report: bool = False,
    ) -> List[List[float]] | Tuple[List[List[float]], List[WaypointResult]]:
        q0 = list(start_q) if start_q is not None else arm.home_joint_angles[:]

        above_pick = [pick_pos[0], pick_pos[1], pick_pos[2] + transit_lift]
        above_place = [place_pos[0], place_pos[1], place_pos[2] + transit_lift]

        waypoints = [above_pick, pick_pos, above_pick, above_place, place_pos, above_place]
        joint_waypoints: List[List[float]] = [q0]
        waypoint_results: List[WaypointResult] = []
        q_curr = q0

        for idx, wp in enumerate(waypoints):
            ik = arm.inverse_kinematics_position_multistart(
                wp,
                seed_angles=q_curr,
                attempts=ik_attempts,
                rng_seed=idx,
            )
            q_curr = ik.joint_angles
            joint_waypoints.append(q_curr)
            achieved = arm.end_effector_position(q_curr)
            err = math.sqrt(sum((wp[i] - achieved[i]) ** 2 for i in range(3)))
            waypoint_results.append(
                WaypointResult(
                    target_position=list(wp),
                    achieved_position=achieved,
                    success=ik.success and err <= max_waypoint_error,
                    iterations=ik.iterations,
                    error_norm=err,
                )
            )
            if strict_ik and err > max_waypoint_error:
                raise ValueError(
                    f"{arm.name} IK failed at waypoint {idx}: "
                    f"error={err:.4f}m > limit {max_waypoint_error:.4f}m"
                )

        segments: List[List[List[float]]] = []
        for i in range(len(joint_waypoints) - 1):
            segments.append(
                arm.interpolate_joint_trajectory(
                    joint_waypoints[i],
                    joint_waypoints[i + 1],
                    segment_steps,
                )
            )

        trajectory = self._stitch_segments(segments)
        if return_report:
            return trajectory, waypoint_results
        return trajectory

    def synchronized_sort_trajectories(
        self,
        defect_pick: Sequence[float],
        defect_place: Sequence[float],
        normal_pick: Sequence[float],
        normal_place: Sequence[float],
        strict_ik: bool = False,
        return_report: bool = False,
    ) -> Tuple[List[List[float]], List[List[float]]] | Tuple[
        List[List[float]], List[List[float]], Dict[str, List[WaypointResult]]
    ]:
        left_traj, left_report = self.plan_pick_and_place(
            self.left_arm,
            normal_pick,
            normal_place,
            strict_ik=strict_ik,
            return_report=True,
        )
        right_traj, right_report = self.plan_pick_and_place(
            self.right_arm,
            defect_pick,
            defect_place,
            strict_ik=strict_ik,
            return_report=True,
        )

        max_len = max(len(left_traj), len(right_traj))
        if len(left_traj) < max_len:
            left_traj.extend([left_traj[-1]] * (max_len - len(left_traj)))
        if len(right_traj) < max_len:
            right_traj.extend([right_traj[-1]] * (max_len - len(right_traj)))

        if return_report:
            return left_traj, right_traj, {"left": left_report, "right": right_report}
        return left_traj, right_traj
