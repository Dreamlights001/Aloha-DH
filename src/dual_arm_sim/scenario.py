"""Sorting scenario simulation and visualization for dual-arm robot."""

from __future__ import annotations

import math
import random
import time
from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

from .config import SORTING_BINS
from .math3d import vector_add, vector_scale
from .robot import DualArmSystem, RobotArm6DOF


@dataclass
class SortingObject:
    position: List[float]
    defective: bool
    radius: float


class SortingScenario:
    def __init__(
        self,
        dual_arm_system: DualArmSystem,
        conveyor_config: Dict[str, float],
        camera_config: Dict[str, Dict[str, Tuple[float, float, float]]],
    ) -> None:
        self.dual_arm_system = dual_arm_system
        self.conveyor_config = conveyor_config
        self.camera_config = camera_config
        self.products: List[SortingObject] = []

    def generate_products(
        self,
        count: int = 10,
        defective_ratio: float = 0.25,
        seed: int = 7,
    ) -> List[SortingObject]:
        rng = random.Random(seed)
        self.products = []

        x_min = 0.10
        x_max = max(x_min + 0.1, self.conveyor_config["length"] - 0.10)
        y_span = self.conveyor_config["width"] * 0.38
        z = self.conveyor_config["height"] + 0.03

        for _ in range(count):
            x = rng.uniform(x_min, x_max)
            y = rng.uniform(-y_span, y_span)
            defective = rng.random() < defective_ratio
            radius = 0.028 if defective else 0.024
            self.products.append(
                SortingObject(position=[x, y, z], defective=defective, radius=radius)
            )
        return self.products

    def _pick_targets(self) -> Tuple[List[float], List[float]]:
        defect = next((obj for obj in self.products if obj.defective), None)
        normal = next((obj for obj in self.products if not obj.defective), None)

        default_z = self.conveyor_config["height"] + 0.03
        defect_pos = defect.position[:] if defect is not None else [0.75, 0.05, default_z]
        normal_pos = normal.position[:] if normal is not None else [0.35, -0.05, default_z]
        return defect_pos, normal_pos

    def plan_sorting_trajectories(
        self,
        strict_ik: bool = False,
        return_report: bool = False,
    ) -> Tuple[List[List[float]], List[List[float]]] | Tuple[
        List[List[float]], List[List[float]], Dict[str, list]
    ]:
        defect_pick, normal_pick = self._pick_targets()
        defect_place = list(SORTING_BINS["defect"])
        normal_place = list(SORTING_BINS["normal"])

        return self.dual_arm_system.synchronized_sort_trajectories(
            defect_pick=defect_pick,
            defect_place=defect_place,
            normal_pick=normal_pick,
            normal_place=normal_place,
            strict_ik=strict_ik,
            return_report=return_report,
        )

    @staticmethod
    def _require_matplotlib() -> None:
        try:
            import matplotlib  # noqa: F401
        except Exception as exc:
            raise RuntimeError(
                "matplotlib is required for visualization. Install it with: pip install matplotlib"
            ) from exc

    @staticmethod
    def _require_pybullet() -> None:
        try:
            import pybullet  # noqa: F401
        except Exception as exc:
            raise RuntimeError(
                "pybullet is required for realtime visualization. Install it with: pip install pybullet"
            ) from exc

    @staticmethod
    def _draw_coordinate_frame(ax, transform: List[List[float]], axis_len: float = 0.06) -> None:
        origin = [transform[0][3], transform[1][3], transform[2][3]]
        x_axis = [transform[0][0], transform[1][0], transform[2][0]]
        y_axis = [transform[0][1], transform[1][1], transform[2][1]]
        z_axis = [transform[0][2], transform[1][2], transform[2][2]]

        x_end = vector_add(origin, vector_scale(x_axis, axis_len))
        y_end = vector_add(origin, vector_scale(y_axis, axis_len))
        z_end = vector_add(origin, vector_scale(z_axis, axis_len))

        ax.plot([origin[0], x_end[0]], [origin[1], x_end[1]], [origin[2], x_end[2]], c="r", lw=2)
        ax.plot([origin[0], y_end[0]], [origin[1], y_end[1]], [origin[2], y_end[2]], c="g", lw=2)
        ax.plot([origin[0], z_end[0]], [origin[1], z_end[1]], [origin[2], z_end[2]], c="b", lw=2)

    @staticmethod
    def _draw_cube(ax, center: Sequence[float], half: float, color: str) -> None:
        from mpl_toolkits.mplot3d.art3d import Poly3DCollection

        x, y, z = center
        h = half
        v = [
            [x - h, y - h, z - h], [x + h, y - h, z - h], [x + h, y + h, z - h], [x - h, y + h, z - h],
            [x - h, y - h, z + h], [x + h, y - h, z + h], [x + h, y + h, z + h], [x - h, y + h, z + h],
        ]
        faces = [
            [v[0], v[1], v[2], v[3]],
            [v[4], v[5], v[6], v[7]],
            [v[0], v[1], v[5], v[4]],
            [v[1], v[2], v[6], v[5]],
            [v[2], v[3], v[7], v[6]],
            [v[3], v[0], v[4], v[7]],
        ]
        ax.add_collection3d(Poly3DCollection(faces, alpha=0.7, facecolor=color, edgecolor="black", linewidths=0.6))

    @staticmethod
    def _draw_tetra(ax, center: Sequence[float], scale: float, color: str) -> None:
        from mpl_toolkits.mplot3d.art3d import Poly3DCollection

        x, y, z = center
        a = scale
        v0 = [x, y, z + a]
        v1 = [x + a, y - a, z - a]
        v2 = [x - a, y - a, z - a]
        v3 = [x, y + a, z - a]
        faces = [[v0, v1, v2], [v0, v2, v3], [v0, v3, v1], [v1, v3, v2]]
        ax.add_collection3d(Poly3DCollection(faces, alpha=0.75, facecolor=color, edgecolor="black", linewidths=0.7))

    def _draw_arm(
        self,
        ax,
        arm: RobotArm6DOF,
        joint_angles: Sequence[float],
        gripper_closed: bool,
    ) -> None:
        _, positions, transforms = arm.forward_kinematics(joint_angles)

        xs = [p[0] for p in positions]
        ys = [p[1] for p in positions]
        zs = [p[2] for p in positions]

        arm_color = "darkorange" if "right" not in arm.name else "gold"
        joint_color = "gray"
        ax.plot(xs, ys, zs, color=arm_color, lw=4.0)

        for p in positions:
            ax.scatter([p[0]], [p[1]], [p[2]], c=joint_color, s=40)

        ee_tf = transforms[-1]
        self._draw_coordinate_frame(ax, ee_tf)

        ee = [ee_tf[0][3], ee_tf[1][3], ee_tf[2][3]]
        x_axis = [ee_tf[0][0], ee_tf[1][0], ee_tf[2][0]]
        y_axis = [ee_tf[0][1], ee_tf[1][1], ee_tf[2][1]]

        max_width = float(arm.gripper_config.get("max_width", 0.10))
        jaw = max_width * (0.2 if gripper_closed else 1.0)
        finger_len = 0.05

        f1_root = vector_add(ee, vector_scale(y_axis, jaw * 0.5))
        f2_root = vector_add(ee, vector_scale(y_axis, -jaw * 0.5))
        f1_tip = vector_add(f1_root, vector_scale(x_axis, finger_len))
        f2_tip = vector_add(f2_root, vector_scale(x_axis, finger_len))

        gripper_color = "orange" if gripper_closed else "gold"
        ax.plot([f1_root[0], f1_tip[0]], [f1_root[1], f1_tip[1]], [f1_root[2], f1_tip[2]], c=gripper_color, lw=4)
        ax.plot([f2_root[0], f2_tip[0]], [f2_root[1], f2_tip[1]], [f2_root[2], f2_tip[2]], c=gripper_color, lw=4)

    def _draw_conveyor(self, ax) -> None:
        self._require_matplotlib()
        from mpl_toolkits.mplot3d.art3d import Poly3DCollection

        length = self.conveyor_config["length"]
        width = self.conveyor_config["width"]
        height = self.conveyor_config["height"]

        x0 = 0.0
        x1 = length
        y0 = -width * 0.5
        y1 = width * 0.5
        z = height

        vertices = [[(x0, y0, z), (x1, y0, z), (x1, y1, z), (x0, y1, z)]]
        poly = Poly3DCollection(vertices, alpha=0.35, facecolor="lightgreen", edgecolor="green")
        ax.add_collection3d(poly)

    def _draw_products(self, ax) -> None:
        for obj in self.products:
            if obj.defective:
                self._draw_tetra(ax, obj.position, obj.radius, color="red")
            else:
                self._draw_cube(ax, obj.position, obj.radius, color="green")

    def _draw_cameras(self, ax) -> None:
        for camera in self.camera_config.values():
            p = camera["position"]
            ax.scatter([p[0]], [p[1]], [p[2]], c="yellow", s=40)

            depth = 0.25
            half = 0.08
            corners = [
                (p[0] - half, p[1] - half, p[2] - depth),
                (p[0] + half, p[1] - half, p[2] - depth),
                (p[0] + half, p[1] + half, p[2] - depth),
                (p[0] - half, p[1] + half, p[2] - depth),
            ]
            for c in corners:
                ax.plot([p[0], c[0]], [p[1], c[1]], [p[2], c[2]], c="yellow", lw=1)
            for i in range(4):
                c0 = corners[i]
                c1 = corners[(i + 1) % 4]
                ax.plot([c0[0], c1[0]], [c0[1], c1[1]], [c0[2], c1[2]], c="yellow", lw=1)

    def _draw_scene_static(self, ax) -> None:
        self._draw_conveyor(ax)
        self._draw_products(ax)
        self._draw_cameras(ax)

        normal_bin = SORTING_BINS["normal"]
        defect_bin = SORTING_BINS["defect"]
        ax.scatter([normal_bin[0]], [normal_bin[1]], [normal_bin[2]], c="green", s=80, marker="s")
        ax.scatter([defect_bin[0]], [defect_bin[1]], [defect_bin[2]], c="red", s=80, marker="^")

    @staticmethod
    def _setup_axes(ax) -> None:
        ax.set_xlabel("X (m)")
        ax.set_ylabel("Y (m)")
        ax.set_zlabel("Z (m)")
        ax.set_xlim(-0.15, 1.25)
        ax.set_ylim(-0.55, 0.55)
        ax.set_zlim(0.0, 1.1)
        ax.set_title("Dual-Arm Defect Sorting Simulation")

        if hasattr(ax, "set_box_aspect"):
            ax.set_box_aspect((1.4, 1.1, 1.0))

    def animate_sorting(
        self,
        save_path: str | None = None,
        interval_ms: int = 80,
        left_traj: List[List[float]] | None = None,
        right_traj: List[List[float]] | None = None,
        show_window: bool | None = None,
    ) -> None:
        """Matplotlib animation mode."""
        self._require_matplotlib()

        import matplotlib.pyplot as plt
        from matplotlib.animation import FuncAnimation

        if not self.products:
            self.generate_products()

        if left_traj is None or right_traj is None:
            left_traj, right_traj = self.plan_sorting_trajectories()
        frame_count = min(len(left_traj), len(right_traj))

        fig = plt.figure(figsize=(11, 7))
        ax = fig.add_subplot(111, projection="3d")

        close_start = int(frame_count * 0.20)
        close_end = int(frame_count * 0.72)

        def update(frame: int):
            ax.cla()
            self._setup_axes(ax)
            self._draw_scene_static(ax)

            closed = close_start <= frame <= close_end

            self._draw_arm(ax, self.dual_arm_system.left_arm, left_traj[frame], gripper_closed=closed)
            self._draw_arm(ax, self.dual_arm_system.right_arm, right_traj[frame], gripper_closed=closed)

            ax.text2D(0.02, 0.96, f"Frame: {frame + 1}/{frame_count}", transform=ax.transAxes)

        animation = FuncAnimation(fig, update, frames=frame_count, interval=interval_ms, repeat=True)

        if save_path:
            animation.save(str(save_path), dpi=120)

        should_show = (save_path is None) if show_window is None else show_window
        if should_show:
            plt.tight_layout()
            plt.show()
        else:
            plt.close(fig)

    @staticmethod
    def _pybullet_add_box_wireframe(
        p,
        center: Sequence[float],
        half_extents: Sequence[float],
        color: Sequence[float],
        width: float = 2.0,
    ) -> List[int]:
        x, y, z = center
        hx, hy, hz = half_extents
        pts = [
            [x - hx, y - hy, z - hz], [x + hx, y - hy, z - hz], [x + hx, y + hy, z - hz], [x - hx, y + hy, z - hz],
            [x - hx, y - hy, z + hz], [x + hx, y - hy, z + hz], [x + hx, y + hy, z + hz], [x - hx, y + hy, z + hz],
        ]
        edges = [
            (0, 1), (1, 2), (2, 3), (3, 0),
            (4, 5), (5, 6), (6, 7), (7, 4),
            (0, 4), (1, 5), (2, 6), (3, 7),
        ]
        ids: List[int] = []
        for a, b in edges:
            ids.append(p.addUserDebugLine(pts[a], pts[b], color, width, lifeTime=0))
        return ids

    @staticmethod
    def _pybullet_add_cube_wireframe(
        p,
        center: Sequence[float],
        half: float,
        color: Sequence[float],
        width: float = 2.0,
    ) -> List[int]:
        return SortingScenario._pybullet_add_box_wireframe(
            p,
            center=center,
            half_extents=[half, half, half],
            color=color,
            width=width,
        )

    @staticmethod
    def _pybullet_add_tetra_wireframe(p, center: Sequence[float], scale: float, color: Sequence[float], width: float = 2.0) -> List[int]:
        x, y, z = center
        a = scale
        v = [
            [x, y, z + a],
            [x + a, y - a, z - a],
            [x - a, y - a, z - a],
            [x, y + a, z - a],
        ]
        edges = [(0, 1), (0, 2), (0, 3), (1, 2), (2, 3), (3, 1)]
        ids: List[int] = []
        for e0, e1 in edges:
            ids.append(p.addUserDebugLine(v[e0], v[e1], color, width, lifeTime=0))
        return ids

    @staticmethod
    def _pybullet_add_target_marker(p, center: Sequence[float], color: Sequence[float]) -> List[int]:
        x, y, z = center
        d = 0.03
        ids = [
            p.addUserDebugLine([x - d, y, z], [x + d, y, z], color, 3.5, lifeTime=0),
            p.addUserDebugLine([x, y - d, z], [x, y + d, z], color, 3.5, lifeTime=0),
            p.addUserDebugLine([x, y, z - d], [x, y, z + d], color, 3.5, lifeTime=0),
        ]
        return ids

    @staticmethod
    def _screen_to_ray(mouse_x: int, mouse_y: int, cam_info) -> Tuple[List[float], List[float]]:
        import numpy as np

        width = max(1, int(cam_info[0]))
        height = max(1, int(cam_info[1]))
        view = np.array(cam_info[2], dtype=float).reshape((4, 4), order="F")
        proj = np.array(cam_info[3], dtype=float).reshape((4, 4), order="F")
        inv = np.linalg.inv(proj @ view)

        x = 2.0 * (mouse_x / width) - 1.0
        y = 1.0 - 2.0 * (mouse_y / height)

        near_clip = np.array([x, y, -1.0, 1.0], dtype=float)
        far_clip = np.array([x, y, 1.0, 1.0], dtype=float)

        near_world = inv @ near_clip
        far_world = inv @ far_clip
        near_world /= near_world[3]
        far_world /= far_world[3]
        return near_world[:3].tolist(), far_world[:3].tolist()

    @staticmethod
    def _ray_plane_intersection(ray_from: Sequence[float], ray_to: Sequence[float], z_plane: float) -> List[float] | None:
        direction = [ray_to[i] - ray_from[i] for i in range(3)]
        dz = direction[2]
        if abs(dz) < 1e-9:
            return None
        t = (z_plane - ray_from[2]) / dz
        if t < 0:
            return None
        return [ray_from[i] + t * direction[i] for i in range(3)]

    def visualize_pybullet(
        self,
        left_traj: List[List[float]],
        right_traj: List[List[float]],
        fps: int = 30,
        realtime: bool = True,
        loop: bool = True,
        prefer_gui: bool = True,
        allow_headless_fallback: bool = True,
        enable_drag_target: bool = False,
    ) -> str:
        """Realtime interactive visualization mode via PyBullet GUI."""
        self._require_pybullet()

        import pybullet as p
        import pybullet_data

        connection_mode_name = "GUI" if prefer_gui else "DIRECT"
        client = p.connect(p.GUI if prefer_gui else p.DIRECT)
        if client < 0 and prefer_gui and allow_headless_fallback:
            client = p.connect(p.DIRECT)
            connection_mode_name = "DIRECT"
        if client < 0:
            raise RuntimeError(
                "Failed to open PyBullet connection (GUI/DIRECT). "
                "Check WSL GUI or install an X/Wayland display server."
            )

        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.81)
        p.loadURDF("plane.urdf")
        if not self.products:
            self.generate_products()

        if connection_mode_name == "GUI":
            p.resetDebugVisualizerCamera(
                cameraDistance=1.7,
                cameraYaw=45,
                cameraPitch=-25,
                cameraTargetPosition=[0.55, 0.0, 0.25],
            )

        frame_count = min(len(left_traj), len(right_traj))
        if frame_count == 0:
            raise ValueError("left_traj/right_traj must contain at least one frame")

        left_q_live = left_traj[0][:]
        right_q_live = right_traj[0][:]
        selected_arm = "left"
        selected_joint_idx: int | None = None
        selected_joint_target: List[float] | None = None
        drag_active = False
        drag_plane_z = self.conveyor_config["height"] + 0.2
        last_drag_error = 0.0
        last_drag_success = False

        mouse_button_event = getattr(p, "MOUSE_BUTTON_EVENT", 2)
        mouse_move_event = getattr(p, "MOUSE_MOVE_EVENT", 1)
        mouse_left = getattr(p, "MOUSE_BUTTON_LEFT", 0)
        key_is_down = getattr(p, "KEY_IS_DOWN", 1)

        static_ids: List[int] = []
        dynamic_line_ids: Dict[str, int] = {}
        dynamic_text_ids: Dict[str, int] = {}

        def upsert_line(
            key: str,
            start: Sequence[float],
            end: Sequence[float],
            color: Sequence[float],
            width: float,
        ) -> None:
            previous = dynamic_line_ids.get(key, -1)
            try:
                new_id = p.addUserDebugLine(
                    start,
                    end,
                    color,
                    width,
                    lifeTime=0,
                    replaceItemUniqueId=previous,
                )
            except TypeError:
                if previous >= 0:
                    p.removeUserDebugItem(previous)
                new_id = p.addUserDebugLine(start, end, color, width, lifeTime=0)
            dynamic_line_ids[key] = new_id

        def upsert_text(
            key: str,
            text: str,
            position: Sequence[float],
            color: Sequence[float],
            size: float,
        ) -> None:
            previous = dynamic_text_ids.get(key, -1)
            try:
                new_id = p.addUserDebugText(
                    text,
                    position,
                    textColorRGB=color,
                    textSize=size,
                    lifeTime=0,
                    replaceItemUniqueId=previous,
                )
            except TypeError:
                if previous >= 0:
                    p.removeUserDebugItem(previous)
                new_id = p.addUserDebugText(text, position, textColorRGB=color, textSize=size, lifeTime=0)
            dynamic_text_ids[key] = new_id

        def draw_scene_objects_once() -> None:
            length = self.conveyor_config["length"]
            width = self.conveyor_config["width"]
            height = self.conveyor_config["height"]
            static_ids.extend(
                self._pybullet_add_box_wireframe(
                    p,
                    center=[length * 0.5, 0.0, height * 0.5],
                    half_extents=[0.5 * length, 0.5 * width, 0.5 * height],
                    color=[0.2, 0.8, 0.2],
                    width=1.2,
                )
            )
            for obj in self.products:
                if obj.defective:
                    static_ids.extend(
                        self._pybullet_add_tetra_wireframe(
                            p, obj.position, obj.radius, [1.0, 0.0, 0.0], width=2.2
                        )
                    )
                else:
                    static_ids.extend(
                        self._pybullet_add_cube_wireframe(
                            p, obj.position, obj.radius, [0.0, 0.9, 0.0], width=2.2
                        )
                    )

        def draw_arm_state(
            arm_key: str,
            arm: RobotArm6DOF,
            q: Sequence[float],
            gripper_closed: bool,
        ) -> List[List[float]]:
            _, positions, transforms = arm.forward_kinematics(q)
            arm_color = [1.0, 0.55, 0.0] if arm_key == "left" else [1.0, 1.0, 0.0]
            joint_color = [0.55, 0.55, 0.55]
            for i in range(len(positions) - 1):
                upsert_line(
                    f"{arm_key}_link_{i}",
                    positions[i],
                    positions[i + 1],
                    arm_color,
                    4.2,
                )
            for i, pos in enumerate(positions):
                marker_half = 0.006
                marker_color = joint_color
                marker_width = 5.0
                if arm_key == selected_arm and selected_joint_idx == i:
                    marker_half = 0.014
                    marker_color = [0.1, 0.9, 1.0]
                    marker_width = 6.0
                upsert_line(
                    f"{arm_key}_joint_{i}",
                    [pos[0], pos[1], pos[2] - marker_half],
                    [pos[0], pos[1], pos[2] + marker_half],
                    marker_color,
                    marker_width,
                )

            ee_tf = transforms[-1]
            ee = [ee_tf[0][3], ee_tf[1][3], ee_tf[2][3]]
            x_axis = [ee_tf[0][0], ee_tf[1][0], ee_tf[2][0]]
            y_axis = [ee_tf[0][1], ee_tf[1][1], ee_tf[2][1]]
            jaw = float(arm.gripper_config.get("max_width", 0.10)) * (0.2 if gripper_closed else 1.0)

            f1_root = vector_add(ee, vector_scale(y_axis, jaw * 0.5))
            f2_root = vector_add(ee, vector_scale(y_axis, -jaw * 0.5))
            f1_tip = vector_add(f1_root, vector_scale(x_axis, 0.05))
            f2_tip = vector_add(f2_root, vector_scale(x_axis, 0.05))
            grip_col = arm_color if gripper_closed else [min(1.0, arm_color[0] + 0.1), min(1.0, arm_color[1] + 0.1), 0.2]
            upsert_line(f"{arm_key}_grip_0", f1_root, f1_tip, grip_col, 4.0)
            upsert_line(f"{arm_key}_grip_1", f2_root, f2_tip, grip_col, 4.0)
            return positions

        def draw_status(text: str) -> None:
            upsert_text("status", text, [0.02, -0.52, 1.03], [1, 1, 1], 1.2)

        def draw_target_marker(center: Sequence[float] | None, color: Sequence[float]) -> None:
            if center is None:
                center = [0.0, 0.0, -10.0]
            d = 0.03
            upsert_line("target_x", [center[0] - d, center[1], center[2]], [center[0] + d, center[1], center[2]], color, 3.0)
            upsert_line("target_y", [center[0], center[1] - d, center[2]], [center[0], center[1] + d, center[2]], color, 3.0)
            upsert_line("target_z", [center[0], center[1], center[2] - d], [center[0], center[1], center[2] + d], color, 3.0)

        def pick_joint_index(
            arm: RobotArm6DOF,
            q: Sequence[float],
            ray_from: Sequence[float],
            ray_to: Sequence[float],
            threshold: float = 0.06,
        ) -> Tuple[int | None, List[float] | None]:
            _, positions, _ = arm.forward_kinematics(q)
            direction = [ray_to[i] - ray_from[i] for i in range(3)]
            ray_len = math.sqrt(sum(v * v for v in direction))
            if ray_len < 1e-9:
                return None, None
            direction = [v / ray_len for v in direction]

            best_idx: int | None = None
            best_dist = float("inf")
            best_pos: List[float] | None = None

            for idx in range(1, len(positions)):
                pos = positions[idx]
                v = [pos[i] - ray_from[i] for i in range(3)]
                t = sum(v[i] * direction[i] for i in range(3))
                if t < 0.0 or t > ray_len:
                    continue
                closest = [ray_from[i] + t * direction[i] for i in range(3)]
                dist = math.sqrt(sum((pos[i] - closest[i]) ** 2 for i in range(3)))
                if dist < best_dist:
                    best_dist = dist
                    best_idx = idx
                    best_pos = pos[:]

            if best_idx is None or best_dist > threshold:
                return None, None
            return best_idx, best_pos

        draw_scene_objects_once()

        frame = 0
        paused = False
        step_once = False

        try:
            while p.isConnected():
                interactive_drag = enable_drag_target and connection_mode_name == "GUI"
                keys = p.getKeyboardEvents() if connection_mode_name == "GUI" else {}

                if connection_mode_name == "GUI":
                    if ord(" ") in keys and keys[ord(" ")] & p.KEY_WAS_TRIGGERED:
                        paused = not paused
                    if ord("n") in keys and keys[ord("n")] & p.KEY_WAS_TRIGGERED:
                        step_once = True
                    if ord("r") in keys and keys[ord("r")] & p.KEY_WAS_TRIGGERED:
                        frame = 0
                        left_q_live = left_traj[0][:]
                        right_q_live = right_traj[0][:]
                        selected_joint_idx = None
                        selected_joint_target = None
                        drag_active = False
                    tab_codes = [9, ord("\t")]
                    tab_const = getattr(p, "B3G_TAB", None)
                    if isinstance(tab_const, int):
                        tab_codes.append(tab_const)
                    if any(code in keys and keys[code] & p.KEY_WAS_TRIGGERED for code in tab_codes):
                        selected_arm = "right" if selected_arm == "left" else "left"

                    escape_codes = [27]
                    for attr_name in ("B3G_ESCAPE", "B3G_ESC"):
                        code = getattr(p, attr_name, None)
                        if isinstance(code, int):
                            escape_codes.append(code)
                    escape_triggered = any(
                        (code in keys and keys[code] & p.KEY_WAS_TRIGGERED)
                        for code in escape_codes
                    )
                    if (ord("q") in keys and keys[ord("q")] & p.KEY_WAS_TRIGGERED) or escape_triggered:
                        break

                if interactive_drag:
                    mouse_events = p.getMouseEvents()
                    for ev in mouse_events:
                        if len(ev) < 5:
                            continue
                        event_type, mx, my, button_idx, button_state = ev[0], ev[1], ev[2], ev[3], ev[4]
                        if event_type == mouse_button_event and button_idx == mouse_left:
                            if (button_state & p.KEY_WAS_TRIGGERED) or (button_state & key_is_down):
                                try:
                                    cam_info = p.getDebugVisualizerCamera()
                                    ray_from, ray_to = self._screen_to_ray(mx, my, cam_info)
                                    if selected_arm == "left":
                                        joint_idx, joint_pos = pick_joint_index(
                                            self.dual_arm_system.left_arm,
                                            left_q_live,
                                            ray_from,
                                            ray_to,
                                        )
                                    else:
                                        joint_idx, joint_pos = pick_joint_index(
                                            self.dual_arm_system.right_arm,
                                            right_q_live,
                                            ray_from,
                                            ray_to,
                                        )
                                    if joint_idx is not None and joint_pos is not None:
                                        selected_joint_idx = joint_idx
                                        selected_joint_target = joint_pos[:]
                                        drag_plane_z = joint_pos[2]
                                        drag_active = True
                                except Exception:
                                    pass
                            if button_state & p.KEY_WAS_RELEASED:
                                drag_active = False
                        if (
                            drag_active
                            and selected_joint_idx is not None
                            and event_type in (mouse_move_event, mouse_button_event)
                        ):
                            try:
                                cam_info = p.getDebugVisualizerCamera()
                                ray_from, ray_to = self._screen_to_ray(mx, my, cam_info)
                                hit = self._ray_plane_intersection(ray_from, ray_to, drag_plane_z)
                                if hit is not None:
                                    selected_joint_target = hit[:]
                                    active_dofs = list(range(min(6, max(1, selected_joint_idx))))
                                    if selected_arm == "left":
                                        ik = self.dual_arm_system.left_arm.inverse_kinematics_point_position(
                                            selected_joint_target,
                                            point_index=selected_joint_idx,
                                            initial_angles=left_q_live,
                                            active_dofs=active_dofs,
                                            max_iters=80,
                                            tol=8e-4,
                                            damping=5e-3,
                                            step_size=0.75,
                                        )
                                        left_q_live = ik.joint_angles
                                    else:
                                        ik = self.dual_arm_system.right_arm.inverse_kinematics_point_position(
                                            selected_joint_target,
                                            point_index=selected_joint_idx,
                                            initial_angles=right_q_live,
                                            active_dofs=active_dofs,
                                            max_iters=80,
                                            tol=8e-4,
                                            damping=5e-3,
                                            step_size=0.75,
                                        )
                                        right_q_live = ik.joint_angles
                                    last_drag_error = ik.error_norm
                                    last_drag_success = ik.success
                            except Exception:
                                pass

                    draw_arm_state("left", self.dual_arm_system.left_arm, left_q_live, gripper_closed=False)
                    draw_arm_state("right", self.dual_arm_system.right_arm, right_q_live, gripper_closed=False)
                    draw_target_marker(selected_joint_target, [0.1, 0.8, 1.0])
                    selected_label = f"J{selected_joint_idx}" if selected_joint_idx is not None else "None"
                    drag_state = "dragging" if drag_active else "idle"
                    draw_status(
                        f"Joint Drag IK | arm={selected_arm} joint={selected_label} {drag_state} "
                        f"| err={last_drag_error:.4f}m ok={int(last_drag_success)} "
                        f"| LMB pick+drag [Tab] arm [R] reset [Q] quit"
                    )
                else:
                    selected_joint_target = None
                    selected_joint_idx = None
                    if connection_mode_name == "DIRECT":
                        left_q_live = left_traj[frame]
                        right_q_live = right_traj[frame]
                        frame += 1
                        if frame >= frame_count:
                            if loop:
                                frame = 0
                            else:
                                break
                    else:
                        if not paused or step_once:
                            left_q_live = left_traj[frame]
                            right_q_live = right_traj[frame]
                            frame += 1
                            if frame >= frame_count:
                                if loop:
                                    frame = 0
                                else:
                                    frame = frame_count - 1
                                    paused = True
                            step_once = False

                    close_state = int(frame >= frame_count * 0.2 and frame <= frame_count * 0.72)
                    draw_arm_state("left", self.dual_arm_system.left_arm, left_q_live, gripper_closed=bool(close_state))
                    draw_arm_state("right", self.dual_arm_system.right_arm, right_q_live, gripper_closed=bool(close_state))
                    draw_target_marker(None, [0.1, 0.8, 1.0])

                    mode_text = "PAUSED" if paused else "RUN"
                    draw_status(f"Frame {frame + 1}/{frame_count} | {mode_text} | [Space] pause [N] step [R] reset [Q] quit")

                p.stepSimulation()
                if realtime:
                    time.sleep(max(0.0, 1.0 / max(1, fps)))
        finally:
            for item_id in static_ids:
                try:
                    p.removeUserDebugItem(item_id)
                except Exception:
                    pass
            if p.isConnected():
                p.disconnect()

        return connection_mode_name
