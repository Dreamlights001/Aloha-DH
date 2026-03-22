"""Sorting scenario simulation and visualization for dual-arm robot."""

from __future__ import annotations

import math
import random
import time
from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

from .config import SORTING_BINS
from .input_controls import GamepadInputManager
from .math3d import vector_add, vector_scale
from .robot import DualArmSystem, RobotArm6DOF


@dataclass
class SortingObject:
    position: List[float]
    defective: bool
    radius: float


class SortingScenario:
    PRODUCT_SCALE = 0.8

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
            base_radius = 0.028 if defective else 0.024
            radius = base_radius * self.PRODUCT_SCALE
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
        ax.add_collection3d(Poly3DCollection(faces, alpha=0.95, facecolor=color, edgecolor="black", linewidths=0.5))

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
        ax.add_collection3d(Poly3DCollection(faces, alpha=0.95, facecolor=color, edgecolor="black", linewidths=0.5))

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
        ax.plot(xs, ys, zs, color=arm_color, lw=5.8)

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
        ax.plot([f1_root[0], f1_tip[0]], [f1_root[1], f1_tip[1]], [f1_root[2], f1_tip[2]], c=gripper_color, lw=5.2)
        ax.plot([f2_root[0], f2_tip[0]], [f2_root[1], f2_tip[1]], [f2_root[2], f2_tip[2]], c=gripper_color, lw=5.2)

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
        ax.set_xticks([round(-0.25 + i * 0.25, 2) for i in range(7)])
        ax.set_yticks([round(-0.5 + i * 0.25, 2) for i in range(5)])
        ax.set_zticks([round(i * 0.25, 2) for i in range(6)])
        ax.grid(True, alpha=0.35)

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

    @staticmethod
    def _ray_plane_intersection_with_normal(
        ray_from: Sequence[float],
        ray_to: Sequence[float],
        plane_point: Sequence[float],
        plane_normal: Sequence[float],
    ) -> List[float] | None:
        direction = [ray_to[i] - ray_from[i] for i in range(3)]
        denom = sum(direction[i] * plane_normal[i] for i in range(3))
        if abs(denom) < 1e-9:
            return None
        t = sum((plane_point[i] - ray_from[i]) * plane_normal[i] for i in range(3)) / denom
        if t < 0:
            return None
        return [ray_from[i] + t * direction[i] for i in range(3)]

    @staticmethod
    def _pybullet_create_solid_cube(p, center: Sequence[float], half: float, rgba: Sequence[float]) -> int:
        vis = p.createVisualShape(
            shapeType=p.GEOM_BOX,
            halfExtents=[half, half, half],
            rgbaColor=list(rgba),
            specularColor=[0.25, 0.25, 0.25],
        )
        return p.createMultiBody(
            baseMass=0.0,
            baseCollisionShapeIndex=-1,
            baseVisualShapeIndex=vis,
            basePosition=list(center),
        )

    @staticmethod
    def _pybullet_create_solid_tetra(p, center: Sequence[float], scale: float, rgba: Sequence[float]) -> int:
        # Unit regular-tetra-like mesh centered around origin, then scaled.
        vertices = [
            [0.0, 0.0, 1.0],
            [1.0, -1.0, -1.0],
            [-1.0, -1.0, -1.0],
            [0.0, 1.0, -1.0],
        ]
        indices = [
            0, 1, 2,
            0, 2, 3,
            0, 3, 1,
            1, 3, 2,
        ]
        vis = p.createVisualShape(
            shapeType=p.GEOM_MESH,
            vertices=vertices,
            indices=indices,
            meshScale=[scale, scale, scale],
            rgbaColor=list(rgba),
            specularColor=[0.25, 0.25, 0.25],
        )
        return p.createMultiBody(
            baseMass=0.0,
            baseCollisionShapeIndex=-1,
            baseVisualShapeIndex=vis,
            basePosition=list(center),
        )

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
        input_device: str = "auto",
        gamepad_enabled: bool = True,
        ui_style: str = "industrial",
        default_drag_mode: bool = True,
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

        camera_distance = 1.7
        camera_yaw = 45.0
        camera_pitch = -25.0
        camera_target = [0.55, 0.0, 0.25]
        if connection_mode_name == "GUI":
            p.resetDebugVisualizerCamera(
                cameraDistance=camera_distance,
                cameraYaw=camera_yaw,
                cameraPitch=camera_pitch,
                cameraTargetPosition=camera_target,
            )

        frame_count = min(len(left_traj), len(right_traj))
        if frame_count == 0:
            raise ValueError("left_traj/right_traj must contain at least one frame")
        input_device = str(input_device).strip().lower()
        if input_device not in {"auto", "mouse", "gamepad"}:
            input_device = "auto"
        ui_style = str(ui_style).strip().lower()
        if ui_style not in {"industrial", "minimal"}:
            ui_style = "industrial"

        left_q_live = left_traj[0][:]
        right_q_live = right_traj[0][:]
        selected_arm = "left"
        selected_joint_idx: int | None = None
        selected_joint_target: List[float] | None = None
        drag_active = False
        drag_mode_enabled = bool(default_drag_mode)
        drag_plane_point = [0.0, 0.0, self.conveyor_config["height"] + 0.2]
        drag_plane_normal = [0.0, 0.0, 1.0]
        last_drag_error = 0.0
        last_drag_success = False
        input_source = "mouse"
        gamepad_manager = GamepadInputManager(enabled=gamepad_enabled and connection_mode_name == "GUI")
        gamepad_name = ""

        mouse_button_event = getattr(p, "MOUSE_BUTTON_EVENT", 2)
        mouse_move_event = getattr(p, "MOUSE_MOVE_EVENT", 1)
        mouse_left = getattr(p, "MOUSE_BUTTON_LEFT", 0)
        mouse_right = getattr(p, "MOUSE_BUTTON_RIGHT", 2)
        key_is_down = getattr(p, "KEY_IS_DOWN", 1)
        camera_drag_active = False
        camera_last_xy: Tuple[int, int] | None = None

        static_debug_ids: List[int] = []
        static_body_ids: List[int] = []
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

        def apply_camera() -> None:
            if connection_mode_name != "GUI":
                return
            p.resetDebugVisualizerCamera(
                cameraDistance=max(0.4, camera_distance),
                cameraYaw=camera_yaw,
                cameraPitch=max(-89.0, min(89.0, camera_pitch)),
                cameraTargetPosition=camera_target,
            )

        def draw_scene_objects_once() -> None:
            length = self.conveyor_config["length"]
            width = self.conveyor_config["width"]
            height = self.conveyor_config["height"]

            conveyor_vis = p.createVisualShape(
                shapeType=p.GEOM_BOX,
                halfExtents=[0.5 * length, 0.5 * width, 0.5 * height],
                rgbaColor=[0.2, 0.8, 0.2, 0.85],
                specularColor=[0.2, 0.2, 0.2],
            )
            static_body_ids.append(
                p.createMultiBody(
                    baseMass=0.0,
                    baseCollisionShapeIndex=-1,
                    baseVisualShapeIndex=conveyor_vis,
                    basePosition=[length * 0.5, 0.0, height * 0.5],
                )
            )
            static_debug_ids.extend(
                self._pybullet_add_box_wireframe(
                    p,
                    center=[length * 0.5, 0.0, height * 0.5],
                    half_extents=[0.5 * length, 0.5 * width, 0.5 * height],
                    color=[0.1, 0.45, 0.1],
                    width=1.0,
                )
            )
            for obj in self.products:
                if obj.defective:
                    try:
                        static_body_ids.append(
                            self._pybullet_create_solid_tetra(
                                p,
                                center=obj.position,
                                scale=obj.radius,
                                rgba=[1.0, 0.0, 0.0, 0.95],
                            )
                        )
                    except Exception:
                        static_debug_ids.extend(
                            self._pybullet_add_tetra_wireframe(
                                p, obj.position, obj.radius, [1.0, 0.0, 0.0], width=2.8
                            )
                        )
                else:
                    static_body_ids.append(
                        self._pybullet_create_solid_cube(
                            p,
                            center=obj.position,
                            half=obj.radius,
                            rgba=[0.0, 0.85, 0.0, 0.95],
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
                    6.0,
                )
            for i, pos in enumerate(positions):
                marker_half = 0.008
                marker_color = joint_color
                marker_width = 7.0
                if arm_key == selected_arm and selected_joint_idx == i:
                    marker_half = 0.016
                    marker_color = [0.1, 0.9, 1.0]
                    marker_width = 8.0
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
            upsert_line(f"{arm_key}_grip_0", f1_root, f1_tip, grip_col, 5.5)
            upsert_line(f"{arm_key}_grip_1", f2_root, f2_tip, grip_col, 5.5)
            return positions

        def draw_status(text: str) -> None:
            upsert_text("status", text, [0.02, -0.52, 1.03], [1, 1, 1], 1.2)

        def draw_target_marker(center: Sequence[float] | None, color: Sequence[float]) -> None:
            if center is None:
                center = [0.0, 0.0, -10.0]
            d = 0.03
            upsert_line("target_x", [center[0] - d, center[1], center[2]], [center[0] + d, center[1], center[2]], color, 4.0)
            upsert_line("target_y", [center[0], center[1] - d, center[2]], [center[0], center[1] + d, center[2]], color, 4.0)
            upsert_line("target_z", [center[0], center[1], center[2] - d], [center[0], center[1], center[2] + d], color, 4.0)

        def draw_joint_panel(left_positions: Sequence[Sequence[float]], right_positions: Sequence[Sequence[float]]) -> None:
            if ui_style == "minimal":
                return
            yaw_rad = math.radians(camera_yaw)
            right_axis = [-math.sin(yaw_rad), math.cos(yaw_rad), 0.0]
            anchor = [
                camera_target[0] + right_axis[0] * 0.95,
                camera_target[1] + right_axis[1] * 0.95,
                camera_target[2] + 0.55,
            ]
            dz = 0.065
            upsert_text("hud_title", "Joint Positions ^0p_i (m)", anchor, [0.95, 0.95, 0.6], 1.1)
            upsert_text(
                "hud_meta",
                f"Input={input_source} | Mode={'DRAG' if drag_mode_enabled else 'NORMAL'}"
                + (f" | Pad={gamepad_name}" if gamepad_name else ""),
                [anchor[0], anchor[1], anchor[2] - dz * 0.6],
                [0.8, 0.9, 1.0],
                0.95,
            )
            left_base = left_positions[0]
            right_base = right_positions[0]
            for i in range(1, 7):
                lp = left_positions[i]
                rp = right_positions[i]
                lp_rel = [lp[0] - left_base[0], lp[1] - left_base[1], lp[2] - left_base[2]]
                rp_rel = [rp[0] - right_base[0], rp[1] - right_base[1], rp[2] - right_base[2]]
                upsert_text(
                    f"hud_left_{i}",
                    f"L-J{i}: ({lp_rel[0]:+.3f}, {lp_rel[1]:+.3f}, {lp_rel[2]:+.3f})",
                    [anchor[0], anchor[1], anchor[2] - dz * i],
                    [1.0, 0.65, 0.1],
                    1.0,
                )
                upsert_text(
                    f"hud_right_{i}",
                    f"R-J{i}: ({rp_rel[0]:+.3f}, {rp_rel[1]:+.3f}, {rp_rel[2]:+.3f})",
                    [anchor[0], anchor[1], anchor[2] - dz * (i + 6)],
                    [1.0, 1.0, 0.25],
                    1.0,
                )

        def get_mode_button_rect(cam_info) -> Tuple[int, int, int, int]:
            width = max(1, int(cam_info[0]))
            x0 = max(10, width - 320)
            y0 = 18
            x1 = max(x0 + 120, width - 12)
            y1 = 60
            return x0, y0, x1, y1

        def draw_mode_toggle_button() -> None:
            if ui_style == "minimal":
                return
            yaw_rad = math.radians(camera_yaw)
            right_axis = [-math.sin(yaw_rad), math.cos(yaw_rad), 0.0]
            anchor = [
                camera_target[0] + right_axis[0] * 1.02,
                camera_target[1] + right_axis[1] * 1.02,
                camera_target[2] + 0.82,
            ]
            mode_text = "DRAG" if drag_mode_enabled else "NORMAL"
            mode_col = [0.1, 1.0, 0.4] if drag_mode_enabled else [0.95, 0.95, 0.95]
            upsert_text("hud_mode_btn", f"[M] Mode: {mode_text}", anchor, mode_col, 1.25)

        def pick_joint_index(
            arm: RobotArm6DOF,
            q: Sequence[float],
            ray_from: Sequence[float],
            ray_to: Sequence[float],
            threshold: float = 0.06,
        ) -> Tuple[int | None, List[float] | None, float]:
            _, positions, _ = arm.forward_kinematics(q)
            direction = [ray_to[i] - ray_from[i] for i in range(3)]
            ray_len = math.sqrt(sum(v * v for v in direction))
            if ray_len < 1e-9:
                return None, None, float("inf")
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
                return None, None, float("inf")
            return best_idx, best_pos, best_dist

        draw_scene_objects_once()

        frame = 0
        paused = False
        step_once = False

        try:
            while p.isConnected():
                interactive_drag = (
                    enable_drag_target
                    and connection_mode_name == "GUI"
                    and drag_mode_enabled
                )
                gp_drag_dx = 0.0
                gp_drag_dy = 0.0
                gp_drag_dz = 0.0
                keys = p.getKeyboardEvents() if connection_mode_name == "GUI" else {}
                gp = gamepad_manager.poll() if connection_mode_name == "GUI" else None
                use_gamepad = bool(
                    gp is not None
                    and gp.connected
                    and gamepad_enabled
                    and input_device in {"auto", "gamepad"}
                )
                if use_gamepad:
                    input_source = "gamepad"
                    gamepad_name = gp.device_name
                elif input_device == "gamepad" and gamepad_enabled and connection_mode_name == "GUI":
                    input_source = "mouse(fallback)"
                    gamepad_name = ""
                else:
                    input_source = "mouse"
                    gamepad_name = ""

                if connection_mode_name == "GUI":
                    if ord("m") in keys and keys[ord("m")] & p.KEY_WAS_TRIGGERED:
                        drag_mode_enabled = not drag_mode_enabled
                        drag_active = False
                    if ord("M") in keys and keys[ord("M")] & p.KEY_WAS_TRIGGERED:
                        drag_mode_enabled = not drag_mode_enabled
                        drag_active = False
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

                    yaw_delta = 0.0
                    pitch_delta = 0.0
                    dist_delta = 0.0
                    for code in (ord("a"), ord("A"), getattr(p, "B3G_LEFT_ARROW", -1)):
                        if code in keys and keys[code] & key_is_down:
                            yaw_delta -= 1.0
                    for code in (ord("d"), ord("D"), getattr(p, "B3G_RIGHT_ARROW", -1)):
                        if code in keys and keys[code] & key_is_down:
                            yaw_delta += 1.0
                    for code in (ord("w"), ord("W"), getattr(p, "B3G_UP_ARROW", -1)):
                        if code in keys and keys[code] & key_is_down:
                            pitch_delta += 0.8
                    for code in (ord("s"), ord("S"), getattr(p, "B3G_DOWN_ARROW", -1)):
                        if code in keys and keys[code] & key_is_down:
                            pitch_delta -= 0.8
                    for code in (ord("z"), ord("Z")):
                        if code in keys and keys[code] & key_is_down:
                            dist_delta += 0.02
                    for code in (ord("x"), ord("X")):
                        if code in keys and keys[code] & key_is_down:
                            dist_delta -= 0.02
                    if abs(yaw_delta) > 1e-9 or abs(pitch_delta) > 1e-9 or abs(dist_delta) > 1e-9:
                        camera_yaw += yaw_delta
                        camera_pitch += pitch_delta
                        camera_distance = max(0.4, camera_distance + dist_delta)
                        apply_camera()

                    if use_gamepad and gp is not None:
                        if "start" in gp.buttons_triggered:
                            paused = not paused
                        if "lb" in gp.buttons_triggered:
                            selected_arm = "left"
                        if "rb" in gp.buttons_triggered:
                            selected_arm = "right"
                        if "x" in gp.buttons_triggered:
                            drag_mode_enabled = not drag_mode_enabled
                            drag_active = False
                        if "a" in gp.buttons_triggered:
                            drag_active = True
                        if "b" in gp.buttons_triggered:
                            drag_active = False
                            selected_joint_target = None

                        dpad_x, dpad_y = gp.dpad
                        if dpad_y != 0:
                            if selected_joint_idx is None:
                                selected_joint_idx = 1 if dpad_y > 0 else 6
                            else:
                                selected_joint_idx = max(1, min(6, selected_joint_idx + dpad_y))

                        lsx, lsy = gp.left_stick
                        rsx, rsy = gp.right_stick
                        camera_yaw += rsx * 2.2
                        camera_pitch += -rsy * 1.8
                        camera_distance = max(0.4, camera_distance + (gp.left_trigger - gp.right_trigger) * 0.03)
                        yaw_rad = math.radians(camera_yaw)
                        # Camera-relative pan basis:
                        # right stick X/left stick X should move to screen-right,
                        # and stick forward should move to screen-forward.
                        right_axis = [math.cos(yaw_rad), math.sin(yaw_rad), 0.0]
                        forward_axis = [-math.sin(yaw_rad), math.cos(yaw_rad), 0.0]
                        if drag_active and drag_mode_enabled:
                            gp_drag_dx = lsx
                            gp_drag_dy = lsy
                            gp_drag_dz = float(dpad_x)
                        else:
                            camera_target[0] += (right_axis[0] * lsx - forward_axis[0] * lsy) * 0.016
                            camera_target[1] += (right_axis[1] * lsx - forward_axis[1] * lsy) * 0.016
                        apply_camera()

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

                if connection_mode_name == "GUI":
                    mouse_events = p.getMouseEvents()
                    for ev in mouse_events:
                        if len(ev) < 5:
                            continue
                        event_type, mx, my, button_idx, button_state = ev[0], ev[1], ev[2], ev[3], ev[4]

                        if event_type == mouse_button_event and button_idx == mouse_right:
                            if (button_state & p.KEY_WAS_TRIGGERED) or (button_state & key_is_down):
                                camera_drag_active = True
                                camera_last_xy = (mx, my)
                            if button_state & p.KEY_WAS_RELEASED:
                                camera_drag_active = False
                                camera_last_xy = None

                        if camera_drag_active and event_type == mouse_move_event and camera_last_xy is not None:
                            dx = mx - camera_last_xy[0]
                            dy = my - camera_last_xy[1]
                            camera_last_xy = (mx, my)
                            camera_yaw += 0.18 * dx
                            camera_pitch -= 0.18 * dy
                            apply_camera()

                        if (
                            event_type == mouse_button_event
                            and button_idx == mouse_left
                            and (button_state & p.KEY_WAS_TRIGGERED)
                        ):
                            try:
                                cam_info = p.getDebugVisualizerCamera()
                                x0, y0, x1, y1 = get_mode_button_rect(cam_info)
                                if x0 <= mx <= x1 and y0 <= my <= y1:
                                    drag_mode_enabled = not drag_mode_enabled
                                    drag_active = False
                                    continue
                            except Exception:
                                pass

                        if not interactive_drag:
                            continue

                        if event_type == mouse_button_event and button_idx == mouse_left:
                            if (button_state & p.KEY_WAS_TRIGGERED) or (button_state & key_is_down):
                                try:
                                    cam_info = p.getDebugVisualizerCamera()
                                    ray_from, ray_to = self._screen_to_ray(mx, my, cam_info)
                                    lj, lp, ld = pick_joint_index(
                                        self.dual_arm_system.left_arm,
                                        left_q_live,
                                        ray_from,
                                        ray_to,
                                    )
                                    rj, rp, rd = pick_joint_index(
                                        self.dual_arm_system.right_arm,
                                        right_q_live,
                                        ray_from,
                                        ray_to,
                                    )
                                    if ld <= rd and lj is not None and lp is not None:
                                        selected_arm = "left"
                                        selected_joint_idx = lj
                                        selected_joint_target = lp[:]
                                    elif rj is not None and rp is not None:
                                        selected_arm = "right"
                                        selected_joint_idx = rj
                                        selected_joint_target = rp[:]

                                    if selected_joint_idx is not None and selected_joint_target is not None:
                                        drag_active = True
                                        drag_plane_point = selected_joint_target[:]
                                        ray_dir = [ray_to[i] - ray_from[i] for i in range(3)]
                                        ray_norm = math.sqrt(sum(v * v for v in ray_dir))
                                        if ray_norm > 1e-9:
                                            drag_plane_normal = [v / ray_norm for v in ray_dir]
                                        else:
                                            drag_plane_normal = [0.0, 0.0, 1.0]
                                except Exception:
                                    pass
                            if button_state & p.KEY_WAS_RELEASED:
                                drag_active = False

                        if (
                            interactive_drag
                            and drag_active
                            and selected_joint_idx is not None
                            and event_type in (mouse_move_event, mouse_button_event)
                        ):
                            try:
                                cam_info = p.getDebugVisualizerCamera()
                                ray_from, ray_to = self._screen_to_ray(mx, my, cam_info)
                                hit = self._ray_plane_intersection_with_normal(
                                    ray_from, ray_to, drag_plane_point, drag_plane_normal
                                )
                                if hit is None:
                                    hit = self._ray_plane_intersection(ray_from, ray_to, drag_plane_point[2])
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

                    if use_gamepad and interactive_drag and drag_active:
                        try:
                            if selected_joint_idx is None:
                                selected_joint_idx = 6
                            selected_joint_idx = max(1, min(6, selected_joint_idx))
                            arm_obj = (
                                self.dual_arm_system.left_arm
                                if selected_arm == "left"
                                else self.dual_arm_system.right_arm
                            )
                            q_live = left_q_live if selected_arm == "left" else right_q_live

                            if selected_joint_target is None:
                                selected_joint_target = arm_obj.point_position(q_live, selected_joint_idx)

                            yaw_rad = math.radians(camera_yaw)
                            right_axis = [math.cos(yaw_rad), math.sin(yaw_rad), 0.0]
                            forward_axis = [-math.sin(yaw_rad), math.cos(yaw_rad), 0.0]
                            selected_joint_target = [
                                selected_joint_target[0] + (right_axis[0] * gp_drag_dx - forward_axis[0] * gp_drag_dy) * 0.012,
                                selected_joint_target[1] + (right_axis[1] * gp_drag_dx - forward_axis[1] * gp_drag_dy) * 0.012,
                                selected_joint_target[2] + gp_drag_dz * 0.008,
                            ]
                            active_dofs = list(range(min(6, max(1, selected_joint_idx))))
                            ik = arm_obj.inverse_kinematics_point_position(
                                selected_joint_target,
                                point_index=selected_joint_idx,
                                initial_angles=q_live,
                                active_dofs=active_dofs,
                                max_iters=60,
                                tol=1e-3,
                                damping=5e-3,
                                step_size=0.75,
                            )
                            if selected_arm == "left":
                                left_q_live = ik.joint_angles
                            else:
                                right_q_live = ik.joint_angles
                            last_drag_error = ik.error_norm
                            last_drag_success = ik.success
                        except Exception:
                            pass

                if interactive_drag:
                    left_positions = draw_arm_state("left", self.dual_arm_system.left_arm, left_q_live, gripper_closed=False)
                    right_positions = draw_arm_state("right", self.dual_arm_system.right_arm, right_q_live, gripper_closed=False)
                    draw_target_marker(selected_joint_target, [0.1, 0.8, 1.0])
                    draw_joint_panel(left_positions, right_positions)
                    draw_mode_toggle_button()
                    selected_label = f"J{selected_joint_idx}" if selected_joint_idx is not None else "None"
                    drag_state = "dragging" if drag_active else "idle"
                    draw_status(
                        f"Joint Drag IK | arm={selected_arm} joint={selected_label} {drag_state} "
                        f"| err={last_drag_error:.4f}m ok={int(last_drag_success)} "
                        f"| mode={'DRAG' if drag_mode_enabled else 'NORMAL'} src={input_source} "
                        f"| LMB drag joint | RMB rotate | wheel zoom | WASD/ZX camera | [M] mode [R] reset [Q] quit"
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
                    left_positions = draw_arm_state("left", self.dual_arm_system.left_arm, left_q_live, gripper_closed=bool(close_state))
                    right_positions = draw_arm_state("right", self.dual_arm_system.right_arm, right_q_live, gripper_closed=bool(close_state))
                    draw_target_marker(None, [0.1, 0.8, 1.0])
                    draw_joint_panel(left_positions, right_positions)
                    draw_mode_toggle_button()

                    mode_text = "PAUSED" if paused else "RUN"
                    draw_status(
                        f"Frame {frame + 1}/{frame_count} | {mode_text} | "
                        f"mode={'DRAG' if drag_mode_enabled else 'NORMAL'} src={input_source} | "
                        f"RMB rotate | wheel zoom | WASD/ZX camera | [M] mode [Space] pause [N] step [R] reset [Q] quit"
                    )

                p.stepSimulation()
                if realtime:
                    time.sleep(max(0.0, 1.0 / max(1, fps)))
        finally:
            for item_id in static_debug_ids:
                try:
                    p.removeUserDebugItem(item_id)
                except Exception:
                    pass
            for body_id in static_body_ids:
                try:
                    p.removeBody(body_id)
                except Exception:
                    pass
            if p.isConnected():
                p.disconnect()

        return connection_mode_name
