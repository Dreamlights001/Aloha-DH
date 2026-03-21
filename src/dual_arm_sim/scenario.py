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

        arm_color = "darkorange"
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

        if connection_mode_name == "GUI":
            p.resetDebugVisualizerCamera(
                cameraDistance=1.7,
                cameraYaw=45,
                cameraPitch=-25,
                cameraTargetPosition=[0.55, 0.0, 0.25],
            )

        frame_count = min(len(left_traj), len(right_traj))
        debug_ids: List[int] = []

        left_q_live = left_traj[0][:]
        right_q_live = right_traj[0][:]
        left_target = self.dual_arm_system.left_arm.end_effector_position(left_q_live)
        right_target = self.dual_arm_system.right_arm.end_effector_position(right_q_live)
        selected_arm = "left"
        drag_active = False

        mouse_button_event = getattr(p, "MOUSE_BUTTON_EVENT", 2)
        mouse_move_event = getattr(p, "MOUSE_MOVE_EVENT", 1)
        mouse_left = getattr(p, "MOUSE_BUTTON_LEFT", 0)
        key_is_down = getattr(p, "KEY_IS_DOWN", 1)

        def clear_debug() -> None:
            nonlocal debug_ids
            for item_id in debug_ids:
                p.removeUserDebugItem(item_id)
            debug_ids = []

        def draw_scene_objects() -> None:
            length = self.conveyor_config["length"]
            width = self.conveyor_config["width"]
            height = self.conveyor_config["height"]
            debug_ids.extend(
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
                    debug_ids.extend(self._pybullet_add_tetra_wireframe(p, obj.position, obj.radius, [1.0, 0.0, 0.0], width=2.2))
                else:
                    debug_ids.extend(self._pybullet_add_cube_wireframe(p, obj.position, obj.radius, [0.0, 0.9, 0.0], width=2.2))

        def draw_arm_state(arm: RobotArm6DOF, q: Sequence[float], gripper_closed: bool) -> None:
            _, positions, transforms = arm.forward_kinematics(q)
            arm_color = [1.0, 0.55, 0.0]
            joint_color = [0.55, 0.55, 0.55]
            for i in range(len(positions) - 1):
                debug_ids.append(p.addUserDebugLine(positions[i], positions[i + 1], arm_color, 4.2, lifeTime=0))
            for pos in positions:
                debug_ids.append(
                    p.addUserDebugLine(
                        [pos[0], pos[1], pos[2] - 0.006],
                        [pos[0], pos[1], pos[2] + 0.006],
                        joint_color,
                        5.0,
                        lifeTime=0,
                    )
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
            grip_col = [1.0, 0.65, 0.1] if gripper_closed else [1.0, 0.8, 0.2]
            debug_ids.append(p.addUserDebugLine(f1_root, f1_tip, grip_col, 4.0, lifeTime=0))
            debug_ids.append(p.addUserDebugLine(f2_root, f2_tip, grip_col, 4.0, lifeTime=0))

        def draw_status(text: str) -> None:
            debug_ids.append(p.addUserDebugText(text, [0.02, -0.52, 1.03], [1, 1, 1], 1.2, lifeTime=0))

        frame = 0
        paused = False
        step_once = False

        try:
            while p.isConnected():
                clear_debug()
                draw_scene_objects()

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
                                drag_active = True
                            if button_state & p.KEY_WAS_RELEASED:
                                drag_active = False
                        if drag_active and event_type in (mouse_move_event, mouse_button_event):
                            try:
                                cam_info = p.getDebugVisualizerCamera()
                                ray_from, ray_to = self._screen_to_ray(mx, my, cam_info)
                                z_ref = left_target[2] if selected_arm == "left" else right_target[2]
                                hit = self._ray_plane_intersection(ray_from, ray_to, z_ref)
                                if hit is not None:
                                    if selected_arm == "left":
                                        left_target = hit
                                    else:
                                        right_target = hit
                            except Exception:
                                pass

                    if selected_arm == "left":
                        ik = self.dual_arm_system.left_arm.inverse_kinematics_position(
                            left_target,
                            initial_angles=left_q_live,
                            max_iters=80,
                            tol=5e-4,
                        )
                        left_q_live = ik.joint_angles
                    else:
                        ik = self.dual_arm_system.right_arm.inverse_kinematics_position(
                            right_target,
                            initial_angles=right_q_live,
                            max_iters=80,
                            tol=5e-4,
                        )
                        right_q_live = ik.joint_angles

                    draw_arm_state(self.dual_arm_system.left_arm, left_q_live, gripper_closed=False)
                    draw_arm_state(self.dual_arm_system.right_arm, right_q_live, gripper_closed=False)
                    debug_ids.extend(self._pybullet_add_target_marker(p, left_target, [0.1, 0.8, 1.0]))
                    debug_ids.extend(self._pybullet_add_target_marker(p, right_target, [1.0, 0.3, 0.3]))
                    draw_status(
                        f"Interactive IK | selected={selected_arm} | drag with left mouse | [Tab] switch [R] reset [Q] quit"
                    )
                else:
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
                    draw_arm_state(self.dual_arm_system.left_arm, left_q_live, gripper_closed=bool(close_state))
                    draw_arm_state(self.dual_arm_system.right_arm, right_q_live, gripper_closed=bool(close_state))

                    mode_text = "PAUSED" if paused else "RUN"
                    draw_status(f"Frame {frame + 1}/{frame_count} | {mode_text} | [Space] pause [N] step [R] reset [Q] quit")

                p.stepSimulation()
                if realtime:
                    time.sleep(max(0.0, 1.0 / max(1, fps)))
        finally:
            if p.isConnected():
                p.disconnect()

        return connection_mode_name
