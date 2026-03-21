"""Sorting scenario simulation and visualization for dual-arm robot."""

from __future__ import annotations

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
            radius = 0.028 if defective else 0.022
            self.products.append(
                SortingObject(position=[x, y, z], defective=defective, radius=radius)
            )
        return self.products

    def _pick_targets(self) -> Tuple[List[float], List[float]]:
        defect = next((obj for obj in self.products if obj.defective), None)
        normal = next((obj for obj in self.products if not obj.defective), None)

        default_z = self.conveyor_config["height"] + 0.03
        if defect is None:
            defect_pos = [0.75, 0.05, default_z]
        else:
            defect_pos = defect.position[:]

        if normal is None:
            normal_pos = [0.35, -0.05, default_z]
        else:
            normal_pos = normal.position[:]

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

    def _draw_arm(
        self,
        ax,
        arm: RobotArm6DOF,
        joint_angles: Sequence[float],
        color: str,
        gripper_closed: bool,
    ) -> None:
        _, positions, transforms = arm.forward_kinematics(joint_angles)

        xs = [p[0] for p in positions]
        ys = [p[1] for p in positions]
        zs = [p[2] for p in positions]

        ax.plot(xs, ys, zs, color=color, lw=2.5)

        for p in positions:
            ax.scatter([p[0]], [p[1]], [p[2]], c=color, s=36)
            ax.plot([p[0], p[0]], [p[1], p[1]], [p[2] - 0.015, p[2] + 0.015], color=color, lw=2)

        ee_tf = transforms[-1]
        self._draw_coordinate_frame(ax, ee_tf)

        ee = [ee_tf[0][3], ee_tf[1][3], ee_tf[2][3]]
        x_axis = [ee_tf[0][0], ee_tf[1][0], ee_tf[2][0]]
        y_axis = [ee_tf[0][1], ee_tf[1][1], ee_tf[2][1]]

        max_width = float(arm.gripper_config.get("max_width", 0.10))
        jaw = max_width * (0.18 if gripper_closed else 1.0)
        finger_len = 0.05

        f1_root = vector_add(ee, vector_scale(y_axis, jaw * 0.5))
        f2_root = vector_add(ee, vector_scale(y_axis, -jaw * 0.5))
        f1_tip = vector_add(f1_root, vector_scale(x_axis, finger_len))
        f2_tip = vector_add(f2_root, vector_scale(x_axis, finger_len))

        gripper_color = "tab:orange" if gripper_closed else "tab:cyan"
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
        poly = Poly3DCollection(vertices, alpha=0.35, facecolor="green", edgecolor="darkgreen")
        ax.add_collection3d(poly)

    def _draw_products(self, ax) -> None:
        for obj in self.products:
            color = "red" if obj.defective else "blue"
            size = max(20, int(obj.radius * 1800))
            ax.scatter(
                [obj.position[0]],
                [obj.position[1]],
                [obj.position[2]],
                c=color,
                s=size,
                depthshade=True,
            )

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
        ax.scatter([normal_bin[0]], [normal_bin[1]], [normal_bin[2]], c="royalblue", s=80, marker="s")
        ax.scatter([defect_bin[0]], [defect_bin[1]], [defect_bin[2]], c="crimson", s=80, marker="s")

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
        """Matplotlib animation mode.

        If save_path is set and show_window is None, the figure is saved without blocking.
        """
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

            left_closed = close_start <= frame <= close_end
            right_closed = close_start <= frame <= close_end

            self._draw_arm(
                ax,
                self.dual_arm_system.left_arm,
                left_traj[frame],
                color="tab:purple",
                gripper_closed=left_closed,
            )
            self._draw_arm(
                ax,
                self.dual_arm_system.right_arm,
                right_traj[frame],
                color="tab:orange",
                gripper_closed=right_closed,
            )

            ax.text2D(0.02, 0.96, f"Frame: {frame + 1}/{frame_count}", transform=ax.transAxes)

        animation = FuncAnimation(
            fig,
            update,
            frames=frame_count,
            interval=interval_ms,
            repeat=True,
        )

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
        corner_min: Sequence[float],
        corner_max: Sequence[float],
        color: Sequence[float],
        width: float = 1.0,
    ) -> List[int]:
        x0, y0, z0 = corner_min
        x1, y1, z1 = corner_max
        pts = [
            [x0, y0, z0], [x1, y0, z0], [x1, y1, z0], [x0, y1, z0],
            [x0, y0, z1], [x1, y0, z1], [x1, y1, z1], [x0, y1, z1],
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

    def visualize_pybullet(
        self,
        left_traj: List[List[float]],
        right_traj: List[List[float]],
        fps: int = 30,
        realtime: bool = True,
        loop: bool = True,
        prefer_gui: bool = True,
        allow_headless_fallback: bool = True,
    ) -> str:
        """Realtime interactive visualization mode via PyBullet GUI.

        Controls:
        - Space: pause/resume
        - N: step one frame (when paused)
        - R: reset frame index
        - Q or ESC: quit
        """
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
        direct_loop = loop if connection_mode_name == "GUI" else False
        debug_ids: List[int] = []

        length = self.conveyor_config["length"]
        width = self.conveyor_config["width"]
        height = self.conveyor_config["height"]
        debug_ids.extend(
            self._pybullet_add_box_wireframe(
                p,
                [0.0, -width * 0.5, 0.0],
                [length, width * 0.5, height],
                [0.2, 0.8, 0.2],
                width=1.5,
            )
        )

        def draw_frame(frame_idx: int, paused: bool) -> None:
            nonlocal debug_ids
            for item_id in debug_ids:
                p.removeUserDebugItem(item_id)
            debug_ids = []

            debug_ids.extend(
                self._pybullet_add_box_wireframe(
                    p,
                    [0.0, -width * 0.5, 0.0],
                    [length, width * 0.5, height],
                    [0.2, 0.8, 0.2],
                    width=1.5,
                )
            )

            for obj in self.products:
                col = [1, 0, 0] if obj.defective else [0, 0.3, 1]
                debug_ids.append(
                    p.addUserDebugLine(
                        [obj.position[0], obj.position[1], obj.position[2] - 0.01],
                        [obj.position[0], obj.position[1], obj.position[2] + 0.01],
                        col,
                        6,
                        lifeTime=0,
                    )
                )

            left_closed = int(frame_idx >= frame_count * 0.2 and frame_idx <= frame_count * 0.72)
            right_closed = left_closed

            for arm, q, color, closed in (
                (self.dual_arm_system.left_arm, left_traj[frame_idx], [0.6, 0.1, 0.9], left_closed),
                (self.dual_arm_system.right_arm, right_traj[frame_idx], [1.0, 0.5, 0.1], right_closed),
            ):
                _, positions, transforms = arm.forward_kinematics(q)
                for i in range(len(positions) - 1):
                    debug_ids.append(
                        p.addUserDebugLine(positions[i], positions[i + 1], color, 3.0, lifeTime=0)
                    )

                ee_tf = transforms[-1]
                ee = [ee_tf[0][3], ee_tf[1][3], ee_tf[2][3]]
                x_axis = [ee_tf[0][0], ee_tf[1][0], ee_tf[2][0]]
                y_axis = [ee_tf[0][1], ee_tf[1][1], ee_tf[2][1]]
                jaw = float(arm.gripper_config.get("max_width", 0.10)) * (0.2 if closed else 1.0)

                f1_root = vector_add(ee, vector_scale(y_axis, jaw * 0.5))
                f2_root = vector_add(ee, vector_scale(y_axis, -jaw * 0.5))
                f1_tip = vector_add(f1_root, vector_scale(x_axis, 0.05))
                f2_tip = vector_add(f2_root, vector_scale(x_axis, 0.05))
                grip_col = [1.0, 0.6, 0.0] if closed else [0.2, 1.0, 1.0]
                debug_ids.append(p.addUserDebugLine(f1_root, f1_tip, grip_col, 4.0, lifeTime=0))
                debug_ids.append(p.addUserDebugLine(f2_root, f2_tip, grip_col, 4.0, lifeTime=0))

            state = "PAUSED" if paused else "RUN"
            info = f"Frame {frame_idx + 1}/{frame_count} | {state} | [Space] pause [N] step [R] reset [Q] quit"
            debug_ids.append(
                p.addUserDebugText(info, [0.02, -0.5, 1.05], textColorRGB=[1, 1, 1], textSize=1.2, lifeTime=0)
            )

        frame = 0
        paused = False
        step_once = False
        escape_key_codes = [27]  # ASCII ESC fallback for older/newer pybullet variants
        for attr_name in ("B3G_ESCAPE", "B3G_ESC"):
            code = getattr(p, attr_name, None)
            if isinstance(code, int):
                escape_key_codes.append(code)

        try:
            while p.isConnected():
                if connection_mode_name == "DIRECT":
                    draw_frame(frame, paused=False)
                    frame += 1
                    if frame >= frame_count:
                        if direct_loop:
                            frame = 0
                        else:
                            break
                    p.stepSimulation()
                    if realtime:
                        time.sleep(max(0.0, 1.0 / max(1, fps)))
                    continue

                keys = p.getKeyboardEvents()
                if ord(" ") in keys and keys[ord(" ")] & p.KEY_WAS_TRIGGERED:
                    paused = not paused
                if ord("n") in keys and keys[ord("n")] & p.KEY_WAS_TRIGGERED:
                    step_once = True
                if ord("r") in keys and keys[ord("r")] & p.KEY_WAS_TRIGGERED:
                    frame = 0
                escape_triggered = any(
                    (code in keys and keys[code] & p.KEY_WAS_TRIGGERED)
                    for code in escape_key_codes
                )
                if (
                    (ord("q") in keys and keys[ord("q")] & p.KEY_WAS_TRIGGERED)
                    or escape_triggered
                ):
                    break

                if not paused or step_once:
                    draw_frame(frame, paused)
                    frame += 1
                    if frame >= frame_count:
                        if loop:
                            frame = 0
                        else:
                            frame = frame_count - 1
                            paused = True
                    step_once = False

                p.stepSimulation()
                if realtime:
                    time.sleep(max(0.0, 1.0 / max(1, fps)))
        finally:
            if p.isConnected():
                p.disconnect()
        return connection_mode_name
