"""Run dual-arm MDH simulation demo for defect sorting.

Cross-platform usage (no PYTHONPATH needed):
  python examples/run_sorting_demo.py --viz matplotlib
  python examples/run_sorting_demo.py --viz pybullet
  python examples/run_sorting_demo.py --viz both --save demo.gif
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Allow running this script directly on Windows/macOS/Linux/WSL without PYTHONPATH.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from dual_arm_sim import (  # noqa: E402
    ARM_OFFSET,
    CAMERA_CONFIG,
    CONVEYOR_CONFIG,
    DEFAULT_ARM_MODEL,
    GRIPPER_CONFIG,
    JOINT_LIMITS_EXAMPLE,
    MDH_PARAMS_EXAMPLE,
    MODEL_SPEC,
    OFFICIAL_SOURCES,
    DualArmSystem,
    SortingScenario,
    configure_matplotlib_runtime,
    get_runtime_info,
    resolve_animation_output_path,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Dual-arm sorting robot MDH simulation")
    parser.add_argument(
        "--viz",
        choices=("matplotlib", "pybullet", "both"),
        default="both",
        help="Visualization mode. 'both' runs PyBullet then Matplotlib export.",
    )
    parser.add_argument(
        "--save",
        type=str,
        default=None,
        help=(
            "Animation output path for Matplotlib mode. "
            "Relative filenames are saved under ./output (e.g., demo.gif -> ./output/demo.gif)."
        ),
    )
    parser.add_argument("--products", type=int, default=10, help="Number of products on conveyor")
    parser.add_argument(
        "--defect-ratio",
        type=float,
        default=0.25,
        help="Defective product ratio in generated scene",
    )
    parser.add_argument(
        "--strict-ik",
        action="store_true",
        help="Fail early if any waypoint position error exceeds solver threshold",
    )
    parser.add_argument(
        "--ik-max-error",
        type=float,
        default=0.03,
        help="Waypoint success threshold in meters for report output",
    )
    parser.add_argument("--fps", type=int, default=30, help="Playback FPS in PyBullet mode")
    parser.add_argument(
        "--no-loop",
        action="store_true",
        help="Disable looping in PyBullet playback",
    )
    parser.add_argument(
        "--no-realtime",
        action="store_true",
        help="Disable realtime sleeping in PyBullet (faster stepping)",
    )
    parser.add_argument(
        "--pybullet-direct",
        action="store_true",
        help="Force PyBullet DIRECT (headless) mode instead of GUI",
    )
    parser.add_argument(
        "--force-gui",
        action="store_true",
        help="Force PyBullet GUI mode (override non-interactive safety checks)",
    )
    parser.add_argument(
        "--no-headless-fallback",
        action="store_true",
        help="Disable automatic fallback from PyBullet GUI to DIRECT mode",
    )
    return parser


def _print_official_spec_summary() -> None:
    sections = MODEL_SPEC["arm_sections_mm"]
    joints = MODEL_SPEC["joint_limits_deg"]
    print(f"Arm model: {MODEL_SPEC['display_name']} ({DEFAULT_ARM_MODEL})")
    print("Official sources:")
    print("  -", OFFICIAL_SOURCES["aloha_vx300s_6dof"])
    print("  -", OFFICIAL_SOURCES["xseries_linkage"])
    print(
        "Sections (mm): "
        f"upper={sections['upper_arm']}, forearm={sections['forearm']}, "
        f"wrist={sections['wrist_tilt_to_rotate']}, rail={sections['gripper_to_rail']}, tip={sections['finger_tip']}"
    )
    print("Joint limits (deg):", joints)


def _print_runtime_info() -> None:
    runtime = get_runtime_info()
    print(
        "Runtime:",
        f"os={runtime.os_name}",
        f"wsl={runtime.is_wsl}",
        f"gui_available={runtime.has_gui}",
    )


def _print_waypoint_report(report: dict, ik_max_error: float) -> None:
    for arm_name in ("left", "right"):
        errors = [wp.error_norm for wp in report[arm_name]]
        max_error = max(errors) if errors else 0.0
        avg_error = sum(errors) / len(errors) if errors else 0.0
        failed = sum(1 for wp in report[arm_name] if wp.error_norm > ik_max_error)
        print(
            f"{arm_name.capitalize()} arm waypoint error: "
            f"max={max_error:.4f}m avg={avg_error:.4f}m failed={failed}/{len(errors)} "
            f"(threshold={ik_max_error:.3f}m)"
        )


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    mpl_dir = configure_matplotlib_runtime()
    runtime = get_runtime_info()

    dual = DualArmSystem(
        mdh_params=MDH_PARAMS_EXAMPLE,
        joint_limits=JOINT_LIMITS_EXAMPLE,
        arm_offset=ARM_OFFSET,
        gripper_config=GRIPPER_CONFIG,
    )

    scene = SortingScenario(
        dual_arm_system=dual,
        conveyor_config=CONVEYOR_CONFIG,
        camera_config=CAMERA_CONFIG,
    )
    scene.generate_products(count=args.products, defective_ratio=args.defect_ratio)
    left_traj, right_traj, report = scene.plan_sorting_trajectories(
        strict_ik=args.strict_ik,
        return_report=True,
    )

    state = dual.forward_both([0.0] * 6, [0.0] * 6)
    left_ee = state["left_end_transform"]
    right_ee = state["right_end_transform"]

    _print_official_spec_summary()
    _print_runtime_info()
    print("Matplotlib cache dir:", mpl_dir)
    print("Left EE @ home:", [round(left_ee[i][3], 4) for i in range(3)])
    print("Right EE @ home:", [round(right_ee[i][3], 4) for i in range(3)])
    print("Planned trajectory frames:", len(left_traj))
    _print_waypoint_report(report, ik_max_error=args.ik_max_error)

    save_path: Path | None = None
    if args.viz in ("matplotlib", "both"):
        save_path = resolve_animation_output_path(
            save_arg=args.save,
            repo_root=PROJECT_ROOT,
            default_filename="sorting_demo.gif",
            output_dir_name="output",
        )

    if args.viz in ("pybullet", "both"):
        interactive_tty = sys.stdin.isatty() and sys.stdout.isatty()
        prefer_gui = (not args.pybullet_direct) and runtime.has_gui and (interactive_tty or args.force_gui)
        mode = scene.visualize_pybullet(
            left_traj=left_traj,
            right_traj=right_traj,
            fps=args.fps,
            realtime=not args.no_realtime,
            loop=not args.no_loop,
            prefer_gui=prefer_gui,
            allow_headless_fallback=not args.no_headless_fallback,
        )
        if mode == "DIRECT":
            print("PyBullet mode: DIRECT (headless). GUI unavailable or disabled.")
        else:
            print("PyBullet mode: GUI")

    if args.viz in ("matplotlib", "both"):
        scene.animate_sorting(
            save_path=str(save_path) if save_path else None,
            interval_ms=max(1, int(1000 / max(1, args.fps))),
            left_traj=left_traj,
            right_traj=right_traj,
            show_window=False,
        )
        if save_path is not None:
            print("Saved animation:", save_path)


if __name__ == "__main__":
    main()
