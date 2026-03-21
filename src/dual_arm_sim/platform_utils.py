"""Cross-platform runtime helpers for Windows/macOS/WSL/Linux."""

from __future__ import annotations

import os
import platform
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping


@dataclass(frozen=True)
class RuntimeInfo:
    os_name: str
    is_windows: bool
    is_macos: bool
    is_linux: bool
    is_wsl: bool
    has_gui: bool


def _normalize_platform_name(name: str) -> str:
    lower = name.lower()
    if lower.startswith("win"):
        return "windows"
    if lower == "darwin":
        return "macos"
    if lower.startswith("linux"):
        return "linux"
    return lower


def detect_wsl(env: Mapping[str, str] | None = None, uname_release: str | None = None) -> bool:
    env_map = os.environ if env is None else env
    if env_map.get("WSL_DISTRO_NAME") or env_map.get("WSL_INTEROP"):
        return True

    release = uname_release if uname_release is not None else platform.uname().release
    release_lower = release.lower()
    return "microsoft" in release_lower or "wsl" in release_lower


def detect_gui_available(
    platform_name: str | None = None,
    env: Mapping[str, str] | None = None,
    is_wsl: bool | None = None,
) -> bool:
    env_map = os.environ if env is None else env
    pname = _normalize_platform_name(platform_name or os.sys.platform)
    wsl = detect_wsl(env_map) if is_wsl is None else is_wsl

    if pname == "windows":
        # Native Windows Python generally has GUI support unless explicitly headless.
        return True

    if pname == "macos":
        # macOS may run headless in CI; check SSH-like headless context conservatively.
        if env_map.get("CI") and not env_map.get("DISPLAY"):
            return False
        return True

    if pname == "linux":
        # X11/Wayland/WSLg hints.
        if env_map.get("DISPLAY") or env_map.get("WAYLAND_DISPLAY"):
            return True
        if wsl and (env_map.get("WSLG_DISPLAY") or env_map.get("WAYLAND_DISPLAY")):
            return True
        return False

    return bool(env_map.get("DISPLAY") or env_map.get("WAYLAND_DISPLAY"))


def get_runtime_info(
    platform_name: str | None = None,
    env: Mapping[str, str] | None = None,
    uname_release: str | None = None,
) -> RuntimeInfo:
    pname = _normalize_platform_name(platform_name or os.sys.platform)
    env_map = os.environ if env is None else env
    is_wsl = detect_wsl(env_map, uname_release=uname_release)
    has_gui = detect_gui_available(platform_name=pname, env=env_map, is_wsl=is_wsl)

    return RuntimeInfo(
        os_name=pname,
        is_windows=pname == "windows",
        is_macos=pname == "macos",
        is_linux=pname == "linux",
        is_wsl=is_wsl,
        has_gui=has_gui,
    )


def default_mpl_config_dir(app_name: str = "aloha_dh_mpl") -> Path:
    return Path(tempfile.gettempdir()) / app_name


def configure_matplotlib_runtime(env: Mapping[str, str] | None = None) -> Path:
    """Ensure MPLCONFIGDIR points to a writable cross-platform temp folder."""
    env_map = os.environ if env is None else env
    if env_map.get("MPLCONFIGDIR"):
        path = Path(env_map["MPLCONFIGDIR"])
    else:
        path = default_mpl_config_dir()
        os.environ["MPLCONFIGDIR"] = str(path)

    path.mkdir(parents=True, exist_ok=True)
    return path
