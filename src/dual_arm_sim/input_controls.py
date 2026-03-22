"""Input helpers for keyboard/mouse/gamepad control."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, Set, Tuple


@dataclass
class GamepadState:
    connected: bool
    device_name: str
    left_stick: Tuple[float, float]
    right_stick: Tuple[float, float]
    left_trigger: float
    right_trigger: float
    dpad: Tuple[int, int]
    buttons_down: Set[str]
    buttons_triggered: Set[str]


class GamepadInputManager:
    """Poll gamepad input via pygame with graceful fallback."""

    BUTTON_MAP: Dict[int, str] = {
        0: "a",
        1: "b",
        2: "x",
        3: "y",
        4: "lb",
        5: "rb",
        6: "back",
        7: "start",
        8: "ls",
        9: "rs",
    }

    def __init__(self, enabled: bool = True, deadzone: float = 0.18) -> None:
        self.enabled = enabled
        self.deadzone = deadzone
        self._pygame = None
        self._joystick = None
        self._initialized = False
        self._prev_buttons_down: Set[str] = set()

    @staticmethod
    def _apply_deadzone(v: float, deadzone: float) -> float:
        if abs(v) < deadzone:
            return 0.0
        sign = 1.0 if v >= 0 else -1.0
        scaled = (abs(v) - deadzone) / max(1e-9, 1.0 - deadzone)
        return sign * scaled

    @staticmethod
    def _trigger_norm(v: float) -> float:
        # Support both [-1, 1] and [0, 1] trigger ranges.
        if v < 0.0:
            return max(0.0, min(1.0, 0.5 * (v + 1.0)))
        return max(0.0, min(1.0, v))

    def _init_backend(self) -> None:
        if self._initialized:
            return
        self._initialized = True

        if not self.enabled:
            return

        try:
            # Avoid pygame startup prompt in terminal.
            os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
            import pygame  # type: ignore
        except Exception:
            self._pygame = None
            return

        try:
            pygame.init()
            pygame.joystick.init()
            self._pygame = pygame
            self._refresh_joystick()
        except Exception:
            self._pygame = None
            self._joystick = None

    def _refresh_joystick(self) -> None:
        if self._pygame is None:
            self._joystick = None
            return
        try:
            self._pygame.joystick.quit()
            self._pygame.joystick.init()
            count = self._pygame.joystick.get_count()
            if count <= 0:
                self._joystick = None
                return
            js = self._pygame.joystick.Joystick(0)
            js.init()
            self._joystick = js
        except Exception:
            self._joystick = None

    def poll(self) -> GamepadState:
        self._init_backend()
        if self._pygame is None:
            self._prev_buttons_down = set()
            return GamepadState(
                connected=False,
                device_name="",
                left_stick=(0.0, 0.0),
                right_stick=(0.0, 0.0),
                left_trigger=0.0,
                right_trigger=0.0,
                dpad=(0, 0),
                buttons_down=set(),
                buttons_triggered=set(),
            )

        try:
            self._pygame.event.pump()
        except Exception:
            self._refresh_joystick()

        if self._joystick is None:
            self._refresh_joystick()

        if self._joystick is None:
            self._prev_buttons_down = set()
            return GamepadState(
                connected=False,
                device_name="",
                left_stick=(0.0, 0.0),
                right_stick=(0.0, 0.0),
                left_trigger=0.0,
                right_trigger=0.0,
                dpad=(0, 0),
                buttons_down=set(),
                buttons_triggered=set(),
            )

        js = self._joystick

        def axis(idx: int) -> float:
            try:
                if idx < js.get_numaxes():
                    return float(js.get_axis(idx))
            except Exception:
                return 0.0
            return 0.0

        left_x = self._apply_deadzone(axis(0), self.deadzone)
        left_y = self._apply_deadzone(axis(1), self.deadzone)
        right_x = self._apply_deadzone(axis(2), self.deadzone)
        right_y = self._apply_deadzone(axis(3), self.deadzone)
        lt = self._trigger_norm(axis(4))
        rt = self._trigger_norm(axis(5))

        try:
            hats = js.get_numhats()
            dpad = js.get_hat(0) if hats > 0 else (0, 0)
        except Exception:
            dpad = (0, 0)
        dpad_xy = (int(dpad[0]), int(dpad[1]))

        buttons_down: Set[str] = set()
        try:
            button_count = js.get_numbuttons()
            for i in range(button_count):
                if js.get_button(i):
                    buttons_down.add(self.BUTTON_MAP.get(i, f"btn_{i}"))
        except Exception:
            buttons_down = set()

        buttons_triggered = buttons_down - self._prev_buttons_down
        self._prev_buttons_down = set(buttons_down)

        try:
            device_name = str(js.get_name())
        except Exception:
            device_name = "Gamepad"

        return GamepadState(
            connected=True,
            device_name=device_name,
            left_stick=(left_x, left_y),
            right_stick=(right_x, right_y),
            left_trigger=lt,
            right_trigger=rt,
            dpad=dpad_xy,
            buttons_down=buttons_down,
            buttons_triggered=buttons_triggered,
        )
