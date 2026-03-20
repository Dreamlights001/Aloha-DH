"""Path and output helpers."""

from __future__ import annotations

from pathlib import Path


def repo_root_from_file(file_path: str) -> Path:
    """Resolve repository root when called from examples scripts."""
    return Path(file_path).resolve().parents[1]


def resolve_animation_output_path(
    save_arg: str | None,
    repo_root: Path,
    default_filename: str = "sorting_demo.gif",
    output_dir_name: str = "output",
) -> Path | None:
    """Resolve save target; relative file names go to <repo>/output/."""
    if save_arg is not None and save_arg.strip().lower() in {"", "none", "null"}:
        return None

    chosen = default_filename if save_arg is None else save_arg
    path = Path(chosen)

    if not path.is_absolute() and path.parent == Path("."):
        path = repo_root / output_dir_name / path.name

    path.parent.mkdir(parents=True, exist_ok=True)
    return path
