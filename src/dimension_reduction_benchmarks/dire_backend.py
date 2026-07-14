"""Load the public :mod:`dire_rapids` API with optional source-checkout pinning."""

from __future__ import annotations

import hashlib
import importlib
import inspect
import subprocess
import sys
from pathlib import Path
from typing import Any


def git_revision(repository: str | Path | None) -> str | None:
    """Return a checkout revision when *repository* is a Git worktree."""
    if repository is None:
        return None
    path = Path(repository).expanduser().resolve()
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=path,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def load_dire(source: str | Path | None = None) -> tuple[Any, dict[str, Any]]:
    """Import ``dire_rapids`` through its public package API.

    Installed-package import is the default. ``source`` is an explicit development and
    historical-reproduction override; when supplied, the imported module must resolve
    inside that checkout.
    """
    source_path = None
    if source is not None:
        source_path = Path(source).expanduser().resolve()
        if not source_path.is_dir():
            raise FileNotFoundError(f"DiRe source checkout does not exist: {source_path}")
        existing = sys.modules.get("dire_rapids")
        if existing is None and str(source_path) not in sys.path:
            sys.path.insert(0, str(source_path))

    module = importlib.import_module("dire_rapids")
    module_path = Path(module.__file__).resolve()
    if source_path is not None and not module_path.is_relative_to(source_path):
        raise RuntimeError(
            f"imported DiRe from {module_path}, outside requested checkout {source_path}"
        )

    return module, {
        "source_checkout": str(source_path) if source_path is not None else None,
        "module_path": str(module_path),
        "version": getattr(module, "__version__", "unknown"),
        "git_revision": git_revision(source_path),
    }


def implementation_identity(obj: Any) -> dict[str, str]:
    """Identify the source implementing one exported DiRe API object."""
    path = Path(inspect.getfile(obj)).resolve()
    return {
        "path": str(path),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }
