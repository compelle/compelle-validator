"""Compelle subnet validator (Bittensor SN82)."""

import os
import subprocess
from importlib.metadata import PackageNotFoundError, version


def _git_sha() -> str:
    """Best-effort short git SHA of the running checkout. Returns 'nogit' if not in a git tree."""
    try:
        return subprocess.check_output(
            ["git", "-C", os.path.dirname(os.path.abspath(__file__)), "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL,
        ).decode().strip()
    except Exception:
        return "nogit"


try:
    __version__ = version("compelle-validator")
except PackageNotFoundError:
    __version__ = "0.1.0"

GIT_SHA = _git_sha()
FULL_VERSION = f"{__version__}+git.{GIT_SHA}" if GIT_SHA != "nogit" else __version__
