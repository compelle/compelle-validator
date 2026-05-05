"""Compelle subnet validator (Bittensor SN82)."""

import os
from importlib.metadata import PackageNotFoundError, version


def _git_sha() -> str:
    """Best-effort short git SHA. Reads .git/HEAD directly so we don't depend on
    the git binary's safe.directory rules under systemd users."""
    git_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".git")
    git_dir = os.path.normpath(git_dir)
    head_path = os.path.join(git_dir, "HEAD")
    try:
        with open(head_path) as f:
            head = f.read().strip()
        if head.startswith("ref: "):
            ref = head[5:]
            ref_path = os.path.join(git_dir, ref)
            if os.path.exists(ref_path):
                with open(ref_path) as f:
                    return f.read().strip()[:7]
            # fallback: packed-refs
            packed = os.path.join(git_dir, "packed-refs")
            if os.path.exists(packed):
                with open(packed) as f:
                    for line in f:
                        if line.endswith(f" {ref}\n") or line.endswith(f" {ref}"):
                            return line.split()[0][:7]
        elif len(head) >= 7:
            return head[:7]
    except Exception:
        pass
    return "nogit"


try:
    __version__ = version("compelle-validator")
except PackageNotFoundError:
    __version__ = "0.1.0"

GIT_SHA = _git_sha()
FULL_VERSION = f"{__version__}+git.{GIT_SHA}" if GIT_SHA != "nogit" else __version__
