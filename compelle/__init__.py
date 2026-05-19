"""Compelle subnet validator (Bittensor SN82)."""

import hashlib
import os
import pathlib
from importlib.metadata import PackageNotFoundError, version


def _source_hash() -> str:
    """SHA1 of all compelle/*.py source files, first 7 hex chars. Independent
    of pip metadata and .git readability — proves what code is actually loaded.

    Useful when operators do `git pull && systemctl restart` without
    `pip install -e .` (importlib.metadata returns stale package version) or
    when .git/HEAD isn't readable (tarball installs, systemd user perms).
    """
    pkg_dir = os.path.dirname(os.path.abspath(__file__))
    h = hashlib.sha1()
    try:
        for p in sorted(pathlib.Path(pkg_dir).glob("*.py")):
            with open(p, "rb") as f:
                # include filename as part of the hash so renaming a file
                # changes the hash even if content is identical
                h.update(p.name.encode() + b"\0" + f.read() + b"\0")
        return h.hexdigest()[:7]
    except Exception:
        return ""


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
SRC_HASH = _source_hash()
FULL_VERSION = __version__
if GIT_SHA != "nogit":
    FULL_VERSION += f"+git.{GIT_SHA}"
if SRC_HASH:
    FULL_VERSION += f"+src.{SRC_HASH}"
