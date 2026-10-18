"""What the build in progress is, kept on disk, so a build that took the
whole process down with it (out of memory, or stuck) is known afterwards and
recorded as failed instead of being tried again and again.

Each build runs in a process of its own (lab_job.py). Before a build it
marks itself (mark); when it ends it clears the mark (clear). If its process
is killed, the mark is still there: the worker then stops that build for
good (stop), and the next time it comes round it is recorded as failed
without building (stopped).
"""
import hashlib
import json
import os
import tempfile
import time
from pathlib import Path
from typing import Optional


def _root() -> Path:
    # (the same place as the Lab's runs: lab_api._lab_root)
    env = os.environ.get("ANATOMY3D_LAB_DIR")
    if env:
        root = Path(env)
    elif Path("/data").is_dir() and os.access("/data", os.W_OK):
        root = Path("/data/lab")
    else:
        root = Path(tempfile.gettempdir()) / "anatomy3d-lab"
    root.mkdir(parents=True, exist_ok=True)
    return root


_JOB = _root() / "_job.json"
_STOPPED = _root() / "_stopped.json"


def _read(path: Path, default):
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return default


def _write(path: Path, data) -> None:
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data))
    tmp.replace(path)


def digest(values) -> str:
    """A short name for a set of values (a study's move)."""
    return hashlib.md5(json.dumps(values, sort_keys=True, default=str).encode()).hexdigest()[:10]


def mark(key: str) -> None:
    _write(_JOB, {"key": key, "at": time.time()})


def clear() -> None:
    try:
        _JOB.unlink()
    except FileNotFoundError:
        pass


def current() -> Optional[str]:
    return (_read(_JOB, {}) or {}).get("key")


def stop(key: str, why: str) -> None:
    stopped = _read(_STOPPED, {})
    stopped[key] = {"why": why, "at": time.time()}
    _write(_STOPPED, stopped)


def stopped(key: str) -> Optional[str]:
    """Why this build was stopped for good, or None."""
    s = _read(_STOPPED, {}).get(key)
    return s["why"] if s else None
