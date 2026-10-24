"""RunManifest construction: reproducibility record for every stage."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import socket
import sys
from datetime import datetime, timezone
from pathlib import Path

from egopersist.contracts import RunManifest


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_file(path: str | Path) -> str:
    return _sha256_bytes(Path(path).read_bytes())


def _git_state(root: str | Path) -> tuple[str, bool]:
    import subprocess

    root = Path(root)

    def run(*args: str) -> str:
        try:
            return subprocess.run(
                ["git", *args],
                cwd=root,
                capture_output=True,
                text=True,
                timeout=10,
            ).stdout.strip()
        except Exception:
            return ""

    commit = run("rev-parse", "HEAD") or "unknown0"
    dirty = bool(run("status", "--porcelain"))
    return commit[:40], dirty


def build_run_manifest(
    run_id: str,
    config_path: str | Path,
    split_file: str | Path,
    label_config_path: str | Path,
    project_root: str | Path,
    extra: dict | None = None,
) -> RunManifest:
    """Build a manifest from the environment and on-disk artifacts."""
    packages: dict[str, str] = {}
    for name in ("numpy", "scipy", "torch", "yaml"):
        try:
            module = __import__(name)
            packages[name] = getattr(module, "__version__", "unknown")
        except Exception:
            pass
    command = tuple(sys.argv) or ("python", "-m", "egopersist.evaluation.cli")
    return RunManifest(
        run_id=run_id,
        git_commit=_git_state(project_root)[0],
        git_dirty=_git_state(project_root)[1],
        command=command,
        config_path=str(config_path),
        config_hash=_sha256_file(config_path),
        data_split_hash=_sha256_file(split_file),
        label_config_hash=_sha256_file(label_config_path),
        seed=int(os.environ.get("EGOPERSIST_SEED", "0")),
        started_at_utc=datetime.now(timezone.utc),
        hostname=socket.gethostname(),
        python_version=platform.python_version(),
        packages=packages,
        extra=extra or {},
    )
