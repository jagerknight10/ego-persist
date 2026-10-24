"""Create a RunManifest from a project state without changing it."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import re
import socket
import subprocess
import sys
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from egopersist.contracts import RunManifest

PACKAGE_NAMES = (
    "egopersist",
    "numpy",
    "scipy",
    "PyYAML",
    "torch",
    "scikit-learn",
    "matplotlib",
    "pytest",
    "ruff",
)


def sha256_file(path: Path) -> str:
    """Return the SHA-256 digest of a file."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git_value(project_root: Path, *args: str) -> str:
    """Return one Git value, or raise an explicit error."""
    command = ["git", "-C", os.fspath(project_root), *args]
    result = subprocess.run(command, check=False, capture_output=True, text=True)
    if result.returncode != 0:
        message = result.stderr.strip() or result.stdout.strip()
        raise RuntimeError(f"Git command failed: {' '.join(command)}: {message}")
    return result.stdout.strip()


def installed_packages() -> dict[str, str]:
    """Return versions for packages that exist in the environment."""
    packages: dict[str, str] = {}
    for name in PACKAGE_NAMES:
        try:
            version = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            continue
        packages[name] = version
    return packages


def build_manifest(
    *,
    project_root: Path,
    run_id: str,
    stage: str,
    config_path: Path,
    data_split_path: Path,
    label_config_path: Path,
    seed: int,
    container_image: str,
    container_sha256: str,
    command: Sequence[str],
    started_at_utc: datetime | None = None,
    environ: dict[str, str] | None = None,
) -> RunManifest:
    """Build a manifest and verify all input files before hashing."""
    for path, label in (
        (config_path, "config"),
        (data_split_path, "data split"),
        (label_config_path, "label config"),
    ):
        if not path.is_file():
            raise FileNotFoundError(f"{label} is not a file: {path}")
    if not command:
        raise ValueError("command must not be empty")
    if not project_root.is_dir():
        raise NotADirectoryError(f"project root is not a directory: {project_root}")
    if not re.fullmatch(r"[A-Za-z0-9._-]+", run_id) or run_id in {".", ".."}:
        raise ValueError(f"run_id has unsafe characters: {run_id}")
    if len(container_sha256) != 64 or any(
        char not in "0123456789abcdef" for char in container_sha256
    ):
        raise ValueError("container_sha256 must be a lowercase SHA-256 digest")

    environment = os.environ if environ is None else environ
    started = started_at_utc or datetime.now(UTC)
    extra: dict[str, Any] = {
        "stage": stage,
        "container_image": container_image,
        "container_sha256": container_sha256,
        "slurm_job_id": environment.get("SLURM_JOB_ID", ""),
        "slurm_array_job_id": environment.get("SLURM_ARRAY_JOB_ID", ""),
        "slurm_array_task_id": environment.get("SLURM_ARRAY_TASK_ID", ""),
    }
    return RunManifest(
        run_id=run_id,
        git_commit=git_value(project_root, "rev-parse", "HEAD"),
        git_dirty=bool(git_value(project_root, "status", "--porcelain")),
        command=tuple(command),
        config_path=os.fspath(config_path),
        config_hash=sha256_file(config_path),
        data_split_hash=sha256_file(data_split_path),
        label_config_hash=sha256_file(label_config_path),
        seed=seed,
        started_at_utc=started,
        hostname=socket.gethostname(),
        python_version=platform.python_version(),
        packages=installed_packages(),
        extra=extra,
    )


def write_manifest(manifest: RunManifest, output_path: Path) -> None:
    """Write a manifest with owner-only permissions and atomic replacement."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_name(f".{output_path.name}.tmp")
    temporary.write_text(manifest.to_json() + "\n", encoding="utf-8")
    temporary.chmod(0o600)
    temporary.replace(output_path)


def _non_negative_int(value: str) -> int:
    number = int(value)
    if number < 0:
        raise argparse.ArgumentTypeError("value must be non-negative")
    return number


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument(
        "--stage",
        choices=("preprocess", "features", "train", "eval"),
        required=True,
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--data-split", type=Path, required=True)
    parser.add_argument("--label-config", type=Path, required=True)
    parser.add_argument("--seed", type=_non_negative_int, required=True)
    parser.add_argument("--container-image", required=True)
    parser.add_argument("--container-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--command", nargs=argparse.REMAINDER, required=True)
    arguments = parser.parse_args(argv)
    if arguments.command and arguments.command[0] == "--":
        arguments.command = arguments.command[1:]
    if not arguments.command:
        parser.error("--command requires at least one value")
    if len(arguments.container_sha256) != 64:
        parser.error("--container-sha256 must have 64 characters")
    return arguments


def main(argv: Sequence[str] | None = None) -> int:
    """Write the manifest and print its path."""
    arguments = parse_args(argv)
    manifest = build_manifest(
        project_root=arguments.project_root,
        run_id=arguments.run_id,
        stage=arguments.stage,
        config_path=arguments.config,
        data_split_path=arguments.data_split,
        label_config_path=arguments.label_config,
        seed=arguments.seed,
        container_image=arguments.container_image,
        container_sha256=arguments.container_sha256,
        command=arguments.command,
    )
    write_manifest(manifest, arguments.output)
    print(json.dumps({"manifest": os.fspath(arguments.output)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
